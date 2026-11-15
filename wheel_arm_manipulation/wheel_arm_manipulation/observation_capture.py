"""Save timestamp-matched RGB/joint observations without commanding motion."""

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import signal
import time
from uuid import uuid4

import cv2
from control_msgs.msg import JointTrajectoryControllerState
from cv_bridge import CvBridge, CvBridgeError
import message_filters
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data
from rclpy.signals import SignalHandlerOptions
from rclpy.utilities import remove_ros_args
from sensor_msgs.msg import Image, JointState

from wheel_arm_manipulation.controller_reference import (
    ARM_JOINTS, FINGER_JOINTS, ControllerReferenceRecorder,
)
from wheel_arm_manipulation.image_qos import image_qos

MAX_SKEW_NS = 50_000_000
MAX_AGE_NS = 500_000_000


def stamp_ns(message):
    stamp = message.header.stamp
    return stamp.sec*1_000_000_000+stamp.nanosec


def extract_state(joints):
    if len(joints.name) != len(joints.position) or len(set(joints.name)) != len(joints.name):
        raise ValueError("Joint names/positions must match and names must be unique")
    positions = dict(zip(joints.name, joints.position))
    required = ARM_JOINTS+FINGER_JOINTS
    if any(name not in positions for name in required):
        raise ValueError("Required arm or finger joint is missing")
    if not all(math.isfinite(positions[name]) for name in required):
        raise ValueError("Required joint positions must be finite")
    fingers = [positions[name] for name in FINGER_JOINTS]
    if any(value < -1e-6 or value > .040001 for value in fingers):
        raise ValueError("Finger position is outside the model's 0..0.04 m range")
    return [positions[name] for name in ARM_JOINTS]+[sum(fingers)]


def observation(rgb, joints, now_ns):
    image_stamp, joint_stamp = stamp_ns(rgb), stamp_ns(joints)
    if min(image_stamp, joint_stamp) <= 0 or now_ns <= 0:
        raise ValueError("Waiting for positive ROS timestamps and /clock")
    if any(not 0 <= now_ns-stamp <= MAX_AGE_NS for stamp in (image_stamp, joint_stamp)):
        raise ValueError("Image or joints are stale, future-dated, or use a different clock")
    if abs(image_stamp-joint_stamp) > MAX_SKEW_NS:
        raise ValueError("Image/joint timestamp difference exceeds 50 ms")
    if rgb.width <= 0 or rgb.height <= 0 or not rgb.header.frame_id:
        raise ValueError("Image dimensions or frame_id are empty")
    return {
        "schema_version": 1,
        "kind": "observation_only",
        "image": {"file": "rgb.png", "stamp_ns": image_stamp,
                  "frame_id": rgb.header.frame_id, "width": rgb.width, "height": rgb.height,
                  "source_encoding": rgb.encoding},
        "joints_stamp_ns": joint_stamp,
        "image_age_ms": (now_ns-image_stamp)/1_000_000,
        "joints_age_ms": (now_ns-joint_stamp)/1_000_000,
        "timestamp_difference_ms": abs(image_stamp-joint_stamp)/1_000_000,
        "state": {"names": [*ARM_JOINTS, "gripper_width"],
                  "units": ["rad", "rad", "rad", "rad", "m"], "values": extract_state(joints)},
        "finger_positions_m": {name: joints.position[joints.name.index(name)] for name in FINGER_JOINTS},
    }


def encode_image(bgr, metadata):
    expected = (metadata["image"]["height"], metadata["image"]["width"], 3)
    if bgr.shape != expected or bgr.dtype.name != "uint8":
        raise ValueError("Converted image must be uint8 BGR with the declared dimensions")
    encoded, data = cv2.imencode(".png", bgr)
    if not encoded:
        raise OSError("Could not encode RGB image")
    return data.tobytes()


def capture_directory(root):
    label = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = Path(root).expanduser()/f"{label}-{uuid4().hex[:12]}"
    destination.mkdir(parents=True, exist_ok=False)
    return destination


def save_observation(root, bgr, metadata):
    data = encode_image(bgr, metadata)
    document = json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False)+"\n"
    destination = capture_directory(root)
    (destination/"rgb.png").write_bytes(data)
    # Metadata is written last; a directory without it is not a complete sample.
    (destination/"observation.json").write_text(document, encoding="utf-8")
    return destination


class CaptureDiagnostics:
    """Count callbacks and time completed operations on the recorder's thread."""

    def __init__(self, rgb_reliability=None):
        self.rgb_reliability = rgb_reliability
        self.pair_ages = {name: {"count": 0, "total_ms": 0., "max_ms": 0.}
                          for name in ("image_age_ms", "joints_age_ms")}
        self.counts = {name: 0 for name in (
            "rgb_received", "joint_states_received", "paired", "invalid_pairs",
            "sampling_skipped", "conversion_failed", "frames_saved")}
        self.timings = {name: {"count": 0, "total_ns": 0, "max_ns": 0} for name in (
            "image_conversion", "png_encoding", "image_file_write", "jsonl_write")}
        self.rgb_first_arrival = self.rgb_last_arrival = None
        self.rgb_max_arrival_gap_ns = 0

    def received_rgb(self, message):
        now = time.monotonic_ns()
        if self.rgb_last_arrival is not None:
            self.rgb_max_arrival_gap_ns = max(self.rgb_max_arrival_gap_ns, now-self.rgb_last_arrival)
        else:
            self.rgb_first_arrival = now
        self.rgb_last_arrival = now
        self.counts["rgb_received"] += 1

    def received_joints(self, message):
        self.counts["joint_states_received"] += 1

    def record_time(self, name, started_ns):
        elapsed = time.perf_counter_ns()-started_ns
        stats = self.timings[name]
        stats["count"] += 1
        stats["total_ns"] += elapsed
        stats["max_ns"] = max(stats["max_ns"], elapsed)

    def valid_pair(self, metadata):
        for name, stats in self.pair_ages.items():
            stats["count"] += 1
            stats["total_ms"] += metadata[name]
            stats["max_ms"] = max(stats["max_ms"], metadata[name])

    def report(self):
        span = (self.rgb_last_arrival-self.rgb_first_arrival)/1e9 if self.counts["rgb_received"] > 1 else 0.
        timings = {}
        for name, values in self.timings.items():
            count = values["count"]
            timings[name] = {"count": count,
                             "mean_ms": values["total_ns"]/count/1e6 if count else None,
                             "max_ms": values["max_ns"]/1e6 if count else None}
        report = {"counts": dict(self.counts), "timings": timings,
                "rgb_callback_wall_span_seconds": span,
                "rgb_callback_wall_hz": (self.counts["rgb_received"]-1)/span if span else None,
                "rgb_max_callback_wall_gap_ms": self.rgb_max_arrival_gap_ns/1e6
                if self.counts["rgb_received"] > 1 else None}
        report["valid_pair_age_relative_to_node_ros_clock"] = {
            name: {"count": stats["count"],
                   "mean_ms": stats["total_ms"]/stats["count"] if stats["count"] else None,
                   "max_ms": stats["max_ms"] if stats["count"] else None}
            for name, stats in self.pair_ages.items()}
        if self.rgb_reliability is not None:
            report["rgb_subscription_qos"] = {"reliability": self.rgb_reliability,
                                               "history": "keep_last", "depth": 5, "durability": "volatile"}
        return report


class EpisodeRecorder:
    def __init__(self, root, hz, diagnostics=None):
        if not math.isfinite(hz) or not 0 < hz <= 100:
            raise ValueError("Recording frequency must be in (0, 100] Hz")
        self.hz = hz
        self.period_ns = round(1_000_000_000/hz)
        self.path = capture_directory(root)
        (self.path/"images").mkdir()
        self.stream = (self.path/"frames.jsonl").open("w", encoding="utf-8", buffering=1)
        self.count = 0
        self.first_stamp = self.last_stamp = self.next_stamp = self.seen_stamp = None
        self.max_gap_ns = 0
        self.started = time.monotonic()
        self.diagnostics = diagnostics if diagnostics is not None else CaptureDiagnostics()
        self.references = None

    def due(self, stamp):
        if self.seen_stamp is not None and stamp < self.seen_stamp:
            raise ValueError("Image clock moved backwards; start a new episode after resetting simulation")
        duplicate = stamp == self.seen_stamp
        self.seen_stamp = stamp
        return not duplicate and (self.next_stamp is None or stamp >= self.next_stamp)

    def append(self, bgr, metadata):
        stamp = metadata["image"]["stamp_ns"]
        if self.last_stamp is not None and stamp <= self.last_stamp:
            raise ValueError("Recorded image timestamps must increase")
        filename = f"images/{self.count:06d}.png"
        frame = {**metadata, "frame_index": self.count, "image": {**metadata["image"], "file": filename}}
        document = json.dumps(frame, ensure_ascii=False, allow_nan=False)+"\n"
        started = time.perf_counter_ns()
        data = encode_image(bgr, metadata)
        self.diagnostics.record_time("png_encoding", started)
        started = time.perf_counter_ns()
        (self.path/filename).write_bytes(data)
        self.diagnostics.record_time("image_file_write", started)
        started = time.perf_counter_ns()
        self.stream.write(document)
        self.diagnostics.record_time("jsonl_write", started)
        if self.first_stamp is None:
            self.first_stamp = stamp
            self.next_stamp = stamp
        else:
            self.max_gap_ns = max(self.max_gap_ns, stamp-self.last_stamp)
        # Skip missed slots rather than duplicating a frame to fill a nominal rate.
        self.next_stamp += ((stamp-self.next_stamp)//self.period_ns+1)*self.period_ns
        self.last_stamp = stamp
        self.count += 1
        self.diagnostics.counts["frames_saved"] += 1

    def close(self, reason, failed=False):
        self.stream.close()
        references = self.references.close() if self.references is not None else None
        if references is not None and not references["both_controllers_recorded"]:
            failed = True
        span = 0. if self.count < 2 else (self.last_stamp-self.first_stamp)/1e9
        summary = {"schema_version": 1, "kind": "observation_only", "frames": self.count,
                   "collection_status": "failed" if failed else "finished", "end_reason": reason,
                   "target_hz": self.hz, "sample_span_sim_seconds": span,
                   "wall_duration_seconds": time.monotonic()-self.started,
                   "actual_hz": (self.count-1)/span if span else None,
                   "max_frame_gap_sim_seconds": self.max_gap_ns/1e9,
                   "frames_file": "frames.jsonl", "diagnostics": self.diagnostics.report()}
        if references is not None:
            summary["kind"] = "observations_with_controller_references"
            summary["controller_references"] = references
        (self.path/"summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
        return summary


class ObservationCapture(Node):
    def __init__(self, options):
        super().__init__("observation_capture", parameter_overrides=[Parameter("use_sim_time", value=True)])
        self.options = options
        self.bridge = CvBridge()
        self.result = None
        self.recorder = None
        self.diagnostics = CaptureDiagnostics(options.rgb_reliability)
        self.last_saved = time.monotonic()
        self.reason = "Waiting for RGB and joint states with matching timestamps"
        self.subscriptions_ = [
            message_filters.Subscriber(self, Image, options.rgb_topic,
                                       qos_profile=image_qos(options.rgb_reliability)),
            message_filters.Subscriber(self, JointState, options.joint_states_topic,
                                       qos_profile=qos_profile_sensor_data),
        ]
        # Count only messages delivered to this node, before they enter the synchronizer.
        self.subscriptions_[0].registerCallback(self.diagnostics.received_rgb)
        self.subscriptions_[1].registerCallback(self.diagnostics.received_joints)
        self.sync = message_filters.ApproximateTimeSynchronizer(self.subscriptions_, queue_size=50, slop=.05)
        self.sync.registerCallback(self.receive)
        self.reference_topics = {"arm": options.arm_reference_topic, "gripper": options.gripper_reference_topic}
        self.reference_subscriptions = []
        if options.record_references:
            for group, topic in self.reference_topics.items():
                self.reference_subscriptions.append(self.create_subscription(
                    JointTrajectoryControllerState, topic,
                    lambda message, group=group: self.receive_reference(message, group), qos_profile_sensor_data))

    def receive_reference(self, message, group):
        if self.recorder is not None and self.recorder.references is not None:
            clock = "ros_sim_time" if self.get_parameter("use_sim_time").value else "ros_time"
            self.recorder.references.append(message, group, self.get_clock().now().nanoseconds, clock)

    def receive(self, rgb, joints):
        if self.result is not None and not self.options.record:
            return
        self.diagnostics.counts["paired"] += 1
        try:
            metadata = observation(rgb, joints, self.get_clock().now().nanoseconds)
        except ValueError as error:
            self.diagnostics.counts["invalid_pairs"] += 1
            self.reason = str(error)
            return
        self.diagnostics.valid_pair(metadata)
        if self.recorder is not None and not self.recorder.due(stamp_ns(rgb)):
            self.diagnostics.counts["sampling_skipped"] += 1
            return
        try:
            started = time.perf_counter_ns()
            bgr = self.bridge.imgmsg_to_cv2(rgb, "bgr8")
            self.diagnostics.record_time("image_conversion", started)
        except CvBridgeError as error:
            self.diagnostics.counts["conversion_failed"] += 1
            self.reason = str(error)
            return
        metadata.update({
            "clock": "ros_sim_time" if self.get_parameter("use_sim_time").value else "ros_time",
            "instruction": self.options.instruction,
            "topics": {"rgb": self.options.rgb_topic, "joints": self.options.joint_states_topic},
            "rgb_subscription_reliability": getattr(self.options, "rgb_reliability", "best_effort"),
        })
        if self.options.record:
            if self.recorder is None:
                self.recorder = EpisodeRecorder(self.options.output_dir, self.options.hz, self.diagnostics)
                if self.options.record_references:
                    self.recorder.references = ControllerReferenceRecorder(self.recorder.path, self.reference_topics)
                self.recorder.due(stamp_ns(rgb))
                print(f"[记录目录] {self.recorder.path.resolve()}", flush=True)
            self.recorder.append(bgr, metadata)
            self.last_saved = time.monotonic()
            return
        self.result = save_observation(self.options.output_dir, bgr, metadata)
        print(f"[保存] {self.result.resolve()}", flush=True)
        print(f"[状态] {metadata['state']['values']}；时间差 "
              f"{metadata['timestamp_difference_ms']:.1f} ms", flush=True)


def main(args=None):
    parser = argparse.ArgumentParser(description="只读采集 RGB 与关节状态；不发送控制指令")
    parser.add_argument("--rgb-topic", default="/camera/image")
    parser.add_argument("--rgb-reliability", choices=("best_effort", "reliable"), default="best_effort",
                        help="仅 RGB 订阅可靠性，队列深度均为 5；默认 best_effort")
    parser.add_argument("--joint-states-topic", default="/wheel_arm/joint_states")
    parser.add_argument("--output-dir", default="observations/vla")
    parser.add_argument("--instruction", default="拿起桌上的杯子")
    parser.add_argument("--record", action="store_true", help="连续记录，Ctrl+C 正常结束；默认仍为单帧")
    parser.add_argument("--record-references", action="store_true",
                        help="配合 --record 保存控制器插值参考目标，尚未对齐为训练动作")
    parser.add_argument("--arm-reference-topic", default="/wheel_arm/arm_controller/controller_state")
    parser.add_argument("--gripper-reference-topic", default="/wheel_arm/gripper_controller/controller_state")
    parser.add_argument("--hz", type=float, default=10., help="连续记录目标频率，按图像仿真时间采样")
    parser.add_argument("--duration", type=float, help="连续记录的墙钟时长上限，秒；默认不限")
    parser.add_argument("--timeout", type=float, default=15., help="等待首帧或无新帧的墙钟超时，秒")
    options = parser.parse_args(remove_ros_args(args=args)[1:])
    if not math.isfinite(options.timeout) or options.timeout <= 0:
        parser.error("--timeout must be positive and finite")
    if options.record_references and not options.record:
        parser.error("--record-references requires --record")
    if not math.isfinite(options.hz) or not 0 < options.hz <= 100:
        parser.error("--hz must be in (0, 100]")
    if options.duration is not None and (not options.record or not math.isfinite(options.duration)
                                         or options.duration <= 0):
        parser.error("--duration requires --record and a positive finite value")
    # Let an in-progress frame finish writing before handling Ctrl+C in the loop.
    stopping = False

    def request_stop(signum, frame):
        nonlocal stopping
        stopping = True

    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    previous_sigint = signal.signal(signal.SIGINT, request_stop)
    node = None
    code = 0
    reason = "completed"
    try:
        node = ObservationCapture(options)
        print("[采集] 等待一组新鲜 RGB 和关节状态；默认使用仿真时间，最大时间差 50 ms。", flush=True)
        print(f"[RGB订阅] {options.rgb_reliability}，队列深度 5。", flush=True)
        if options.record:
            print(f"[连续记录] 目标 {options.hz:g} Hz，Ctrl+C 结束；未生成训练动作或任务成功标签。", flush=True)
        if options.record_references:
            print("[参考目标] 保存控制器当前插值位置参考，按原始时间戳记录；尚未与图像对齐。", flush=True)
        started = time.monotonic()
        while rclpy.ok() and node.result is None and not stopping:
            if options.duration is not None and time.monotonic()-started >= options.duration:
                reason = "duration_reached"
                break
            if time.monotonic()-node.last_saved >= options.timeout:
                reason, code = "data_timeout", 1
                break
            rclpy.spin_once(node, timeout_sec=.05)
        if stopping:
            reason = "ctrl_c"
            if not options.record:
                code = 130
        if node.result is None and (not options.record or node.recorder is None) and not stopping:
            code = 1
        if code == 1:
            print(f"[失败] 记录未正常完成：{node.reason}", flush=True)
    except KeyboardInterrupt:
        reason, code = "interrupted", 130
    except (OSError, ValueError, cv2.error) as error:
        print(f"[失败] {error}", flush=True)
        reason, code = "error: "+str(error), 1
    finally:
        if node is not None:
            if node.recorder is not None:
                try:
                    summary = node.recorder.close(reason, failed=code != 0)
                    hz = summary["actual_hz"]
                    rate = "不足两帧" if hz is None else f"{hz:.2f} Hz"
                    print(f"[结束] {summary['frames']} 帧，仿真采样跨度 "
                          f"{summary['sample_span_sim_seconds']:.2f} s，实际频率 {rate}；{reason}", flush=True)
                    if "controller_references" in summary:
                        refs = summary["controller_references"]
                        print(f"[参考目标] 保存机械臂 {refs['saved']['arm']} 条，夹爪 {refs['saved']['gripper']} 条；"
                              f"拒绝机械臂 {refs['rejected']['arm']} 条，夹爪 {refs['rejected']['gripper']} 条。", flush=True)
                        if not refs["both_controllers_recorded"]:
                            print("[失败] 缺少机械臂或夹爪参考目标，请检查 controller_state 话题及控制器状态。", flush=True)
                            code = 1
                    diag = summary["diagnostics"]
                    counts = diag["counts"]
                    print(f"[有效配对年龄/本节点ROS时钟] "
                          f"{diag['valid_pair_age_relative_to_node_ros_clock']}", flush=True)
                    print(f"[计数] 收到图像 {counts['rgb_received']}，关节 {counts['joint_states_received']}，"
                          f"配对 {counts['paired']}，无效 {counts['invalid_pairs']}，"
                          f"采样跳过 {counts['sampling_skipped']}，转换失败 {counts['conversion_failed']}，"
                          f"保存 {counts['frames_saved']}", flush=True)
                    for name, label in (("image_conversion", "图像转换"), ("png_encoding", "PNG编码"),
                                        ("image_file_write", "图片写入"), ("jsonl_write", "状态写入")):
                        timing = diag["timings"][name]
                        if timing["count"]:
                            print(f"[耗时] {label}: 平均 {timing['mean_ms']:.2f} ms，"
                                  f"最大 {timing['max_ms']:.2f} ms", flush=True)
                except OSError as error:
                    print(f"[失败] 无法保存记录统计：{error}", flush=True)
                    code = 1
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        signal.signal(signal.SIGINT, previous_sigint)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
