"""Capture one fresh RGB-D observation with transforms at the image timestamp."""

import argparse
import math
import signal
import sys
import time

from wheel_arm_navigation.mission import utc_now
from wheel_arm_navigation.observation import depth_meters, save_observation


def positive(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("必须是大于零的有限数值")
    return number


def parser():
    result = argparse.ArgumentParser(description="保存同步 RGB-D、内参和拍摄时刻 TF，不移动机器人")
    result.add_argument("--output-dir", default="observations")
    result.add_argument("--label", default="", help="人工备注，不作为物体识别结果")
    result.add_argument("--rgb-topic", default="/camera/image")
    result.add_argument("--depth-topic", default="/camera/depth_image")
    result.add_argument("--info-topic", default="/camera/camera_info")
    result.add_argument("--reference-frame", default="map")
    result.add_argument("--base-frame", default="base_footprint")
    result.add_argument("--camera-frame", default="camera_optical_frame")
    result.add_argument("--wait", type=positive, default=15)
    result.add_argument("--max-age", type=positive, default=0.5)
    result.add_argument("--sync-slop", type=positive, default=0.02)
    return result


def stamp_ns(message):
    return message.header.stamp.sec * 1000000000 + message.header.stamp.nanosec


def validate_headers(rgb, depth, info, now_ns, started_ns, options):
    stamps = [stamp_ns(message) for message in (rgb, depth, info)]
    if (now_ns <= 0 or min(stamps) <= 0 or min(stamps) < started_ns
            or any(not 0 <= (now_ns - stamp) / 1e9 <= options.max_age for stamp in stamps)):
        raise ValueError("图像过期、早于采集请求或与 ROS 时钟不一致")
    if (max(stamps) - min(stamps)) / 1e9 > options.sync_slop:
        raise ValueError("RGB、深度和内参时间戳不同步")
    if any(message.header.frame_id != options.camera_frame for message in (rgb, depth, info)):
        raise ValueError("图像/内参 frame_id 与预期光学坐标系不一致")


def transform_record(message):
    transform = message.transform
    return {"parent_frame": message.header.frame_id, "child_frame": message.child_frame_id,
            "stamp": {"sec": message.header.stamp.sec, "nanosec": message.header.stamp.nanosec},
            "translation": {key: float(getattr(transform.translation, key)) for key in ("x", "y", "z")},
            "rotation": {key: float(getattr(transform.rotation, key)) for key in ("x", "y", "z", "w")}}


def capture(options, argv):
    import message_filters
    import rclpy
    from cv_bridge import CvBridge, CvBridgeError
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from rclpy.signals import SignalHandlerOptions
    from rclpy.time import Time
    from sensor_msgs.msg import CameraInfo, Image
    from tf2_ros import Buffer, TransformException, TransformListener

    node = None
    handlers = {}
    try:
        rclpy.init(args=argv, signal_handler_options=SignalHandlerOptions.NO)
        node = Node("scene_snapshot")
        state = {"stopping": False, "started_ns": None, "sample": None, "reason": "等待相机和 /clock"}
        for signum in (signal.SIGINT, signal.SIGTERM):
            handlers[signum] = signal.signal(signum, lambda *_: state.update(stopping=True))
        buffer = Buffer()
        listener = TransformListener(buffer, node)
        bridge = CvBridge()
        subscriptions = [
            message_filters.Subscriber(node, kind, topic, qos_profile=qos_profile_sensor_data)
            for kind, topic in ((Image, options.rgb_topic), (Image, options.depth_topic),
                                (CameraInfo, options.info_topic))]
        synchronizer = message_filters.ApproximateTimeSynchronizer(
            subscriptions, queue_size=10, slop=options.sync_slop)

        def received(rgb, depth, info):
            if state["started_ns"] is None:
                return
            try:
                validate_headers(rgb, depth, info, node.get_clock().now().nanoseconds,
                                 state["started_ns"], options)
                state["sample"] = (rgb, depth, info)
            except ValueError as error:
                state["reason"] = str(error)

        synchronizer.registerCallback(received)
        deadline = time.monotonic() + options.wait
        node.get_logger().info("等待新鲜同步 RGB-D 和拍摄时刻 TF；不会发送移动指令")
        while rclpy.ok() and not state["stopping"] and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.05)
            now_ns = node.get_clock().now().nanoseconds
            if state["started_ns"] is None and now_ns > 0:
                state["started_ns"] = now_ns
            if state["sample"] is None:
                continue
            rgb, depth, info = state["sample"]
            try:
                validate_headers(rgb, depth, info, now_ns, state["started_ns"], options)
                stamp = Time.from_msg(rgb.header.stamp)
                base_tf = buffer.lookup_transform(options.reference_frame, options.base_frame, stamp)
                camera_tf = buffer.lookup_transform(options.reference_frame, options.camera_frame, stamp)
            except (TransformException, ValueError) as error:
                state["reason"] = str(error)
                continue
            rgb_array = bridge.imgmsg_to_cv2(rgb, desired_encoding="bgr8")
            depth_array = depth_meters(bridge.imgmsg_to_cv2(depth, desired_encoding="passthrough"), depth.encoding)
            metadata = {
                "captured_at": utc_now(), "label": options.label,
                "reference_frame": options.reference_frame, "camera_frame": options.camera_frame,
                "topics": {"rgb": options.rgb_topic, "depth": options.depth_topic, "camera_info": options.info_topic},
                "stamps": {name: {"sec": message.header.stamp.sec, "nanosec": message.header.stamp.nanosec}
                           for name, message in (("rgb", rgb), ("depth", depth), ("camera_info", info))},
                "source_encodings": {"rgb": rgb.encoding, "depth": depth.encoding},
                "sync_span_sec": (max(stamp_ns(m) for m in (rgb, depth, info))
                                  - min(stamp_ns(m) for m in (rgb, depth, info))) / 1e9,
                "image_age_sec": (now_ns - stamp_ns(rgb)) / 1e9,
                "camera_info": {"width": info.width, "height": info.height,
                                "distortion_model": info.distortion_model,
                                **{key: list(getattr(info, key)) for key in ("k", "d", "r", "p")}},
                "transforms": {"reference_to_base": transform_record(base_tf),
                               "reference_to_camera": transform_record(camera_tf)},
            }
            path = save_observation(options.output_dir, rgb_array, depth_array, metadata)
            node.get_logger().info(f"观测已保存：{path}")
            return 0
        if state["stopping"]:
            return 130
        raise RuntimeError(f"采集超时：{state['reason']}；请检查相机、/clock、定位和 TF")
    except CvBridgeError as error:
        raise ValueError(f"图像转换失败：{error}") from error
    finally:
        for signum, handler in handlers.items():
            signal.signal(signum, handler)
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def main(args=None):
    from rclpy.utilities import remove_ros_args

    argv = sys.argv if args is None else ["scene_snapshot", *args]
    options = parser().parse_args(remove_ros_args(argv)[1:])
    try:
        return capture(options, argv)
    except (OSError, ValueError, RuntimeError) as error:
        print(f"采集错误：{error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130
