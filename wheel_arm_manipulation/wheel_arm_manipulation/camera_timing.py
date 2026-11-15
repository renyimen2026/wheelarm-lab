"""Read-only RGB timing probe; buffer small metadata rows, never decode pixels."""

from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time
from uuid import uuid4

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import Image

from wheel_arm_manipulation.image_qos import image_qos as timing_qos


def gap_statistics(values):
    if not values:
        return {"count": 0, "mean_ms": None, "p95_ms": None, "max_ms": None,
                "over_100ms": 0, "over_250ms": 0}
    ordered = sorted(values)
    return {"count": len(values), "mean_ms": sum(values)/len(values)/1e6,
            "p95_ms": ordered[math.ceil(.95*len(values))-1]/1e6, "max_ms": ordered[-1]/1e6,
            "over_100ms": sum(v > 100_000_000 for v in values),
            "over_250ms": sum(v > 250_000_000 for v in values)}


class TimingDiagnostics:
    def __init__(self):
        self.rows = []
        self.previous_source_stamp = None

    def receive(self, stamp_ns, arrival_ns):
        if self.rows and arrival_ns < self.rows[-1]["arrival_monotonic_ns"]:
            raise ValueError("Arrival monotonic clock moved backwards")
        previous = self.previous_source_stamp
        source_gap = None if previous is None or stamp_ns <= 0 else stamp_ns-previous
        row = {"sample_index": len(self.rows), "source_stamp_ns": stamp_ns,
               "arrival_monotonic_ns": arrival_ns,
               "arrival_gap_ns": None if not self.rows else arrival_ns-self.rows[-1]["arrival_monotonic_ns"],
               "source_gap_ns": source_gap}
        self.rows.append(row)
        if stamp_ns > 0:
            self.previous_source_stamp = stamp_ns

    def report(self):
        valid = [r for r in self.rows if r["source_stamp_ns"] > 0]
        source_gaps = [r["source_gap_ns"] for r in self.rows if r["source_gap_ns"] is not None]
        arrival_gaps = [r["arrival_gap_ns"] for r in self.rows if r["arrival_gap_ns"] is not None]
        backwards = sum(g < 0 for g in source_gaps)
        span_wall = (self.rows[-1]["arrival_monotonic_ns"]-self.rows[0]["arrival_monotonic_ns"])/1e9 if self.rows else 0.
        span_source = (valid[-1]["source_stamp_ns"]-valid[0]["source_stamp_ns"])/1e9 if valid else 0.
        largest = sorted((r for r in self.rows if r["source_gap_ns"] is not None
                          and r["source_gap_ns"] > 0), key=lambda r: r["source_gap_ns"], reverse=True)[:10]
        return {
            "messages": len(self.rows), "nonpositive_source_stamps": len(self.rows)-len(valid),
            "duplicate_source_stamps": sum(g == 0 for g in source_gaps),
            "backward_source_stamps": backwards, "arrival_wall_span_seconds": span_wall,
            "source_stamp_span_seconds": span_source,
            "arrival_wall_hz": (len(self.rows)-1)/span_wall if span_wall > 0 else None,
            "source_stamp_hz": (len(valid)-1)/span_source if span_source > 0 and not backwards else None,
            "arrival_gaps": gap_statistics(arrival_gaps),
            "positive_source_gaps": gap_statistics([g for g in source_gaps if g > 0]),
            "largest_source_gap_pairs": largest,
        }


class CameraTiming(Node):
    def __init__(self, topic, reliability="best_effort"):
        super().__init__("camera_timing_"+reliability,
                         parameter_overrides=[Parameter("use_sim_time", value=True)])
        self.diagnostics = TimingDiagnostics()
        self.subscription = self.create_subscription(Image, topic, self.receive, timing_qos(reliability))

    def receive(self, message):
        arrival = time.monotonic_ns()
        stamp = message.header.stamp
        self.diagnostics.receive(stamp.sec*1_000_000_000+stamp.nanosec, arrival)


def save_report(options, diagnostics, elapsed, reason):
    label = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = Path(options.output_dir).expanduser()/f"{label}-{uuid4().hex[:12]}"
    root.mkdir(parents=True, exist_ok=False)
    report = {"schema_version": 1, "kind": "camera_timing_diagnostic", "label": options.label,
              "topic": options.rgb_topic,
              "qos": {"reliability": getattr(options, "reliability", "best_effort"),
                      "history": "keep_last", "depth": 5, "durability": "volatile"},
              "requested_wall_seconds": options.seconds, "elapsed_wall_seconds": elapsed,
              "end_reason": reason, "measurement": diagnostics.report(),
              "messages_file": "messages.jsonl",
              "limitations": "Measures this subscriber only; source gaps can include dropped messages; "
                             "arrival gaps include scheduling; no end-to-end transport latency measurement"}
    (root/"messages.jsonl").write_text("".join(json.dumps(r)+"\n" for r in diagnostics.rows), encoding="utf-8")
    (root/"summary.json").write_text(json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2)+"\n",
                                      encoding="utf-8")
    return root, report


def run_timing(options, args=None):
    rclpy.init(args=args)
    node = None
    reason = "duration_reached"
    started = time.monotonic()
    try:
        reliability = options.reliability
        node = CameraTiming(options.rgb_topic, reliability)
        started = time.monotonic()
        print(f"[计时] {options.label}；{options.rgb_topic}；QoS={reliability}/depth=5；观测 {options.seconds:g} 秒，"
              "只缓存时间戳，不转换或保存图像。", flush=True)
        while rclpy.ok() and time.monotonic()-started < options.seconds:
            rclpy.spin_once(node, timeout_sec=.05)
    except (KeyboardInterrupt, ExternalShutdownException):
        reason = "interrupted"
    finally:
        elapsed = time.monotonic()-started
        diagnostics = node.diagnostics if node is not None else TimingDiagnostics()
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    try:
        root, report = save_report(options, diagnostics, elapsed, reason)
    except OSError as error:
        print(f"[失败] 无法保存时间诊断：{error}", flush=True)
        return 1
    stats = report["measurement"]
    print(f"[结果] {options.label}；QoS={options.reliability}/depth=5", flush=True)
    print(f"[消息] {stats['messages']} 条；墙钟接收频率 {stats['arrival_wall_hz']} Hz；"
          f"消息时间戳频率 {stats['source_stamp_hz']} Hz。", flush=True)
    print(f"[到达间隔] {stats['arrival_gaps']}", flush=True)
    print(f"[消息时间戳间隔] {stats['positive_source_gaps']}", flush=True)
    print(f"[异常时间戳] 非正值 {stats['nonpositive_source_stamps']}，"
          f"重复 {stats['duplicate_source_stamps']}，倒退 {stats['backward_source_stamps']}。", flush=True)
    print(f"[保存] {root.resolve()}", flush=True)
    print("[说明] 此处测量订阅端表现；仅凭这些间隔不能区分源端漏帧与传输丢帧。", flush=True)
    return 0 if stats["messages"] >= 2 and reason == "duration_reached" else 1
