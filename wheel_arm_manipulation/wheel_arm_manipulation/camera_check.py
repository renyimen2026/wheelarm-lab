"""Read-only raw RGB-D diagnostics, independent of RViz display normalization."""

import argparse
import time

from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.utilities import remove_ros_args
from sensor_msgs.msg import Image


def depth_statistics(depth):
    array = np.asarray(depth)
    total = array.size
    if not total:
        raise ValueError("空深度图")
    valid = np.isfinite(array) & (array > 0)
    samples = array[valid]
    result = {"valid": float(valid.mean()), "nan": float(np.isnan(array).mean()),
              "posinf": float(np.isposinf(array).mean()), "neginf": float(np.isneginf(array).mean()),
              "zero": float((array == 0).mean()),
              "display": float((valid & (array >= .1) & (array <= 3.)).mean())}
    result["percentiles"] = np.percentile(samples, [5, 50, 95]).tolist() if samples.size else None
    return result


class CameraCheck(Node):
    def __init__(self, options):
        super().__init__("camera_check")
        self.bridge = CvBridge()
        self.depth = self.rgb = None
        self.create_subscription(Image, options.depth_topic, self.read_depth, qos_profile_sensor_data)
        self.create_subscription(Image, options.rgb_topic, self.read_rgb, qos_profile_sensor_data)

    def read_depth(self, message):
        try:
            array = self.bridge.imgmsg_to_cv2(message, "passthrough")
            if message.encoding == "16UC1":
                array = array.astype(np.float32)/1000
            elif message.encoding != "32FC1":
                raise ValueError("深度仅支持 32FC1 米 / 16UC1 毫米")
            self.depth = message, time.monotonic(), depth_statistics(array)
        except (ValueError, CvBridgeError) as error:
            self.get_logger().error(str(error))

    def read_rgb(self, message):
        try:
            image = self.bridge.imgmsg_to_cv2(message, "bgr8")
            self.rgb = time.monotonic(), float(image.mean()), float(image.std())
        except CvBridgeError as error:
            self.get_logger().error(str(error))

    def report(self):
        if self.depth is None:
            print("[相机] 尚未收到原始深度消息", flush=True)
            return
        message, receipt, stats = self.depth
        quantiles = stats["percentiles"]
        distance = "无有效距离" if quantiles is None else "P5/P50/P95="+"/".join(f"{v:.3f}" for v in quantiles)+" m"
        age = time.monotonic()-receipt
        if age > 1.:
            hint = "消息已中断或仿真暂停"
        elif not stats["valid"]:
            hint = "源深度无有效值，不是 RViz 显示范围问题；需检查视野/裁剪/渲染"
        elif stats["display"] < .01:
            hint = "有效深度大部分不在当前 RViz 的 0.1~3 m 显示范围"
        else:
            hint = "源深度有有效值；若 RViz 仍黑，请对照显示范围和话题"
        rgb = "RGB 尚未收到" if self.rgb is None else (
            f"RGB均值/标准差={self.rgb[1]:.1f}/{self.rgb[2]:.1f}，接收间隔={time.monotonic()-self.rgb[0]:.2f}s")
        print(f"[相机] stamp={message.header.stamp.sec}.{message.header.stamp.nanosec:09d} "
              f"接收间隔={age:.2f}s 有效={stats['valid']:.1%} "
              f"+Inf={stats['posinf']:.1%} -Inf={stats['neginf']:.1%} "
              f"NaN={stats['nan']:.1%} 零值={stats['zero']:.1%} "
              f"显示范围内={stats['display']:.1%} {distance} | {rgb} | {hint}", flush=True)


def main(args=None):
    parser = argparse.ArgumentParser(description="只读 RGB-D 诊断；运行期间可移动小车复现黑图，不发送运动命令")
    parser.add_argument("--seconds", type=float, default=30.)
    parser.add_argument("--depth-topic", default="/camera/depth_image")
    parser.add_argument("--rgb-topic", default="/camera/image")
    options = parser.parse_args(remove_ros_args(args=args)[1:])
    if not np.isfinite(options.seconds) or options.seconds <= 0:
        parser.error("--seconds 必须是正数")
    rclpy.init(args=args)
    node = CameraCheck(options)
    try:
        end, next_report = time.monotonic()+options.seconds, time.monotonic()
        while rclpy.ok() and time.monotonic() < end:
            rclpy.spin_once(node, timeout_sec=.05)
            if time.monotonic() >= next_report:
                node.report()
                next_report = time.monotonic()+1.
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
