"""Freeze one new RGB frame at request time, without depth or localization TF."""

import signal
import time

import cv2

from wheel_arm_navigation.mission import utc_now


def validate_frame(message, now_ns, requested_ns, max_age):
    stamp = message.header.stamp.sec * 1000000000 + message.header.stamp.nanosec
    if (requested_ns is None or now_ns <= 0 or stamp <= 0 or stamp < requested_ns
            or not 0 <= (now_ns - stamp) / 1e9 <= max_age):
        raise ValueError("图像早于请求、过期或与 ROS 时钟不一致")
    if not 0 < message.width <= 1920 or not 0 < message.height <= 1080:
        raise ValueError("图片尺寸必须在 1920×1080 以内")
    return stamp


def capture(options, argv):
    import rclpy
    from cv_bridge import CvBridge, CvBridgeError
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from rclpy.signals import SignalHandlerOptions
    from sensor_msgs.msg import Image

    node = None
    handlers = {}
    try:
        # Simulation is the default; a later user ROS argument can override it.
        rclpy.init(args=[argv[0], "--ros-args", "-p", "use_sim_time:=true", "--", *argv[1:]],
                   signal_handler_options=SignalHandlerOptions.NO)
        node = Node("scene_describe_capture")
        bridge = CvBridge()
        state = {"requested_ns": None, "image": None, "age": None, "stopping": False,
                 "reason": "等待相机和 ROS 时钟"}
        for signum in (signal.SIGINT, signal.SIGTERM):
            handlers[signum] = signal.signal(signum, lambda *_: state.update(stopping=True))
        now = node.get_clock().now().nanoseconds
        if now > 0:
            state["requested_ns"] = now

        def received(message):
            if state["image"] is not None:
                return
            now_ns = node.get_clock().now().nanoseconds
            try:
                stamp = validate_frame(message, now_ns, state["requested_ns"], options.max_age)
            except ValueError as error:
                state["reason"] = str(error)
                return
            state["image"], state["age"] = message, (now_ns - stamp) / 1e9

        subscription = node.create_subscription(Image, options.rgb_topic, received, qos_profile_sensor_data)
        deadline = time.monotonic() + options.wait
        node.get_logger().info("获取请求后的第一帧新鲜 RGB；不等待深度或定位 TF，不发送运动指令")
        while rclpy.ok() and not state["stopping"] and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.05)
            if state["requested_ns"] is None:
                now = node.get_clock().now().nanoseconds
                if now > 0:
                    state["requested_ns"] = now
            if state["image"] is None:
                continue
            message = state["image"]
            array = bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
            if array.shape != (message.height, message.width, 3):
                raise ValueError("RGB 图像尺寸不一致")
            success, encoded = cv2.imencode(".png", array)
            if not success:
                raise ValueError("RGB PNG 编码失败")
            metadata = {"captured_at": utc_now(), "observation_kind": "rgb",
                        "capture_mode": "live_once", "topic": options.rgb_topic,
                        "camera_frame": message.header.frame_id,
                        "requested_stamp_ns": state["requested_ns"],
                        "image_age_at_capture_sec": state["age"],
                        "source_encoding": message.encoding,
                        "image_size": {"width": message.width, "height": message.height},
                        "stamps": {"rgb": {"sec": message.header.stamp.sec,
                                            "nanosec": message.header.stamp.nanosec}}}
            return encoded.tobytes(), metadata
        if state["stopping"]:
            raise KeyboardInterrupt()
        raise RuntimeError(f"相机采集超时：{state['reason']}；请检查 /camera/image、/clock 和时间参数")
    except CvBridgeError as error:
        raise ValueError(f"图像转换失败：{error}") from error
    finally:
        for signum, handler in handlers.items():
            signal.signal(signum, handler)
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
