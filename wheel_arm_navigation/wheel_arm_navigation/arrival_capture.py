"""Observe only after fresh low-speed odometry and image-time goal alignment."""

import math
import time

import cv2
import rclpy
from cv_bridge import CvBridge
from nav_msgs.msg import Odometry
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import Image
from tf2_ros import Buffer, TransformException, TransformListener

from wheel_arm_navigation.live_image import validate_frame
from wheel_arm_navigation.mission import utc_now
from wheel_arm_navigation.scene_snapshot import transform_record
from wheel_arm_navigation.semantic_goal import GoalClient


class Stillness:
    def __init__(self, duration=0.5, linear_limit=0.02, angular_limit=0.04, max_age=0.5):
        self.duration, self.linear_limit = duration, linear_limit
        self.angular_limit, self.max_age = angular_limit, max_age
        self.reset()

    def reset(self):
        self.start_ns = self.last_ns = self.last_wall = None
        self.speed = None

    def update(self, stamp_ns, linear, angular, now_ns, wall):
        valid = (stamp_ns > 0 and 0 <= (now_ns - stamp_ns) / 1e9 <= self.max_age
                 and math.isfinite(linear) and math.isfinite(angular)
                 and 0 <= linear <= self.linear_limit and 0 <= angular <= self.angular_limit)
        if not valid:
            self.reset()
            return
        if (self.last_ns is None or stamp_ns <= self.last_ns
                or (stamp_ns - self.last_ns) / 1e9 > self.max_age
                or wall - self.last_wall > self.max_age):
            self.start_ns = stamp_ns
        self.last_ns, self.last_wall, self.speed = stamp_ns, wall, {"linear": linear, "angular": angular}

    def ready(self, now_ns, wall):
        return (self.last_ns is not None
                and 0 <= (now_ns - self.last_ns) / 1e9 <= self.max_age
                and 0 <= wall - self.last_wall <= self.max_age
                and (self.last_ns - self.start_ns) / 1e9 >= self.duration)


def near_goal(transform, pose, xy_tolerance=0.35, yaw_tolerance=0.35):
    position, q = transform.transform.translation, transform.transform.rotation
    values = (position.x, position.y, q.x, q.y, q.z, q.w)
    if not all(math.isfinite(value) for value in values):
        return False
    yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
    error = math.atan2(math.sin(yaw - pose["yaw"]), math.cos(yaw - pose["yaw"]))
    return math.hypot(position.x - pose["x"], position.y - pose["y"]) <= xy_tolerance and abs(error) <= yaw_tolerance


class ObserveClient(GoalClient):
    def __init__(self, options):
        super().__init__("language_observe")
        self.options = options
        self.phase = "navigation"
        self.vision_preload = None
        self.motion_after_ns = 0
        self.motion = Stillness(duration=options.settle_duration)
        self.buffer = Buffer()
        self.listener = TransformListener(self.buffer, self)
        self.odom_sub = self.create_subscription(Odometry, options.odom_topic, self.odometry,
                                                 qos_profile_sensor_data)

    def on_goal_accepted(self, name):
        if self.vision_preload is not None:
            self.vision_preload.start()

    def odometry(self, message):
        if message.child_frame_id != "base_footprint":
            self.motion.reset()
            return
        twist = message.twist.twist
        linear = math.hypot(*(getattr(twist.linear, key) for key in ("x", "y", "z")))
        angular = math.hypot(*(getattr(twist.angular, key) for key in ("x", "y", "z")))
        stamp = Time.from_msg(message.header.stamp).nanoseconds
        if stamp < self.motion_after_ns:
            self.motion.reset()
            return
        self.motion.update(stamp, linear, angular, self.get_clock().now().nanoseconds, time.monotonic())

    def capture_at_goal(self, pose, frame_id):
        self.phase = "capture"
        self.motion.reset()
        self.motion_after_ns = self.get_clock().now().nanoseconds
        state = {"sample": None, "requested_ns": None, "age": None}
        bridge = CvBridge()

        def received(message):
            now = self.get_clock().now().nanoseconds
            if state["sample"] is not None or not self.motion.ready(now, time.monotonic()):
                return
            if state["requested_ns"] is None:
                state["requested_ns"] = now
            try:
                stamp = validate_frame(message, now, state["requested_ns"], self.options.max_age)
            except ValueError:
                return
            state["sample"], state["age"] = message, (now - stamp) / 1e9

        subscription = self.create_subscription(Image, self.options.rgb_topic, received, qos_profile_sensor_data)
        deadline = time.monotonic() + self.options.capture_wait
        reason = "等待到达后连续低速里程计"
        try:
            while not self.stopping and rclpy.ok() and time.monotonic() < deadline:
                rclpy.spin_once(self, timeout_sec=0.05)
                now = self.get_clock().now().nanoseconds
                if not self.motion.ready(now, time.monotonic()):
                    state.update(sample=None, requested_ns=None)
                    continue
                message = state["sample"]
                if message is None:
                    reason = "等待停稳后的新鲜 RGB"
                    continue
                try:
                    validate_frame(message, now, state["requested_ns"], self.options.max_age)
                    stamp = Time.from_msg(message.header.stamp)
                    base = self.buffer.lookup_transform(frame_id, "base_footprint", stamp)
                    camera = self.buffer.lookup_transform(frame_id, message.header.frame_id, stamp)
                    if not near_goal(base, pose):
                        raise ValueError("拍摄时刻地图位姿偏离目标位置或朝向")
                except (ValueError, TransformException) as error:
                    reason = str(error)
                    if (now - Time.from_msg(message.header.stamp).nanoseconds) / 1e9 > self.options.max_age:
                        state["sample"] = None
                    continue
                array = bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
                success, encoded = cv2.imencode(".png", array)
                if not success:
                    raise ValueError("RGB 编码失败")
                metadata = {"captured_at": utc_now(), "observation_kind": "rgb",
                            "capture_mode": "after_navigation", "topic": self.options.rgb_topic,
                            "camera_frame": message.header.frame_id, "reference_frame": frame_id,
                            "requested_stamp_ns": state["requested_ns"],
                            "image_age_at_capture_sec": state["age"], "source_encoding": message.encoding,
                            "image_size": {"width": message.width, "height": message.height},
                            "stamps": {"rgb": {"sec": message.header.stamp.sec, "nanosec": message.header.stamp.nanosec}},
                            "transforms": {"reference_to_base": transform_record(base),
                                           "reference_to_camera": transform_record(camera)},
                            "stop_check": {"odom_topic": self.options.odom_topic, "duration_sec": self.motion.duration,
                                           "linear_limit": self.motion.linear_limit, "angular_limit": self.motion.angular_limit,
                                           "measured_speed": self.motion.speed}}
                return encoded.tobytes(), metadata
            if self.stopping:
                raise KeyboardInterrupt()
            raise RuntimeError(f"到达后采集失败：{reason}；请检查里程计、定位、TF 和相机")
        finally:
            self.destroy_subscription(subscription)
