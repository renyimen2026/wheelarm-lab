#!/usr/bin/env python3

import math
from collections import deque

import rclpy
from rclpy.executors import ExternalShutdownException
from nav_msgs.msg import Odometry
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import Imu, LaserScan
from tf2_ros import Buffer, TransformException, TransformListener


def stamp_to_sec(stamp):
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def yaw_from_quat(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def normalize_angle(angle):
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def yaw_from_imu(msg):
    return yaw_from_quat(msg.orientation)


def pose2d_from_odom(msg):
    pose = msg.pose.pose
    twist = msg.twist.twist
    return {
        "x": pose.position.x,
        "y": pose.position.y,
        "yaw": yaw_from_quat(pose.orientation),
        "vx": twist.linear.x,
        "vy": twist.linear.y,
        "wz": twist.angular.z,
    }


def pose2d_from_transform(transform):
    trans = transform.transform.translation
    rot = transform.transform.rotation
    return {
        "x": trans.x,
        "y": trans.y,
        "yaw": yaw_from_quat(rot),
    }


class TopicStats:
    def __init__(self, window_size=100):
        self.times = deque(maxlen=window_size)
        self.last_stamp = None
        self.last_age = None

    def update(self, msg, now_sec):
        msg_sec = stamp_to_sec(msg.header.stamp)
        self.times.append(msg_sec)
        self.last_stamp = msg_sec
        self.last_age = now_sec - msg_sec

    def rate(self):
        if len(self.times) < 2:
            return 0.0
        duration = self.times[-1] - self.times[0]
        if duration <= 0.0:
            return 0.0
        return float(len(self.times) - 1) / duration


class LocalizationDebug(Node):
    def __init__(self):
        super().__init__("localization_debug")

        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("imu_topic", "/imu")
        self.declare_parameter("raw_odom_topic", "/wheel_arm/diff_drive_base_controller/odom")
        self.declare_parameter("filtered_odom_topic", "/odometry/filtered")
        self.declare_parameter("report_period", 1.0)
        self.declare_parameter("jump_translation_threshold", 0.15)
        self.declare_parameter("jump_yaw_threshold", 0.7)

        scan_topic = self.get_parameter("scan_topic").value
        imu_topic = self.get_parameter("imu_topic").value
        raw_odom_topic = self.get_parameter("raw_odom_topic").value
        filtered_odom_topic = self.get_parameter("filtered_odom_topic").value
        report_period = float(self.get_parameter("report_period").value)
        self.jump_translation_threshold = float(
            self.get_parameter("jump_translation_threshold").value
        )
        self.jump_yaw_threshold = float(self.get_parameter("jump_yaw_threshold").value)

        self.scan_stats = TopicStats()
        self.imu_stats = TopicStats()
        self.raw_odom_stats = TopicStats()
        self.filtered_odom_stats = TopicStats()

        self.last_scan = None
        self.last_imu = None
        self.last_raw_odom = None
        self.last_filtered_odom = None
        self.prev_filtered_pose = None
        self.prev_map_to_odom = None
        self.prev_odom_to_base = None

        self.tf_buffer = Buffer(cache_time=Duration(seconds=10.0))
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.create_subscription(LaserScan, scan_topic, self.scan_callback, 20)
        self.create_subscription(Imu, imu_topic, self.imu_callback, 50)
        self.create_subscription(Odometry, raw_odom_topic, self.raw_odom_callback, 20)
        self.create_subscription(
            Odometry,
            filtered_odom_topic,
            self.filtered_odom_callback,
            20,
        )

        self.create_timer(report_period, self.report)
        self.get_logger().info(
            "Localization debug enabled: scan=%s imu=%s raw_odom=%s filtered_odom=%s"
            % (scan_topic, imu_topic, raw_odom_topic, filtered_odom_topic)
        )

    def now_sec(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def scan_callback(self, msg):
        self.scan_stats.update(msg, self.now_sec())
        self.last_scan = msg

    def imu_callback(self, msg):
        self.imu_stats.update(msg, self.now_sec())
        self.last_imu = msg

    def raw_odom_callback(self, msg):
        self.raw_odom_stats.update(msg, self.now_sec())
        self.last_raw_odom = msg

    def filtered_odom_callback(self, msg):
        self.filtered_odom_stats.update(msg, self.now_sec())
        pose = pose2d_from_odom(msg)
        self.check_pose_jump("filtered_odom", self.prev_filtered_pose, pose)
        self.prev_filtered_pose = pose
        self.last_filtered_odom = msg

    def lookup_pose(self, target_frame, source_frame):
        try:
            transform = self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                Time(),
                timeout=Duration(seconds=0.02),
            )
        except TransformException as exc:
            self.get_logger().warn(
                "TF missing %s -> %s: %s" % (target_frame, source_frame, exc),
                throttle_duration_sec=2.0,
            )
            return None
        return pose2d_from_transform(transform)

    def report(self):
        raw = pose2d_from_odom(self.last_raw_odom) if self.last_raw_odom else None
        filtered = (
            pose2d_from_odom(self.last_filtered_odom) if self.last_filtered_odom else None
        )
        map_to_odom = self.lookup_pose("map", "odom")
        odom_to_base = self.lookup_pose("odom", "base_footprint")
        base_to_lidar = self.lookup_pose("base_footprint", "lidar_link")

        if map_to_odom:
            self.check_pose_jump("tf_map_to_odom", self.prev_map_to_odom, map_to_odom)
            self.prev_map_to_odom = map_to_odom
        if odom_to_base:
            self.check_pose_jump("tf_odom_to_base", self.prev_odom_to_base, odom_to_base)
            self.prev_odom_to_base = odom_to_base

        scan_info = self.scan_summary()
        imu_info = self.imu_summary()
        self.get_logger().info(
            "rates Hz scan=%.1f imu=%.1f raw_odom=%.1f filtered=%.1f | "
            "age s scan=%s imu=%s raw=%s filtered=%s"
            % (
                self.scan_stats.rate(),
                self.imu_stats.rate(),
                self.raw_odom_stats.rate(),
                self.filtered_odom_stats.rate(),
                self.format_age(self.scan_stats.last_age),
                self.format_age(self.imu_stats.last_age),
                self.format_age(self.raw_odom_stats.last_age),
                self.format_age(self.filtered_odom_stats.last_age),
            )
        )
        self.get_logger().info(scan_info)
        self.get_logger().info(imu_info)
        self.get_logger().info(self.compare_summary(raw, filtered))
        self.get_logger().info(
            "pose raw=%s | filtered=%s | tf map->odom=%s | tf odom->base=%s | "
            "tf base->lidar=%s"
            % (
                self.format_pose(raw),
                self.format_pose(filtered),
                self.format_pose(map_to_odom),
                self.format_pose(odom_to_base),
                self.format_pose(base_to_lidar),
            )
        )

    def scan_summary(self):
        if self.last_scan is None:
            return "scan no message received"

        finite_ranges = [
            value
            for value in self.last_scan.ranges
            if math.isfinite(value)
            and self.last_scan.range_min <= value <= self.last_scan.range_max
        ]
        finite_count = len(finite_ranges)
        total_count = len(self.last_scan.ranges)
        if finite_ranges:
            min_range = min(finite_ranges)
            mean_range = sum(finite_ranges) / float(finite_count)
            range_text = "min=%.2f mean=%.2f" % (min_range, mean_range)
        else:
            range_text = "no finite ranges"

        return (
            "scan frame=%s valid=%d/%d %s angle=[%.2f, %.2f] inc=%.4f"
            % (
                self.last_scan.header.frame_id,
                finite_count,
                total_count,
                range_text,
                self.last_scan.angle_min,
                self.last_scan.angle_max,
                self.last_scan.angle_increment,
            )
        )

    def imu_summary(self):
        if self.last_imu is None:
            return "imu no message received"

        imu = self.last_imu
        yaw = yaw_from_imu(imu)
        return (
            "imu frame=%s yaw=%.3f wz=%.3f "
            "orientation_cov_diag=[%.3g, %.3g, %.3g] "
            "angular_cov_diag=[%.3g, %.3g, %.3g]"
            % (
                imu.header.frame_id,
                yaw,
                imu.angular_velocity.z,
                imu.orientation_covariance[0],
                imu.orientation_covariance[4],
                imu.orientation_covariance[8],
                imu.angular_velocity_covariance[0],
                imu.angular_velocity_covariance[4],
                imu.angular_velocity_covariance[8],
            )
        )

    def compare_summary(self, raw, filtered):
        if raw is None or filtered is None:
            return "compare raw-filtered=n/a"

        raw_filtered_yaw = normalize_angle(raw["yaw"] - filtered["yaw"])
        raw_filtered_wz = raw["wz"] - filtered["wz"]
        text = (
            "compare raw-filtered: dx=%.3f dy=%.3f dyaw=%.3f dwz=%.3f"
            % (
                raw["x"] - filtered["x"],
                raw["y"] - filtered["y"],
                raw_filtered_yaw,
                raw_filtered_wz,
            )
        )

        if self.last_imu is not None:
            imu_yaw = yaw_from_imu(self.last_imu)
            text += (
                " | imu-filtered: dyaw=%.3f dwz=%.3f"
                % (
                    normalize_angle(imu_yaw - filtered["yaw"]),
                    self.last_imu.angular_velocity.z - filtered["wz"],
                )
            )

        return text

    def check_pose_jump(self, name, previous, current):
        if previous is None or current is None:
            return
        dx = current["x"] - previous["x"]
        dy = current["y"] - previous["y"]
        dyaw = normalize_angle(current["yaw"] - previous["yaw"])
        distance = math.hypot(dx, dy)
        if (
            distance > self.jump_translation_threshold
            or abs(dyaw) > self.jump_yaw_threshold
        ):
            self.get_logger().warn(
                "%s jump: dxy=%.3f dyaw=%.3f current=%s previous=%s"
                % (
                    name,
                    distance,
                    dyaw,
                    self.format_pose(current),
                    self.format_pose(previous),
                )
            )

    @staticmethod
    def format_age(age):
        if age is None:
            return "n/a"
        return "%.3f" % age

    @staticmethod
    def format_pose(pose):
        if pose is None:
            return "n/a"
        text = "x=%.3f y=%.3f yaw=%.3f" % (pose["x"], pose["y"], pose["yaw"])
        if "vx" in pose:
            text += " vx=%.3f vy=%.3f wz=%.3f" % (
                pose["vx"],
                pose["vy"],
                pose["wz"],
            )
        return text


def main():
    rclpy.init()
    node = LocalizationDebug()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException, RuntimeError):
        pass
    finally:
        try:
            node.destroy_node()
        except (KeyboardInterrupt, RuntimeError):
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
