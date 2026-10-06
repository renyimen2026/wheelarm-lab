import math
import signal
import time

import rclpy
from geometry_msgs.msg import Twist
from rclpy.duration import Duration
from rclpy.clock import Clock, ClockType
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from wheel_arm_teleop.key_state import KeyState


HELP_TEXT = """
Wheel arm keyboard teleop (focus the keyboard window)

Base:
  w/s: forward/backward
  a/d: turn left/right
  w+a / w+d / s+a / s+d: move and turn together
  release: stop the released axis
  space: stop base (release movement keys before restarting)

Arm:
  q/e: shoulder_pan_joint +/-
  r/f: shoulder_lift_joint +/-
  t/g: elbow_joint +/-
  y/h: wrist_pitch_joint +/-
  x: reset arm joints to zero

Focus lost: stop base
Escape / Ctrl-C / close window: quit
"""


class KeyboardTeleop(Node):
    ARM_JOINTS = [
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_pitch_joint",
    ]

    JOINT_LIMITS = {
        "shoulder_pan_joint": (-3.14, 3.14),
        "shoulder_lift_joint": (-1.57, 1.57),
        "elbow_joint": (-2.2, 2.2),
        "wrist_pitch_joint": (-2.6, 2.6),
    }

    ARM_KEYS = {
        "q": ("shoulder_pan_joint", 1.0),
        "e": ("shoulder_pan_joint", -1.0),
        "r": ("shoulder_lift_joint", 1.0),
        "f": ("shoulder_lift_joint", -1.0),
        "t": ("elbow_joint", 1.0),
        "g": ("elbow_joint", -1.0),
        "y": ("wrist_pitch_joint", 1.0),
        "h": ("wrist_pitch_joint", -1.0),
    }

    def __init__(self):
        super().__init__("keyboard_teleop")

        self.declare_parameter(
            "base_cmd_topic",
            "/wheel_arm/diff_drive_base_controller/cmd_vel_unstamped",
        )
        self.declare_parameter(
            "arm_trajectory_topic",
            "/wheel_arm/arm_controller/joint_trajectory",
        )
        self.declare_parameter("primary_joint_states_topic", "/wheel_arm/joint_states")
        self.declare_parameter("fallback_joint_states_topic", "/joint_states")
        self.declare_parameter("linear_speed", 0.5)
        self.declare_parameter("angular_speed", 1.0)
        self.declare_parameter("joint_step", 0.05)
        self.declare_parameter("trajectory_time", 0.25)
        self.declare_parameter("publish_rate", 50.0)
        self.declare_parameter("arm_repeat_rate", 5.0)

        self.base_cmd_topic = (
            self.get_parameter("base_cmd_topic").get_parameter_value().string_value
        )
        self.arm_trajectory_topic = (
            self.get_parameter("arm_trajectory_topic").get_parameter_value().string_value
        )
        primary_joint_states_topic = (
            self.get_parameter("primary_joint_states_topic")
            .get_parameter_value()
            .string_value
        )
        fallback_joint_states_topic = (
            self.get_parameter("fallback_joint_states_topic")
            .get_parameter_value()
            .string_value
        )
        self.linear_speed = (
            self.get_parameter("linear_speed").get_parameter_value().double_value
        )
        self.angular_speed = (
            self.get_parameter("angular_speed").get_parameter_value().double_value
        )
        self.joint_step = (
            self.get_parameter("joint_step").get_parameter_value().double_value
        )
        self.trajectory_time = (
            self.get_parameter("trajectory_time").get_parameter_value().double_value
        )
        publish_rate = self.get_parameter("publish_rate").get_parameter_value().double_value
        arm_repeat_rate = self.get_parameter("arm_repeat_rate").value
        for name, value in (
                ("linear_speed", self.linear_speed), ("angular_speed", self.angular_speed),
                ("joint_step", self.joint_step), ("trajectory_time", self.trajectory_time),
                ("publish_rate", publish_rate), ("arm_repeat_rate", arm_repeat_rate)):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a positive finite number")
        self.arm_repeat_interval = 1.0 / arm_repeat_rate
        self.arm_next_repeat = {}

        self.cmd_pub = self.create_publisher(Twist, self.base_cmd_topic, 1)
        self.arm_pub = self.create_publisher(
            JointTrajectory,
            self.arm_trajectory_topic,
            10,
        )
        self.create_subscription(
            JointState,
            primary_joint_states_topic,
            self.joint_state_callback,
            10,
        )
        if fallback_joint_states_topic != primary_joint_states_topic:
            self.create_subscription(
                JointState,
                fallback_joint_states_topic,
                self.joint_state_callback,
                10,
            )

        self.joint_positions = {joint: 0.0 for joint in self.ARM_JOINTS}
        self.key_state = KeyState()
        self.window = None
        self.close_requested = False
        self.create_timer(1.0 / publish_rate, self.timer_callback,
                          clock=Clock(clock_type=ClockType.STEADY_TIME))
        print(HELP_TEXT)
        self.get_logger().info("Publishing base commands to %s" % self.base_cmd_topic)
        self.get_logger().info("Publishing arm trajectories to %s" % self.arm_trajectory_topic)

    def open_window(self):
        from tkinter import TclError
        from wheel_arm_teleop.keyboard_window import KeyboardWindow

        try:
            self.window = KeyboardWindow(self.key_pressed, self.key_released,
                                         self.clear_keys, self.request_close, self.stop_base)
        except TclError as error:
            raise RuntimeError("需要桌面显示环境，请在本地桌面启动键盘窗口") from error

    def joint_state_callback(self, msg):
        for name, position in zip(msg.name, msg.position):
            if name in self.joint_positions:
                self.joint_positions[name] = position

    def timer_callback(self):
        if self.window is not None and not self.close_requested:
            self.window.update()
        now = time.monotonic()
        if not self.close_requested:
            for key in list(self.arm_next_repeat):
                if key in self.key_state.pressed and now >= self.arm_next_repeat[key]:
                    joint_name, direction = self.ARM_KEYS[key]
                    self.move_joint(joint_name, direction * self.joint_step)
                    self.arm_next_repeat[key] = now + self.arm_repeat_interval
        self.publish_base()

    def key_pressed(self, key):
        if self.close_requested:
            return
        if key not in self.key_state.BASE_KEYS | set(self.ARM_KEYS) | {"space", "x"}:
            return
        if not self.key_state.press(key):
            return
        if key in self.ARM_KEYS:
            joint_name, direction = self.ARM_KEYS[key]
            self.move_joint(joint_name, direction * self.joint_step)
            self.arm_next_repeat[key] = time.monotonic() + self.arm_repeat_interval
        elif key == "x":
            self.reset_arm()
        self.publish_base()

    def key_released(self, key):
        self.key_state.release(key)
        self.arm_next_repeat.pop(key, None)
        self.publish_base()

    def publish_base(self):
        linear, angular = self.key_state.velocity(self.linear_speed, self.angular_speed)
        if self.close_requested:
            linear, angular = 0.0, 0.0
        twist = Twist()
        twist.linear.x = linear
        twist.angular.z = angular
        self.cmd_pub.publish(twist)
        if self.window is not None:
            self.window.display(linear, angular, self.key_state.pressed)

    def clear_keys(self):
        self.key_state.clear()
        self.arm_next_repeat.clear()
        self.publish_base()

    def request_close(self):
        self.close_requested = True
        self.clear_keys()

    def move_joint(self, joint_name, delta):
        lower, upper = self.JOINT_LIMITS[joint_name]
        target = self.clamp(self.joint_positions[joint_name] + delta, lower, upper)
        self.joint_positions[joint_name] = target
        self.publish_arm_trajectory()

    def reset_arm(self):
        self.joint_positions = {joint: 0.0 for joint in self.ARM_JOINTS}
        self.publish_arm_trajectory()

    def publish_arm_trajectory(self):
        msg = JointTrajectory()
        msg.joint_names = list(self.ARM_JOINTS)

        point = JointTrajectoryPoint()
        point.positions = [self.joint_positions[joint] for joint in self.ARM_JOINTS]
        point.time_from_start = Duration(seconds=self.trajectory_time).to_msg()

        msg.points = [point]
        self.arm_pub.publish(msg)

    @staticmethod
    def clamp(value, lower, upper):
        return min(max(value, lower), upper)

    def stop_base(self):
        self.key_state.stop()
        self.cmd_pub.publish(Twist())
        if self.window is not None:
            self.window.display(0.0, 0.0, self.key_state.pressed)


def main(args=None):
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = None
    handlers = {}
    try:
        node = KeyboardTeleop()
        for signum in (signal.SIGINT, signal.SIGTERM):
            handlers[signum] = signal.signal(
                signum, lambda *_: setattr(node, "close_requested", True))
        node.open_window()
        while rclpy.ok() and not node.close_requested:
            rclpy.spin_once(node, timeout_sec=0.02)
    finally:
        for signum, handler in handlers.items():
            signal.signal(signum, handler)
        if node is not None:
            node.close_requested = True
            node.clear_keys()
            # Allow repeated zero commands to leave DDS before destroying the publisher.
            for _ in range(3):
                rclpy.spin_once(node, timeout_sec=0.02)
            if node.window is not None:
                node.window.destroy()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
