import select
import sys
import termios
import tty

import rclpy
from geometry_msgs.msg import Twist
from rclpy.duration import Duration
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


HELP_TEXT = """
Wheel arm keyboard teleop

Base:
  w/s: forward/backward
  a/d: turn left/right
  space: stop base

Arm:
  q/e: shoulder_pan_joint +/-
  r/f: shoulder_lift_joint +/-
  t/g: elbow_joint +/-
  y/h: wrist_pitch_joint +/-
  x: reset arm joints to zero

Ctrl-C: quit
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
        self.declare_parameter("linear_speed", 0.15)
        self.declare_parameter("angular_speed", 0.45)
        self.declare_parameter("joint_step", 0.05)
        self.declare_parameter("trajectory_time", 0.25)
        self.declare_parameter("publish_rate", 20.0)
        self.declare_parameter("command_hold_time", 0.35)

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
        self.command_hold_time = (
            self.get_parameter("command_hold_time").get_parameter_value().double_value
        )
        publish_rate = self.get_parameter("publish_rate").get_parameter_value().double_value

        self.cmd_pub = self.create_publisher(Twist, self.base_cmd_topic, 10)
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
        self.base_twist = Twist()
        self.last_base_key_time = self.get_clock().now()
        self.term_settings = None

        if not sys.stdin.isatty():
            raise RuntimeError("keyboard_teleop must be run from an interactive terminal")

        self.term_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())

        self.create_timer(1.0 / publish_rate, self.timer_callback)
        print(HELP_TEXT)
        self.get_logger().info("Publishing base commands to %s" % self.base_cmd_topic)
        self.get_logger().info("Publishing arm trajectories to %s" % self.arm_trajectory_topic)

    def joint_state_callback(self, msg):
        for name, position in zip(msg.name, msg.position):
            if name in self.joint_positions:
                self.joint_positions[name] = position

    def timer_callback(self):
        key = self.read_key()
        if key:
            self.handle_key(key)

        now = self.get_clock().now()
        elapsed = (now - self.last_base_key_time).nanoseconds / 1e9
        if elapsed > self.command_hold_time:
            self.base_twist = Twist()

        self.cmd_pub.publish(self.base_twist)

    def read_key(self):
        readable, _, _ = select.select([sys.stdin], [], [], 0.0)
        if not readable:
            return None
        return sys.stdin.read(1).lower()

    def handle_key(self, key):
        if key in ("w", "s", "a", "d", " "):
            self.handle_base_key(key)
        elif key in self.ARM_KEYS:
            joint_name, direction = self.ARM_KEYS[key]
            self.move_joint(joint_name, direction * self.joint_step)
        elif key == "x":
            self.reset_arm()

    def handle_base_key(self, key):
        twist = Twist()
        if key == "w":
            twist.linear.x = self.linear_speed
        elif key == "s":
            twist.linear.x = -self.linear_speed
        elif key == "a":
            twist.angular.z = self.angular_speed
        elif key == "d":
            twist.angular.z = -self.angular_speed
        self.base_twist = twist
        self.last_base_key_time = self.get_clock().now()

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
        self.cmd_pub.publish(Twist())

    def restore_terminal(self):
        if self.term_settings is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.term_settings)
            self.term_settings = None


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = KeyboardTeleop()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.stop_base()
            node.restore_terminal()
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
