#!/usr/bin/env python3

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import Imu


class ImuCovariance(Node):
    def __init__(self):
        super().__init__("imu_covariance")

        self.declare_parameter("input_topic", "/imu/raw")
        self.declare_parameter("output_topic", "/imu")
        self.declare_parameter("frame_id", "imu_link")
        self.declare_parameter(
            "orientation_covariance",
            [0.05, 0.0, 0.0, 0.0, 0.05, 0.0, 0.0, 0.0, 0.005],
        )
        self.declare_parameter(
            "angular_velocity_covariance",
            [0.02, 0.0, 0.0, 0.0, 0.02, 0.0, 0.0, 0.0, 0.002],
        )
        self.declare_parameter(
            "linear_acceleration_covariance",
            [0.2, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.2],
        )

        input_topic = self.get_parameter("input_topic").value
        output_topic = self.get_parameter("output_topic").value
        self.frame_id = self.get_parameter("frame_id").value
        self.orientation_covariance = list(
            self.get_parameter("orientation_covariance").value
        )
        self.angular_velocity_covariance = list(
            self.get_parameter("angular_velocity_covariance").value
        )
        self.linear_acceleration_covariance = list(
            self.get_parameter("linear_acceleration_covariance").value
        )

        self.publisher = self.create_publisher(Imu, output_topic, 20)
        self.subscription = self.create_subscription(
            Imu,
            input_topic,
            self.imu_callback,
            50,
        )

        self.get_logger().info(
            "IMU covariance enabled: %s -> %s frame=%s"
            % (input_topic, output_topic, self.frame_id)
        )

    def imu_callback(self, msg):
        msg.header.frame_id = self.frame_id
        msg.orientation_covariance = self.orientation_covariance
        msg.angular_velocity_covariance = self.angular_velocity_covariance
        msg.linear_acceleration_covariance = self.linear_acceleration_covariance
        self.publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = ImuCovariance()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
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
