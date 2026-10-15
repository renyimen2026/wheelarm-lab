"""Opt-in actual AMCL test with synthetic map/scan/TF, never sends movement."""

import math
import os
from pathlib import Path
import signal
import subprocess
import time

import pytest
import yaml


pytestmark = pytest.mark.skipif(os.environ.get("WHEELARM_ROS_TEST") != "1",
                                reason="Set WHEELARM_ROS_TEST=1 in an isolated ROS domain")


def test_amcl_prior_then_manual_override(tmp_path):
    import rclpy
    from geometry_msgs.msg import PoseWithCovarianceStamped, TransformStamped
    from lifecycle_msgs.msg import Transition
    from lifecycle_msgs.srv import ChangeState
    from nav_msgs.msg import OccupancyGrid
    from rclpy.qos import DurabilityPolicy, QoSProfile
    from sensor_msgs.msg import LaserScan
    from tf2_ros import Buffer, StaticTransformBroadcaster, TransformListener

    rclpy.init()
    node = rclpy.create_node("amcl_prior_test")
    buffer = Buffer()
    listener = TransformListener(buffer, node)
    broadcaster = StaticTransformBroadcaster(node)
    transforms = []
    for parent, child in (("odom", "base_footprint"), ("base_footprint", "lidar_link")):
        transform = TransformStamped()
        transform.header.frame_id, transform.child_frame_id = parent, child
        transform.transform.rotation.w = 1.0
        transforms.append(transform)
    broadcaster.sendTransform(transforms)
    map_publisher = node.create_publisher(OccupancyGrid, "/map", QoSProfile(
        depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
    scan_publisher = node.create_publisher(LaserScan, "/scan", 10)
    pose_publisher = node.create_publisher(PoseWithCovarianceStamped, "/initialpose", 10)
    poses = []
    subscription = node.create_subscription(PoseWithCovarianceStamped, "/amcl_pose", poses.append, 10)
    grid = OccupancyGrid()
    grid.header.frame_id = "map"
    grid.info.resolution, grid.info.width, grid.info.height = 0.1, 100, 100
    grid.info.origin.position.x = grid.info.origin.position.y = -5.0
    grid.info.origin.orientation.w = 1.0
    grid.data = [0] * 10000
    map_publisher.publish(grid)

    def tick():
        scan = LaserScan()
        scan.header.frame_id, scan.header.stamp = "lidar_link", node.get_clock().now().to_msg()
        scan.angle_min, scan.angle_max, scan.angle_increment = -math.pi, math.pi, math.pi / 180
        scan.range_min, scan.range_max = 0.1, 8.0
        scan.ranges = [float("inf")] * 361
        scan_publisher.publish(scan)

    timer = node.create_timer(0.05, tick)
    document = yaml.safe_load((Path(__file__).resolve().parents[1] / "config/nav2_params.yaml").read_text())
    parameters = document["amcl"]["ros__parameters"]
    parameters.update(use_sim_time=False, set_initial_pose=True,
                      initial_pose={"x": 1.0, "y": 2.0, "z": 0.0, "yaw": 0.0})
    path = tmp_path / "params.yaml"
    path.write_text(yaml.safe_dump({"amcl": {"ros__parameters": parameters}}))
    log = (tmp_path / "amcl.log").open("w")
    process = None

    def wait_for(predicate, timeout=15):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            assert process.poll() is None, (tmp_path / "amcl.log").read_text()
            rclpy.spin_once(node, timeout_sec=0.05)
            if predicate():
                return
        pytest.fail("AMCL timeout: " + (tmp_path / "amcl.log").read_text())

    try:
        process = subprocess.Popen(["ros2", "run", "nav2_amcl", "amcl", "--ros-args",
                                    "--params-file", str(path)], stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        client = node.create_client(ChangeState, "/amcl/change_state")
        wait_for(lambda: client.service_is_ready())
        for transition in (Transition.TRANSITION_CONFIGURE, Transition.TRANSITION_ACTIVATE):
            request = ChangeState.Request()
            request.transition.id = transition
            future = client.call_async(request)
            wait_for(future.done)
            assert future.result().success
        wait_for(lambda: poses and buffer.can_transform("map", "base_footprint", rclpy.time.Time()))
        assert poses[-1].pose.pose.position.x == pytest.approx(1.0, abs=0.25)
        assert poses[-1].pose.pose.position.y == pytest.approx(2.0, abs=0.25)
        manual = PoseWithCovarianceStamped()
        manual.header.frame_id, manual.header.stamp = "map", node.get_clock().now().to_msg()
        manual.pose.pose.position.x, manual.pose.pose.position.y = -1.0, -2.0
        manual.pose.pose.orientation.w = 1.0
        for index in (0, 7, 35):
            manual.pose.covariance[index] = 1e-6
        pose_publisher.publish(manual)
        wait_for(lambda: poses and abs(poses[-1].pose.pose.position.x + 1) < 0.05
                 and abs(poses[-1].pose.pose.position.y + 2) < 0.05)
        # The startup parameters must not reset a manually corrected pose on later scans.
        deadline = time.monotonic() + 0.5
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.05)
        assert poses[-1].pose.pose.position.x == pytest.approx(-1.0, abs=0.05)
    finally:
        timer.cancel()
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGINT)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)
        log.close()
        node.destroy_node()
        rclpy.shutdown()
