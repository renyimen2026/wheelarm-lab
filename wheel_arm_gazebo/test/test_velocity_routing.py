"""Exercise actual ROS remapping and stop behavior in an unused DDS domain."""

import importlib.util
import os
from pathlib import Path
import subprocess
import time

import pytest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("nav_launch", ROOT / "launch/nav.launch.py")
nav = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nav)
pytestmark = pytest.mark.skipif(os.environ.get("WHEELARM_ROS_TEST") != "1",
                                reason="Set WHEELARM_ROS_TEST=1 in an unused ROS domain")


def arguments(local):
    return ["--ros-args"] + [item for src, dst in nav.NAV2_VELOCITY_REMAPPINGS + local
                             for item in ("-r", f"{src}:={dst}")]


@pytest.mark.parametrize("name,local,expected", [
    ("controller_server", [("cmd_vel", "cmd_vel_nav")], {"cmd_vel": "/cmd_vel_nav"}),
    ("velocity_smoother", [("cmd_vel", "cmd_vel_nav"), ("cmd_vel_smoothed", "cmd_vel")],
     {"cmd_vel": "/cmd_vel_nav", "cmd_vel_smoothed": nav.BASE_COMMAND_TOPIC}),
    ("behavior_server", [], {"cmd_vel": nav.BASE_COMMAND_TOPIC}),
    ("unrelated_node", [], {"cmd_vel": "/cmd_vel"}),
])
def test_upstream_remaps_resolve_without_feedback(name, local, expected):
    import rclpy
    rclpy.init()
    node = rclpy.create_node(name, cli_args=arguments(local))
    try:
        assert {topic: node.resolve_topic_name(topic) for topic in expected} == expected
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_real_smoother_stops_after_input_timeout():
    import rclpy
    from geometry_msgs.msg import Twist
    from lifecycle_msgs.srv import ChangeState

    executable = "/opt/ros/humble/lib/nav2_velocity_smoother/velocity_smoother"
    process = subprocess.Popen([executable] + arguments([
        ("cmd_vel", "cmd_vel_nav"), ("cmd_vel_smoothed", "cmd_vel")]) + [
        "-p", "use_sim_time:=false", "-p", "velocity_timeout:=0.3"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    rclpy.init()
    node = rclpy.create_node("velocity_routing_test")
    received = []
    node.create_subscription(Twist, nav.BASE_COMMAND_TOPIC,
                             lambda msg: received.append((time.monotonic(), msg.linear.x, msg.angular.z)), 10)
    publisher = node.create_publisher(Twist, "/cmd_vel_nav", 10)
    service = node.create_client(ChangeState, "/velocity_smoother/change_state")
    try:
        assert service.wait_for_service(timeout_sec=10)
        for transition in (1, 3):
            request = ChangeState.Request()
            request.transition.id = transition
            future = service.call_async(request)
            rclpy.spin_until_future_complete(node, future, timeout_sec=5)
            assert future.done() and future.result().success
        command = Twist()
        command.linear.x, command.angular.z = 0.2, 0.3
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            publisher.publish(command)
            rclpy.spin_once(node, timeout_sec=0.05)
        stopped_input = time.monotonic()
        deadline = stopped_input + 2.0
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.05)
        assert any(x > 0.05 and z > 0.05 for _, x, z in received)
        after = [(x, z) for stamp, x, z in received if stamp > stopped_input + 1.0]
        # The smoother may cease publishing once zero is reached.
        assert received[-1][1:] == (0.0, 0.0)
        assert all(x == 0.0 and z == 0.0 for x, z in after)
    finally:
        node.destroy_node()
        rclpy.shutdown()
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
