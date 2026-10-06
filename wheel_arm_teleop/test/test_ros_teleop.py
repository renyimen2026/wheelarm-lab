"""Opt-in ROS and desktop tests; no simulated or physical robot is driven."""

import os
import time

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("WHEELARM_ROS_TEST") != "1", reason="Opt-in isolated DDS tests")


@pytest.fixture
def nodes():
    import rclpy
    from geometry_msgs.msg import Twist
    from rclpy.executors import SingleThreadedExecutor
    from trajectory_msgs.msg import JointTrajectory
    from wheel_arm_teleop.keyboard_teleop import KeyboardTeleop

    rclpy.init(args=["test", "--ros-args", "-p", "use_sim_time:=true"])
    teleop = KeyboardTeleop()
    observer = rclpy.create_node("teleop_test_observer")
    messages, trajectories = [], []
    observer.create_subscription(Twist, teleop.base_cmd_topic, messages.append, 10)
    observer.create_subscription(JointTrajectory, teleop.arm_trajectory_topic,
                                 trajectories.append, 10)
    executor = SingleThreadedExecutor()
    executor.add_node(teleop)
    executor.add_node(observer)

    def spin(seconds=0.1):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            executor.spin_once(timeout_sec=0.01)

    try:
        spin(0.3)
        yield teleop, messages, trajectories, spin
    finally:
        teleop.clear_keys()
        spin(0.05)
        if teleop.window is not None:
            teleop.window.destroy()
        executor.shutdown()
        teleop.destroy_node()
        observer.destroy_node()
        rclpy.shutdown()


def test_combined_motion_and_release_without_clock(nodes):
    teleop, messages, _, spin = nodes
    teleop.key_pressed("w")
    teleop.key_pressed("a")
    spin()
    assert messages[-1].linear.x == 0.5 and messages[-1].angular.z == 1.0
    # No /clock is published: keyboard heartbeat still runs on a steady clock.
    before = len(messages)
    spin()
    assert len(messages) > before
    teleop.key_released("w")
    spin(0.05)
    assert messages[-1].linear.x == 0 and messages[-1].angular.z == 1.0
    teleop.key_released("a")
    spin(0.05)
    assert messages[-1].linear.x == messages[-1].angular.z == 0


def test_arm_repeat_and_focus_clear(nodes):
    teleop, messages, trajectories, spin = nodes
    teleop.key_pressed("q")
    spin(0.45)
    assert len(trajectories) >= 3
    teleop.clear_keys()
    count = len(trajectories)
    spin(0.25)
    assert len(trajectories) == count
    assert messages[-1].linear.x == messages[-1].angular.z == 0


def test_close_stops_held_keys(nodes):
    teleop, messages, _, spin = nodes
    teleop.key_pressed("s")
    spin(0.05)
    teleop.request_close()
    spin(0.05)
    assert messages[-1].linear.x == messages[-1].angular.z == 0


@pytest.mark.skipif(os.environ.get("WHEELARM_GUI_TEST") != "1", reason="Requires desktop")
def test_actual_keyboard_window(nodes):
    teleop, messages, _, spin = nodes
    teleop.open_window()
    root = teleop.window.root
    root.update()
    root.focus_force()
    spin(0.1)
    root.event_generate("<KeyPress-w>")
    root.event_generate("<KeyPress-d>")
    spin(0.1)
    assert messages[-1].linear.x == 0.5 and messages[-1].angular.z == -1.0
    root.event_generate("<KeyRelease-w>")
    spin(0.05)
    assert messages[-1].linear.x == 0 and messages[-1].angular.z == -1.0
    root.event_generate("<FocusOut>")
    spin(0.05)
    assert messages[-1].linear.x == messages[-1].angular.z == 0
