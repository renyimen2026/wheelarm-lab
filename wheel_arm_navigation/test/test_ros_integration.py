"""Opt-in DDS tests; run in an isolated ROS_DOMAIN_ID with the workspace sourced."""

import os
import json
import signal
import subprocess
import threading
import time

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("WHEELARM_ROS_TEST") != "1", reason="Set WHEELARM_ROS_TEST=1 for DDS tests")

from wheel_arm_navigation.goals import load, save_pose


@pytest.fixture
def server():
    import rclpy
    from geometry_msgs.msg import TransformStamped
    from nav2_msgs.action import NavigateToPose
    from rclpy.action import ActionServer, CancelResponse, GoalResponse
    from rclpy.callback_groups import ReentrantCallbackGroup
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.time import Time
    from rosgraph_msgs.msg import Clock
    from tf2_ros import TransformBroadcaster

    rclpy.init()
    node = rclpy.create_node("semantic_goal_test_server")
    broadcaster = TransformBroadcaster(node)
    clock = node.create_publisher(Clock, "/clock", 10)
    state = {"poses": [], "canceled": threading.Event(), "accepted": threading.Event(),
             "tf_age": 0.0, "closing": False}

    def tick():
        stamp = node.get_clock().now().to_msg()
        clock.publish(Clock(clock=stamp))
        transform = TransformStamped()
        transform.header.stamp = Time(
            nanoseconds=node.get_clock().now().nanoseconds
            - int(state["tf_age"] * 1e9)).to_msg()
        transform.header.frame_id = "map"
        transform.child_frame_id = "base_footprint"
        transform.transform.translation.x = 1.25
        transform.transform.translation.y = -0.5
        transform.transform.rotation.w = 1.0
        broadcaster.sendTransform(transform)

    def execute(handle):
        state["poses"].append(handle.request.pose)
        state["accepted"].set()
        deadline = time.monotonic() + (
            15 if handle.request.pose.pose.position.x in (3, 5) else 0.2)
        while time.monotonic() < deadline:
            if state["closing"]:
                handle.abort()
                return NavigateToPose.Result()
            if handle.is_cancel_requested:
                state["canceled"].set()
                handle.canceled()
                return NavigateToPose.Result()
            time.sleep(0.02)
        if handle.request.pose.pose.position.x == 2:
            handle.abort()
        else:
            handle.succeed()
        return NavigateToPose.Result()

    action = ActionServer(
        node, NavigateToPose, "navigate_to_pose", execute,
        goal_callback=lambda request: (
            GoalResponse.REJECT if request.pose.pose.position.x == 4 else GoalResponse.ACCEPT),
        cancel_callback=lambda handle: (
            CancelResponse.REJECT if handle.request.pose.pose.position.x == 5
            else CancelResponse.ACCEPT),
        callback_group=ReentrantCallbackGroup())
    timer = node.create_timer(0.05, tick)
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    thread = threading.Thread(target=executor.spin)
    thread.start()
    try:
        yield state
    finally:
        state["closing"] = True
        timer.cancel()
        executor.shutdown(timeout_sec=15)
        thread.join(timeout=15)
        action.destroy()
        node.destroy_node()
        rclpy.shutdown()


def command(path, *args):
    return ["ros2", "run", "wheel_arm_navigation", "semantic_goal",
            "--file", str(path), *args, "--ros-args", "-p", "use_sim_time:=true"]


def run(path, *args):
    return subprocess.run(command(path, *args), capture_output=True, text=True, timeout=20)


def test_record_and_navigation(server, tmp_path):
    path = tmp_path / "goals.yaml"
    result = run(path, "save", "客厅")
    assert result.returncode == 0, result.stdout + result.stderr
    assert load(path)["goals"]["客厅"]["x"] == pytest.approx(1.25)
    result = run(path, "go", "客厅")
    assert result.returncode == 0, result.stdout + result.stderr
    sent = server["poses"][-1]
    assert sent.header.frame_id == "map"
    assert sent.header.stamp.sec > 0
    assert sent.pose.position.y == pytest.approx(-0.5)
    assert sent.pose.orientation.w == pytest.approx(1.0)


@pytest.mark.parametrize("x", [2, 4])
def test_failed_or_rejected_goal(server, tmp_path, x):
    path = tmp_path / "goals.yaml"
    save_pose(path, "room", x, 0, 0)
    result = run(path, "go", "room")
    assert result.returncode != 0


def test_timeout_cancels(server, tmp_path):
    path = tmp_path / "goals.yaml"
    save_pose(path, "room", 3, 0, 0)
    result = run(path, "go", "room", "--timeout", "0.4")
    assert result.returncode != 0
    assert server["canceled"].wait(2), result.stdout + result.stderr


def test_interrupt_cancels(server, tmp_path):
    path = tmp_path / "goals.yaml"
    save_pose(path, "room", 3, 0, 0)
    process = subprocess.Popen(command(path, "go", "room"), stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True, start_new_session=True)
    try:
        assert server["accepted"].wait(10)
        os.killpg(process.pid, signal.SIGINT)
        stdout, stderr = process.communicate(timeout=10)
        assert process.returncode != 0
        assert server["canceled"].wait(2), stdout + stderr
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            process.communicate(timeout=10)


def test_stale_tf_not_recorded(server, tmp_path):
    server["tf_age"] = 5.0
    path = tmp_path / "goals.yaml"
    result = run(path, "save", "room", "--wait", "0.5")
    assert result.returncode != 0
    assert not path.exists()


def test_unknown_goal_not_sent(server, tmp_path):
    path = tmp_path / "goals.yaml"
    save_pose(path, "room", 1, 0, 0)
    result = run(path, "go", "missing")
    assert result.returncode != 0
    assert not server["poses"]


def mission_command(path, logs, *args):
    return ["ros2", "run", "wheel_arm_navigation", "mission_runner", "--file", str(path),
            "--log-dir", str(logs), *args, "--ros-args", "-p", "use_sim_time:=true"]


def mission_log(logs):
    paths = list(logs.glob("*.json"))
    assert len(paths) == 1
    return json.loads(paths[0].read_text(encoding="utf-8"))


def test_mission_sequence(server, tmp_path):
    path, logs = tmp_path / "goals.yaml", tmp_path / "logs"
    save_pose(path, "客厅", 1, 0, 0)
    save_pose(path, "卧室", 1.5, 0, 0)
    result = subprocess.run(mission_command(path, logs, "客厅", "卧室", "--repeat", "2",
                                            "--stay", "0.1"),
                            capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr
    assert [pose.pose.position.x for pose in server["poses"]] == [1, 1.5, 1, 1.5]
    log = mission_log(logs)
    assert log["status"] == "completed"
    assert all(step["status"] == "succeeded" for step in log["steps"])
    assert all(step["stay_sec"] >= 0.09 for step in log["steps"])


@pytest.mark.parametrize("policy,expected,status", [
    ("stop", [2], "failed"), ("skip", [2, 1], "completed_with_failures"),
])
def test_mission_failure_policy(server, tmp_path, policy, expected, status):
    path, logs = tmp_path / "goals.yaml", tmp_path / "logs"
    save_pose(path, "failed", 2, 0, 0)
    save_pose(path, "room", 1, 0, 0)
    result = subprocess.run(mission_command(path, logs, "failed", "room", "--on-failure", policy),
                            capture_output=True, text=True, timeout=20)
    assert result.returncode != 0
    assert [pose.pose.position.x for pose in server["poses"]] == expected
    assert mission_log(logs)["status"] == status


@pytest.mark.parametrize("x,expected,status", [
    (3, [3, 1], "completed_with_failures"), (5, [5], "unsafe_stop"),
])
def test_mission_timeout_safety(server, tmp_path, x, expected, status):
    path, logs = tmp_path / "goals.yaml", tmp_path / "logs"
    save_pose(path, "slow", x, 0, 0)
    save_pose(path, "room", 1, 0, 0)
    result = subprocess.run(mission_command(path, logs, "slow", "room", "--on-failure", "skip",
                                            "--timeout", "0.3"),
                            capture_output=True, text=True, timeout=20)
    assert result.returncode != 0
    assert [pose.pose.position.x for pose in server["poses"]] == expected
    assert mission_log(logs)["status"] == status


@pytest.mark.parametrize("phase", ["navigation", "stay"])
def test_mission_interrupt(server, tmp_path, phase):
    path, logs = tmp_path / "goals.yaml", tmp_path / "logs"
    save_pose(path, "first", 3 if phase == "navigation" else 1, 0, 0)
    save_pose(path, "second", 1.5, 0, 0)
    process = subprocess.Popen(mission_command(path, logs, "first", "second", "--stay", "10"),
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True, start_new_session=True)
    try:
        assert server["accepted"].wait(10)
        if phase == "stay":
            deadline = time.monotonic() + 5
            while mission_log(logs)["steps"][0]["status"] != "succeeded":
                assert time.monotonic() < deadline
                time.sleep(0.05)
        os.killpg(process.pid, signal.SIGINT)
        stdout, stderr = process.communicate(timeout=10)
        assert process.returncode != 0
        log = mission_log(logs)
        assert log["status"] == "canceled", stdout + stderr
        assert log["steps"][1]["status"] == "not_run"
        assert len(server["poses"]) == 1
        if phase == "navigation":
            assert server["canceled"].is_set()
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            process.communicate(timeout=10)
