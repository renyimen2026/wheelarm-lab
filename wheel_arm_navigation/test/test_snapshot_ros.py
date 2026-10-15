"""Opt-in synthetic RGB-D/TF tests in an unused ROS domain, never send actions."""

import json
import os
import subprocess
import threading

import numpy as np
import pytest


pytestmark = pytest.mark.skipif(
    os.environ.get("WHEELARM_ROS_TEST") != "1", reason="Set WHEELARM_ROS_TEST=1 for DDS tests")


@pytest.fixture
def rgbd_server():
    import rclpy
    from cv_bridge import CvBridge
    from geometry_msgs.msg import TransformStamped
    from rclpy.executors import SingleThreadedExecutor
    from rosgraph_msgs.msg import Clock
    from sensor_msgs.msg import CameraInfo, Image
    from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster

    rclpy.init()
    node = rclpy.create_node("rgbd_test_server")
    publishers = [node.create_publisher(kind, topic, 10) for kind, topic in (
        (Image, "/camera/image"), (Image, "/camera/depth_image"),
        (CameraInfo, "/camera/camera_info"), (Clock, "/clock"))]
    bridge = CvBridge()
    broadcaster = TransformBroadcaster(node)
    static = StaticTransformBroadcaster(node)
    extrinsic = TransformStamped()
    extrinsic.header.frame_id = "base_footprint"
    extrinsic.child_frame_id = "camera_optical_frame"
    extrinsic.transform.translation.x = 0.34
    extrinsic.transform.translation.z = 0.23
    q = extrinsic.transform.rotation
    q.x, q.y, q.z, q.w = -0.5, 0.5, -0.5, 0.5
    static.sendTransform(extrinsic)
    state = {"publish_tf": True, "stale": False, "frame": "camera_optical_frame"}

    def tick():
        stamp = node.get_clock().now().to_msg()
        publishers[3].publish(Clock(clock=stamp))
        pose = TransformStamped()
        pose.header.stamp = stamp
        pose.header.frame_id = "map"
        pose.child_frame_id = "base_footprint"
        pose.transform.translation.x = 1.0
        pose.transform.translation.y = 2.0
        pose.transform.rotation.w = 1.0
        if state["publish_tf"]:
            broadcaster.sendTransform(pose)
        rgb = bridge.cv2_to_imgmsg(np.full((3, 4, 3), [255, 0, 0], dtype=np.uint8), "rgb8")
        depth = bridge.cv2_to_imgmsg(np.full((3, 4), 2.0, dtype=np.float32), "32FC1")
        info = CameraInfo(width=4, height=3)
        info.k = [2.0, 0.0, 2.0, 0.0, 2.0, 1.5, 0.0, 0.0, 1.0]
        for publisher, message in zip(publishers, (rgb, depth, info)):
            message.header.stamp = stamp
            if state["stale"]:
                message.header.stamp.sec -= 2
            message.header.frame_id = state["frame"]
            publisher.publish(message)

    timer = node.create_timer(0.05, tick)
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    thread = threading.Thread(target=executor.spin)
    thread.start()
    try:
        yield state
    finally:
        timer.cancel()
        executor.shutdown(timeout_sec=5)
        thread.join(timeout=5)
        node.destroy_node()
        rclpy.shutdown()


def run_snapshot(tmp_path, *arguments):
    return subprocess.run([
        "ros2", "run", "wheel_arm_navigation", "scene_snapshot",
        "--output-dir", str(tmp_path), "--wait", "2", *arguments,
        "--ros-args", "-p", "use_sim_time:=true"],
        text=True, capture_output=True, timeout=15)


def test_snapshot_uses_image_time_tf(rgbd_server, tmp_path):
    result = run_snapshot(tmp_path, "--label", "餐厅")
    assert result.returncode == 0, result.stdout + result.stderr
    record = json.loads(next(tmp_path.glob("*/metadata.json")).read_text(encoding="utf-8"))
    assert record["label"] == "餐厅" and record["sync_span_sec"] == 0
    assert record["transforms"]["reference_to_base"]["stamp"] == record["stamps"]["rgb"]
    assert record["transforms"]["reference_to_base"]["translation"]["x"] == 1
    assert record["transforms"]["reference_to_camera"]["translation"]["x"] == pytest.approx(1.34)
    assert record["depth_statistics"]["median_m"] == 2


@pytest.mark.parametrize("failure", ["no_tf", "stale", "wrong_frame"])
def test_bad_sensor_state_does_not_save(rgbd_server, tmp_path, failure):
    if failure == "no_tf":
        rgbd_server["publish_tf"] = False
    elif failure == "stale":
        rgbd_server["stale"] = True
    else:
        rgbd_server["frame"] = "wrong_frame"
    result = run_snapshot(tmp_path)
    assert result.returncode != 0
    assert not list(tmp_path.glob("*/metadata.json"))
