"""Confirmed CLI workflow in an isolated DDS domain with fake actions and models."""

import base64
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import pty
import signal
import subprocess
import threading
import time

import pytest

from test_ros_integration import server
from wheel_arm_navigation.goals import save_pose


pytestmark = pytest.mark.skipif(os.environ.get("WHEELARM_ROS_TEST") != "1",
                                reason="Set WHEELARM_ROS_TEST=1 in an unused ROS domain")


@pytest.fixture
def sensors(server):
    import numpy as np
    import rclpy
    from cv_bridge import CvBridge
    from geometry_msgs.msg import TransformStamped
    from nav_msgs.msg import Odometry
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.time import Time
    from sensor_msgs.msg import Image
    from tf2_ros import StaticTransformBroadcaster

    node = rclpy.create_node("observe_test_sensors")
    odom = node.create_publisher(Odometry, "/odometry/filtered", 10)
    camera = node.create_publisher(Image, "/camera/image", 10)
    bridge = CvBridge()
    broadcaster = StaticTransformBroadcaster(node)
    transform = TransformStamped()
    transform.header.frame_id, transform.child_frame_id = "base_footprint", "camera_optical_frame"
    transform.transform.rotation.w = 1.0
    broadcaster.sendTransform(transform)
    state = {"speed": 0.0, "image_frame": "camera_optical_frame", "image_age": 0}

    def tick():
        # Keep sensor stamps behind the independently published simulation clock.
        stamp = Time(nanoseconds=node.get_clock().now().nanoseconds - 80_000_000).to_msg()
        message = Odometry()
        message.header.frame_id, message.child_frame_id = "odom", "base_footprint"
        message.header.stamp = stamp
        message.twist.twist.linear.x = state["speed"]
        odom.publish(message)
        image = bridge.cv2_to_imgmsg(np.full((3, 4, 3), [20, 40, 60], dtype=np.uint8), "bgr8")
        image.header.frame_id, image.header.stamp = state["image_frame"], stamp
        image.header.stamp.sec -= state["image_age"]
        camera.publish(image)

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


@pytest.mark.parametrize("mode", ["success", "nav_abort", "moving", "wrong_pose",
                                  "no_camera_tf", "model_error", "preflight_error", "stale_image", "cancel_nav",
                                  "preload_success", "preload_error", "preload_rejected", "preload_cancel", "preload_disabled"])
def test_confirmed_workflow(server, sensors, tmp_path, mode):
    goals = tmp_path / "goals.yaml"
    cancel = mode in ("cancel_nav", "preload_cancel")
    x = 3 if cancel else 4 if mode == "preload_rejected" else 2 if mode == "nav_abort" else 0 if mode == "wrong_pose" else 1.25
    if mode.startswith("preload"):
        server["duration"] = 0.8
    save_pose(goals, "厨房", x, -0.5, 0)
    sensors["speed"] = 0.2 if mode == "moving" else 0.0
    sensors["image_frame"] = "missing_optical_frame" if mode == "no_camera_tf" else "camera_optical_frame"
    sensors["image_age"] = 2 if mode == "stale_image" else 0
    visual_images = []
    preload_calls, preloaded = [], threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if self.path == "/api/show":
                response = {"capabilities": ["completion"] if mode == "preflight_error" else ["vision"]}
                if mode.startswith("preload"):
                    response.update(backend="pytorch-cuda", features=["preload"])
            elif self.path == "/api/preload":
                preload_calls.append((server["accepted"].is_set(), not server["finished"].is_set()))
                time.sleep(1.0)
                preloaded.set()
                if mode == "preload_error":
                    self.send_response(500)
                    self.end_headers()
                    self.wfile.write(b'{"error":"test load failed"}')
                    return
                response = {"model": payload["model"], "status": "ready", "backend": "pytorch-cuda"}
            elif "images" in payload["messages"][-1]:
                if mode in ("preload_success", "preload_error"):
                    assert preloaded.is_set()
                visual_images.append(base64.b64decode(payload["messages"][-1]["images"][0]))
                result = {"summary": "当前画面未确认椅子", "objects": [], "uncertainties": ["单帧视角有限"]}
                if mode == "model_error":
                    result = {"cmd_vel": 1}
                response = {"done": True, "message": {"content": json.dumps(result)}}
            else:
                response = {"done": True, "message": {"content": json.dumps({
                    "status": "ready", "message": "到达后观察", "goal": "厨房",
                    "question": "当前画面是否看到椅子？"})}}
            self.send_response(200)
            self.end_headers()
            try:
                self.wfile.write(json.dumps(response).encode())
            except BrokenPipeError:
                pass

        def log_message(self, *_args):
            pass

    http = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=http.serve_forever)
    thread.start()
    master, slave = pty.openpty()
    process = None
    try:
        command = [
            "ros2", "run", "wheel_arm_navigation", "language_observe", "去厨房看看有没有椅子",
            "--file", str(goals), "--log-dir", str(tmp_path / "logs"),
            "--observations-dir", str(tmp_path / "images"), "--capture-wait", "2",
            "--settle-duration", "0.1", "--endpoint", f"http://127.0.0.1:{http.server_port}"]
        if mode == "preload_disabled":
            command.append("--no-vision-preload")
        process = subprocess.Popen(command,
            stdin=slave, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
        os.close(slave)
        slave = None
        os.write(master, b"y\n")
        if cancel:
            assert server["accepted"].wait(timeout=10)
            if mode == "preload_cancel":
                deadline = time.monotonic() + 3
                while not preload_calls and time.monotonic() < deadline:
                    time.sleep(0.02)
                assert preload_calls
            os.killpg(process.pid, signal.SIGINT)
        output, _ = process.communicate(timeout=20)
        success = mode in ("success", "preload_success", "preload_error", "preload_disabled")
        assert process.returncode == (0 if success else 130 if cancel else 1), output
        record = json.loads(next((tmp_path / "logs").glob("*.json")).read_text())
        if cancel:
            assert record["status"] == "canceled" and server["canceled"].is_set()
            assert not visual_images
        elif mode == "preflight_error":
            assert not server["poses"] and not visual_images
            assert record["status"] == "preflight_failed"
        elif mode in ("nav_abort", "preload_rejected"):
            assert record["status"] == "failed" and not visual_images
        elif mode in ("moving", "wrong_pose", "no_camera_tf", "stale_image"):
            assert record["status"] == "capture_failed" and not visual_images
        else:
            assert len(visual_images) == 1
            directory = Path(record["observation"]["directory"])
            assert (directory / "rgb.png").read_bytes() == visual_images[0]
            metadata = json.loads((directory / "metadata.json").read_text())
            assert metadata["transforms"]["reference_to_base"]["stamp"] == metadata["stamps"]["rgb"]
            assert record["status"] == ("completed" if success else "vision_failed")
            if success:
                assert Path(record["observation"]["report"]).is_file()
        if mode in ("preload_success", "preload_error"):
            assert preload_calls == [(True, True)]
            assert record["vision_preload"]["status"] == ("ready" if mode == "preload_success" else "failed")
        elif mode in ("preload_rejected", "preload_disabled") or not mode.startswith("preload"):
            assert not preload_calls
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGINT)
            try:
                process.wait(timeout=8)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)
        os.close(master)
        if slave is not None:
            os.close(slave)
        http.shutdown()
        thread.join(timeout=5)
        http.server_close()
