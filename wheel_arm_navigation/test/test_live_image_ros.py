import base64
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import subprocess
import threading

import pytest

from test_snapshot_ros import rgbd_server


pytestmark = pytest.mark.skipif(os.environ.get("WHEELARM_ROS_TEST") != "1",
                                reason="Set WHEELARM_ROS_TEST=1 in an isolated ROS domain")


@pytest.mark.parametrize("mode", ["fresh", "system_time", "stale", "missing"])
def test_live_cli_one_frame_without_tf(rgbd_server, tmp_path, mode):
    rgbd_server["publish_tf"] = False
    rgbd_server["stale"] = mode == "stale"
    images = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            images.append(base64.b64decode(payload["messages"][-1]["images"][0]))
            result = {"summary": "测试画面", "objects": [], "uncertainties": ["仅用于接口测试"]}
            body = json.dumps({"done": True, "message": {"content": json.dumps(result)}}).encode()
            self.send_response(200)
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        topic = "/missing_camera" if mode == "missing" else "/camera/image"
        result = subprocess.run([
            "ros2", "run", "wheel_arm_navigation", "scene_describe",
            "--observations-dir", str(tmp_path), "--wait", "2", "--rgb-topic", topic,
            "--endpoint", f"http://127.0.0.1:{server.server_port}",
            *(["--ros-args", "-p", "use_sim_time:=false"] if mode == "system_time" else [])],
            text=True, capture_output=True, timeout=15)
        if mode in ("fresh", "system_time"):
            assert result.returncode == 0, result.stdout + result.stderr
            assert len(images) == 1
            record = json.loads(next(tmp_path.glob("*/metadata.json")).read_text())
            rgb = next(tmp_path.glob("*/rgb.png")).read_bytes()
            assert images[0] == rgb
            assert record["requested_stamp_ns"] <= (
                record["stamps"]["rgb"]["sec"] * 1_000_000_000
                + record["stamps"]["rgb"]["nanosec"])
            assert "transforms" not in record and "camera_info" not in record
            report = json.loads(next(tmp_path.glob("*/descriptions/*.json")).read_text())
            assert report["source_mode"] == "live_once" and report["reference_frame"] is None
        else:
            assert result.returncode == 1, result.stdout + result.stderr
            assert not images and not list(tmp_path.iterdir())
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()
