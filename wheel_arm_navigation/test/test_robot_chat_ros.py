"""Persistent CLI tasks against isolated mock Nav2, camera and model services."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import pty
import select
import signal
import subprocess
import threading
import time

import pytest

from test_language_observe_ros import sensors
from test_ros_integration import server
from wheel_arm_navigation.goals import save_pose


pytestmark = pytest.mark.skipif(os.environ.get("WHEELARM_ROS_TEST") != "1",
                                reason="Set WHEELARM_ROS_TEST=1 in an unused ROS domain")


@pytest.mark.parametrize("mode", ["success", "decline", "rejected", "cancel", "unsafe_cancel",
                                  "capture_failure", "parse_failure"])
def test_persistent_terminal_flow(server, sensors, tmp_path, mode):
    goals = tmp_path / "goals.yaml"
    x = 3 if mode == "cancel" else 5 if mode == "unsafe_cancel" else 4 if mode == "rejected" else 1.25
    save_pose(goals, "厨房", x, -0.5, 0)
    sensors["speed"] = 0.2 if mode == "capture_failure" else 0.0
    visual_calls = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if self.path == "/api/show":
                response = {"capabilities": ["vision"]}
            elif "images" in body["messages"][-1]:
                visual_calls.append(body)
                result = {"summary": "当前画面没有确认杯子", "objects": [], "uncertainties": ["画面有遮挡"]}
                response = {"done": True, "message": {"content": json.dumps(result)}}
            else:
                result = {"status": "ready", "message": "到达后观察", "goal": "厨房", "question": "看到杯子吗？"}
                if mode == "parse_failure":
                    result.update(status="clarify", message="请明确地点", goal="", question="")
                response = {"done": True, "message": {"content": json.dumps(result)}}
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())

        def log_message(self, *_args):
            pass

    http = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=http.serve_forever)
    thread.start()
    master, slave = pty.openpty()
    output = bytearray()
    process = None

    def wait_for(marker, since=0, timeout=20):
        deadline = time.monotonic() + timeout
        wanted = marker.encode()
        while wanted not in output[since:]:
            assert time.monotonic() < deadline, output.decode(errors="replace")
            ready, _, _ = select.select([process.stdout], [], [], 0.1)
            if ready:
                data = os.read(process.stdout.fileno(), 65536)
                assert data, output.decode(errors="replace")
                output.extend(data)

    try:
        command = ["ros2", "run", "wheel_arm_navigation", "robot_chat",
            "--file", str(goals), "--log-dir", str(tmp_path / "logs"),
            "--observations-dir", str(tmp_path / "images"), "--capture-wait", "1.2",
            "--settle-duration", "0.1", "--endpoint", f"http://127.0.0.1:{http.server_port}"]
        if mode == "unsafe_cancel":
            command.extend(["--nav-timeout", "0.3"])
        process = subprocess.Popen(command, stdin=slave, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        os.close(slave)
        slave = None
        wait_for("你：")
        for number in range(2 if mode == "success" else 1):
            cursor = len(output)
            os.write(master, "去厨房看看有没有杯子\n".encode())
            if mode == "parse_failure":
                wait_for("你：", cursor)
                break
            wait_for("执行导航并单帧观察？[y/N]", cursor)
            cursor = len(output)
            os.write(master, b"n\n" if mode == "decline" else b"y\n")
            if mode == "cancel":
                assert server["accepted"].wait(10)
                os.killpg(process.pid, signal.SIGINT)
            if mode == "unsafe_cancel":
                wait_for("会话停止接收新任务", cursor)
                break
            if mode == "success":
                wait_for("[任务] 识别完成。", cursor)
            wait_for("你：", cursor)
        if mode != "unsafe_cancel":
            os.write(master, b"/quit\n")
        rest, _ = process.communicate(timeout=20)
        output.extend(rest)
        text = output.decode(errors="replace")
        assert process.returncode == (1 if mode == "unsafe_cancel" else 0), text
        records = [json.loads(path.read_text()) for path in (tmp_path / "logs").glob("*.json")]
        if mode == "success":
            assert len(records) == len(visual_calls) == 2, text
            assert all(record["status"] == "completed" for record in records)
            assert len({record["source"]["session_id"] for record in records}) == 1
            assert all(record["source"]["confirmed"] for record in records)
            markers = ["[任务] 任务解析中", "[任务] 指令解析完成", "[任务] 前往目标地点",
                       "[任务] 已到达", "[任务] 画面已采集，场景识别中", "[任务] 识别完成"]
            positions = [text.index(marker) for marker in markers]
            assert positions == sorted(positions)
            assert text.count("[任务] 识别完成") == 2
            assert "[不确定性] 画面有遮挡" in text
        elif mode in ("decline", "parse_failure"):
            assert not server["poses"] and not visual_calls and not records
            assert "[任务] 前往目标地点" not in text
        else:
            assert len(records) == 1 and not visual_calls
            expected = {"rejected": "failed", "cancel": "canceled", "unsafe_cancel": "unsafe_stop",
                        "capture_failure": "capture_failed"}[mode]
            assert records[0]["status"] == expected
            assert "[任务] 识别完成" not in text
            assert "[任务] 画面已采集，场景识别中" not in text
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
