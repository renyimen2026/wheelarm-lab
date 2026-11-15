"""A persistent ROS subprocess with atomic request/result files and bounded waits."""

import json
from pathlib import Path
import signal
import subprocess
import time


class RosProbeSession:
    def __init__(self, root, command=None):
        self.root = Path(root)
        self.root.mkdir()
        self.sequence = 0
        if command is None:
            command = ["/usr/bin/python3", str(Path(__file__).with_name("execute_vla_short_motion.py")),
                       "--session-dir", str(self.root), "--execute"]
        self.child = subprocess.Popen(command)
        try:
            ready = self.wait_json(self.root/"ready.json", 30)
            if ready.get("status") != "ready":
                raise RuntimeError(f"ROS 会话初始化失败：{ready}")
        except BaseException:
            self.close()
            raise

    def wait_json(self, path, timeout):
        deadline = time.monotonic()+timeout
        while time.monotonic() < deadline:
            if path.exists():
                return json.loads(path.read_text())
            if self.child.poll() is not None:
                raise RuntimeError(f"ROS 会话退出，code={self.child.returncode}，缺少 {path}")
            time.sleep(.01)
        raise TimeoutError(f"ROS 会话响应超时：{path}")

    def request(self, value, timeout=45):
        number = self.sequence
        self.sequence += 1
        path = self.root/f"request_{number:03d}.json"
        temporary = path.with_suffix(".tmp")
        with temporary.open("x") as stream:
            json.dump(value, stream, ensure_ascii=False, allow_nan=False)
        temporary.rename(path)
        result = self.wait_json(self.root/f"result_{number:03d}.json", timeout)
        if result.get("status") != "completed":
            raise RuntimeError(f"ROS 操作失败：{result}；结果文件 {self.root/f'result_{number:03d}.json'}")
        return result

    def close(self):
        if self.child.poll() is None:
            self.child.send_signal(signal.SIGINT)
            try:
                self.child.wait(timeout=45)
            except subprocess.TimeoutExpired:
                # Do not kill a process that may still be confirming motion cancellation.
                raise RuntimeError(f"ROS 进程 {self.child.pid} 未在取消等待期内退出，请检查仿真和控制器停止状态")
