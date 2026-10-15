"""Optional network preload in a daemon thread; no ROS spinning or robot commands."""

import threading
import time

from wheel_arm_navigation.observe_llm import preload_model


class VisionPreload:
    def __init__(self, model, endpoint, timeout, progress, stopping):
        self.model, self.endpoint, self.timeout = model, endpoint, timeout
        self.progress, self.stopping = progress, stopping
        self.enabled = False
        self.lock = threading.Lock()
        self.done, self.closed = threading.Event(), threading.Event()
        self.thread = None
        self.state = {"status": "disabled", "model": model, "reason": "", "elapsed_sec": None}

    def enable(self, enabled):
        self.enabled = enabled
        with self.lock:
            self.state["status"] = "not_started" if enabled else "disabled"

    def snapshot(self):
        with self.lock:
            return dict(self.state)

    def start(self):
        if not self.enabled or self.thread is not None or self.closed.is_set() or self.stopping():
            return
        with self.lock:
            self.state["status"] = "loading"
        self.thread = threading.Thread(target=self._load, name="vision-preload", daemon=True)
        try:
            self.thread.start()
        except RuntimeError as error:
            with self.lock:
                self.state.update(status="failed", reason=str(error))
            self.done.set()
            raise
        self.progress("Nav2 已接受目标，后台预加载视觉模型；不阻塞导航")

    def _load(self):
        started = time.monotonic()
        try:
            preload_model(self.model, self.endpoint, self.timeout)
            status, reason = "ready", ""
        except Exception as error:
            status, reason = "failed", str(error)
        elapsed = round(time.monotonic() - started, 3)
        with self.lock:
            self.state.update(status=status, reason=reason, elapsed_sec=elapsed)
        self.done.set()
        if not self.closed.is_set():
            self.progress(f"视觉模型预加载完成，耗时 {elapsed:g} 秒" if status == "ready"
                          else f"视觉模型预加载失败；导航继续，到达后按原流程加载：{reason}")

    def wait(self, timeout):
        if self.thread is None:
            return False
        deadline = time.monotonic() + timeout
        while not self.done.wait(0.05):
            if self.stopping():
                raise KeyboardInterrupt()
            if time.monotonic() >= deadline:
                raise RuntimeError("视觉预加载仍未完成，等待超时；已保留拍摄图片")
        if self.stopping():
            raise KeyboardInterrupt()
        return self.snapshot()["status"] == "ready"

    def close(self):
        # A client exit cannot cancel a CUDA load already running on the server.
        self.closed.set()
