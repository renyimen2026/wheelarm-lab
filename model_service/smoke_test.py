"""Exercise actual CUDA weights and existing clients, without any robot actions."""

import argparse
import json
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import time
from urllib.request import ProxyHandler, build_opener


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-root", type=Path, default=Path("~/Models/transformers"))
    parser.add_argument("--image", type=Path)
    parser.add_argument("--port", type=int, default=11545)
    parser.add_argument("--preload", choices=["none", "text", "vision"], default="none")
    parser.add_argument("--overlap-vision", action="store_true",
                        help="Test background vision loading while the main thread stays active")
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "wheel_arm_navigation"))
    from wheel_arm_navigation import llm, observe_llm, vision

    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(("127.0.0.1", args.port))
    endpoint = f"http://127.0.0.1:{args.port}"
    opener = build_opener(ProxyHandler({}))
    document = {"frame_id": "map", "goals": {"厨房": {"x": 1, "y": 2, "yaw": 0},
                                               "卧室": {"x": 3, "y": 4, "yaw": 0}}}
    with tempfile.NamedTemporaryFile(prefix="wheelarm-cuda-smoke-", suffix=".log", delete=False) as log:
        command = [sys.executable, "-m", "model_service.server", "--port", str(args.port),
                   "--models-root", str(args.models_root.expanduser()), "--keep-loaded"]
        if args.preload:
            command.extend(["--preload", args.preload])
        process = subprocess.Popen(command, stdout=log, stderr=log)
        print("Server log:", log.name, flush=True)
        background = None
        try:
            deadline = time.monotonic() + 60
            while True:
                if process.poll() is not None:
                    raise RuntimeError("CUDA service exited; inspect server log")
                try:
                    with opener.open(endpoint + "/health", timeout=2) as response:
                        assert json.load(response)["backend"] == "pytorch-cuda"
                    break
                except OSError:
                    if time.monotonic() > deadline:
                        raise
                    time.sleep(0.3)
            for label in ("cold", "warm"):
                result, plan, elapsed = llm.parse_instruction("先去厨房再去卧室", document,
                                                             endpoint=endpoint, timeout=600)
                assert plan and plan["goals"] == ["厨房", "卧室"]
                print(json.dumps({"test": "text_" + label, "elapsed_sec": elapsed, "result": result},
                                 ensure_ascii=False), flush=True)
            result, task, elapsed = observe_llm.parse_instruction("去厨房看看有没有椅子", document,
                                                                  endpoint=endpoint, timeout=600)
            assert task and task["goal"] == "厨房"
            print(json.dumps({"test": "observe_parser", "elapsed_sec": elapsed, "result": result},
                             ensure_ascii=False), flush=True)
            if args.image:
                info = observe_llm.check_vision_model("qwen3-vl:4b-instruct", endpoint, 30)
                if args.overlap_vision:
                    from wheel_arm_navigation.vision_preload import VisionPreload
                    assert info.get("preload_supported")
                    background = VisionPreload("qwen3-vl:4b-instruct", endpoint, 120, print, lambda: False)
                    background.enable(True)
                    started = time.monotonic()
                    background.start()
                    returned = time.monotonic() - started
                    assert returned < 1.0
                    print("Main thread continues during model load; no navigation commands are sent", flush=True)
                    time.sleep(2)
                    assert background.wait(120)
                    print(json.dumps({"test": "background_vision_preload", "start_return_sec": returned,
                                      "state": background.snapshot()}, ensure_ascii=False), flush=True)
                _, _, image = vision.read_observation(args.image.parent)
                for label in ("cold", "warm"):
                    result, elapsed = vision.describe(image, "画面里有哪些家具？", endpoint=endpoint, timeout=600)
                    print(json.dumps({"test": "vision_" + label, "elapsed_sec": elapsed, "result": result},
                                     ensure_ascii=False), flush=True)
        finally:
            if background is not None:
                background.close()
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
                try:
                    process.wait(timeout=180)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
            print("Test server stopped", flush=True)


if __name__ == "__main__":
    main()
