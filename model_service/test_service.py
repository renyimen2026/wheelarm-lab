import base64
from http.server import ThreadingHTTPServer
import io
import json
from pathlib import Path
import sys
import threading
from urllib.error import HTTPError
from urllib.request import ProxyHandler, Request, build_opener

from PIL import Image
import pytest

from model_service.protocol import ALIASES, decode_image, model_info, strict_json, validate_chat
from model_service.server import handler_for

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "wheel_arm_navigation"))


def payload(**changes):
    return {"model": "qwen3:4b", "messages": [{"role": "user", "content": "hello"}],
            "stream": False, "think": False, "keep_alive": 0, **changes}


@pytest.mark.parametrize("changes", [
    {"stream": True}, {"think": True}, {"messages": []}, {"messages": [{}]},
    {"messages": [{"role": "tool", "content": "x"}]},
    {"messages": [{"role": "user", "content": None}]},
    {"options": {"num_predict": True}}, {"options": {"num_predict": 1025}},
    {"options": {"num_ctx": 4097}}, {"options": {"temperature": float("nan")}},
    {"options": {"temperature": True}}, {"options": {"unknown": 0}},
    {"keep_alive": "2m"}, {"format": "json"},
    {"messages": [{"role": "user", "content": "x"*24001}]},
    {"messages": [{"role": "user", "content": "x", "images": ["http://example.com"]}]},
])
def test_reject_invalid_requests(changes):
    with pytest.raises(ValueError):
        validate_chat(payload(**changes), False)


@pytest.mark.parametrize("value", ['{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}'])
def test_strict_json(value):
    with pytest.raises(ValueError):
        strict_json(value)


def test_image_conversion_and_bounds():
    image = Image.new("RGB", (4, 3), "red")
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    encoded = base64.b64encode(stream.getvalue()).decode()
    result, _, _, _ = validate_chat(payload(model="qwen3-vl:4b-instruct",
        messages=[{"role": "user", "content": "question", "images": [encoded]}]), True)
    assert result[0]["content"][0]["image"].size == (4, 3)
    assert result[0]["content"][1]["text"] == "question"
    with pytest.raises(ValueError):
        validate_chat(payload(messages=[{"role": "user", "content": "x", "images": [encoded, encoded]}]), True)
    for invalid in ["http://example.com", "invalid", base64.b64encode(b"not an image").decode()]:
        with pytest.raises(ValueError):
            decode_image(invalid)


@pytest.fixture
def root(tmp_path):
    for name, _ in ALIASES.values():
        directory = tmp_path / name
        directory.mkdir()
        (directory / "config.json").write_text("{}")
        (directory / "model.safetensors").write_bytes(b"fixture")
    return tmp_path


def test_allowlisted_local_models(root):
    assert model_info(payload(), root) == (root / "Qwen3-4B", False)
    with pytest.raises(ValueError):
        model_info(payload(model="../../arbitrary"), root)
    (root / "Qwen3-4B/model.safetensors").unlink()
    with pytest.raises(ValueError):
        model_info(payload(), root)


def test_incomplete_shards_fail_preflight(root):
    path = root / "Qwen3-4B"
    (path / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {"a": "model-00001.safetensors", "b": "model-00002.safetensors"}}))
    (path / "model-00001.safetensors").write_bytes(b"fixture")
    with pytest.raises(ValueError, match="missing"):
        model_info(payload(), root)
    (path / "model-00002.safetensors").write_bytes(b"fixture")
    assert model_info(payload(), root)[0] == path


@pytest.fixture
def server(root):
    class Engine:
        keep_loaded = False
        started, release = threading.Event(), threading.Event()
        blocking = False
        result = {}
        loads = []

        def load(self, path, vision):
            self.loads.append((path, vision))
            if self.blocking:
                self.started.set()
                assert self.release.wait(5)

        def unload(self):
            pass

        def chat(self, body, path, vision):
            validate_chat(body, vision)
            if self.blocking:
                self.started.set()
                assert self.release.wait(5)
            return {"done": True, "done_reason": "stop", "message": {
                "content": json.dumps(self.result, ensure_ascii=False)}}

    engine = Engine()
    http = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(engine, root))
    thread = threading.Thread(target=http.serve_forever)
    thread.start()
    def request(route, body=None):
        req = Request(f"http://127.0.0.1:{http.server_port}" + route,
            data=None if body is None else json.dumps(body).encode(),
            headers={"Content-Type": "application/json"})
        opener = build_opener(ProxyHandler({}))
        try:
            with opener.open(req, timeout=10) as response:
                return response.status, json.load(response)
        except HTTPError as response:
            return response.code, json.load(response)
    request.endpoint = f"http://127.0.0.1:{http.server_port}"
    try:
        yield engine, request
    finally:
        engine.release.set()
        http.shutdown()
        http.server_close()
        thread.join(timeout=5)


def test_protocol_routes(server):
    _, request = server
    assert request("/health")[1]["backend"] == "pytorch-cuda"
    assert request("/api/show", payload(model="qwen3-vl:4b-instruct"))[1]["capabilities"] == ["completion", "vision"]
    assert request("/api/chat", payload())[1]["done"] is True
    assert request("/api/chat", payload(stream=True))[0] == 400
    assert request("/api/show", payload(model="cloud-model"))[0] == 400
    assert request("/other", payload())[0] == 404


def test_preload_requires_residency_and_advertises_support(server):
    engine, request = server
    assert request("/api/preload", payload())[0] == 400
    assert request("/api/show", payload())[1]["features"] == []
    engine.keep_loaded = True
    assert request("/api/show", payload())[1]["features"] == ["preload"]
    response = request("/api/preload", payload(model="qwen3-vl:4b-instruct"))
    assert response[0] == 200 and response[1]["status"] == "ready"
    assert engine.loads[-1][1] is True
    assert request("/api/chat", payload())[0] == 200


def test_background_load_serializes_model_access(server):
    engine, request = server
    engine.keep_loaded = engine.blocking = True
    worker = threading.Thread(target=lambda: request("/api/preload", payload()))
    worker.start()
    try:
        assert engine.started.wait(3)
        assert request("/api/chat", payload())[0] == 503
        assert request("/api/preload", payload())[0] == 503
    finally:
        engine.release.set()
        worker.join(timeout=5)


def test_busy_request_is_rejected_not_queued(server):
    engine, request = server
    engine.blocking = True
    worker = threading.Thread(target=lambda: request("/api/chat", payload()))
    worker.start()
    try:
        assert engine.started.wait(3)
        assert request("/api/chat", payload())[0] == 503
    finally:
        engine.release.set()
        worker.join(timeout=5)


def test_existing_text_client_connects_unchanged(server):
    from wheel_arm_navigation import llm
    engine, request = server
    engine.result = {"status": "ready", "message": "依次导航", "goals": ["厨房", "卧室"],
                     "stay": 0, "repeat": 1}
    document = {"frame_id": "map", "goals": {"厨房": {"x": 1, "y": 2, "yaw": 0},
                                               "卧室": {"x": 3, "y": 4, "yaw": 0}}}
    _, plan, _ = llm.parse_instruction("先去厨房再去卧室", document, endpoint=request.endpoint)
    assert plan["goals"] == ["厨房", "卧室"]
    engine.result["goals"] = ["未记录房间"]
    with pytest.raises(ValueError):
        llm.parse_instruction("去厨房", document, endpoint=request.endpoint)


def test_existing_observe_and_vision_clients_connect_unchanged(server):
    from wheel_arm_navigation import observe_llm, vision
    engine, request = server
    assert "vision" in observe_llm.check_vision_model("qwen3-vl:4b-instruct", request.endpoint, 5)["capabilities"]
    engine.result = {"status": "ready", "message": "导航观察", "goal": "厨房", "question": "能看到椅子吗？"}
    document = {"frame_id": "map", "goals": {"厨房": {"x": 1, "y": 2, "yaw": 0}}}
    _, task, _ = observe_llm.parse_instruction("去厨房看看有没有椅子", document, endpoint=request.endpoint)
    assert task["goal"] == "厨房"
    engine.result = {"summary": "可见椅子", "objects": [], "uncertainties": []}
    image = io.BytesIO()
    Image.new("RGB", (4, 3)).save(image, format="PNG")
    result, _ = vision.describe(image.getvalue(), endpoint=request.endpoint)
    assert result["summary"] == "可见椅子"


@pytest.mark.parametrize("preload,keep", [(None, False), (None, True), ("text", False), ("vision", False), ("default", False)])
def test_preload_happens_before_listening_and_implies_residency(monkeypatch, root, preload, keep):
    from model_service import engine as engine_module, server as server_module
    calls = []

    class Engine:
        def __init__(self, resident, timeout):
            self.keep_loaded = resident
            calls.append(("init", resident))

        def load(self, path, vision):
            calls.append(("load", path.name, vision))

        def unload(self):
            calls.append(("unload",))

    class HTTP:
        def __init__(self, address, handler):
            calls.append(("listen",))

        def serve_forever(self):
            raise KeyboardInterrupt()

        def server_close(self):
            calls.append(("close",))

    args = ["server", "--models-root", str(root)]
    if preload != "default":
        args.extend(["--preload", preload or "none"])
    if preload == "default":
        preload = "text"
    if keep:
        args.append("--keep-loaded")
    monkeypatch.setattr(sys, "argv", args)
    monkeypatch.setattr(engine_module, "CudaEngine", Engine)
    monkeypatch.setattr(server_module, "ThreadingHTTPServer", HTTP)
    server_module.main()
    expected = [("init", keep or preload is not None)]
    if preload:
        expected.append(("load", "Qwen3-4B" if preload == "text" else "Qwen3-VL-4B-Instruct", preload == "vision"))
    assert calls == expected + [("listen",), ("close",), ("unload",)]


@pytest.mark.parametrize("missing", [True, False])
def test_preload_failure_never_listens_and_cleans_up(monkeypatch, root, missing):
    from model_service import engine as engine_module, server as server_module
    calls = []

    class Engine:
        def __init__(self, resident, timeout):
            pass

        def load(self, path, vision):
            raise RuntimeError("CUDA load failed")

        def unload(self):
            calls.append("unload")

    if missing:
        (root / "Qwen3-4B/model.safetensors").unlink()
    monkeypatch.setattr(sys, "argv", ["server", "--models-root", str(root), "--preload", "text"])
    monkeypatch.setattr(engine_module, "CudaEngine", Engine)
    monkeypatch.setattr(server_module, "ThreadingHTTPServer", lambda *args: calls.append("listen"))
    with pytest.raises((ValueError, RuntimeError)):
        server_module.main()
    assert calls == ["unload"]
