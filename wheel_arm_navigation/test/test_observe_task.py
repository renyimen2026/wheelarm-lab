import io
import json
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from wheel_arm_navigation import language_observe, observe_llm, observe_task
from wheel_arm_navigation.arrival_capture import Stillness, near_goal
from wheel_arm_navigation.semantic_goal import NavigationResult


@pytest.fixture
def document():
    return {"frame_id": "map", "goals": {"厨房": {"x": 1.0, "y": 2.0, "yaw": 0.0}}}


def parsed(**changes):
    return {"status": "ready", "message": "到达后单帧观察", "goal": "厨房",
            "question": "当前画面中是否能看到椅子？", **changes}


@pytest.mark.parametrize("changes", [
    {"goal": "厨房门口"}, {"goal": ["厨房"]}, {"question": ""}, {"question": "x" * 501},
    {"status": "execute"}, {"message": ""}, {"x": 1}, {"question": None},
    {"status": "clarify"}, {"status": "unsupported"}])
def test_bad_task_response(document, changes):
    with pytest.raises(ValueError):
        observe_llm.validate_response(parsed(**changes), document)


def test_valid_and_non_ready(document):
    assert observe_llm.validate_response(parsed(), document)["goal"] == "厨房"
    assert observe_llm.validate_response(parsed(status="clarify", goal="", question=""), document) is None


def fake_http(monkeypatch, body):
    calls = []
    class Opener:
        def open(self, req, timeout):
            calls.append((req.full_url, json.loads(req.data), timeout))
            return io.BytesIO(body)
    monkeypatch.setattr(observe_llm, "build_opener", lambda *_: Opener())
    return calls


def test_parse_http(document, monkeypatch):
    calls = fake_http(monkeypatch, json.dumps({"done": True, "message": {
        "content": json.dumps(parsed())}}).encode())
    _, task, _ = observe_llm.parse_instruction("去厨房看看椅子", document)
    assert task["goal"] == "厨房"
    payload = calls[0][1]
    assert payload["format"]["properties"]["goal"]["enum"] == ["", "厨房"]
    assert payload["think"] is False and payload["keep_alive"] == 0


@pytest.mark.parametrize("body", [b"invalid", b"x" * 65537, b'{"done":false}',
    b'{"done":true,"done_reason":"length"}', b'{"done":true,"message":[]}',
    b'{"done":true,"message":{"content":"{}"}}'])
def test_malformed_parse(document, monkeypatch, body):
    fake_http(monkeypatch, body)
    with pytest.raises(ValueError):
        observe_llm.parse_instruction("去厨房看看椅子", document)


@pytest.mark.parametrize("details", [{"capabilities": ["completion"]}, {},
    {"capabilities": ["vision"], "remote_model": "remote"},
    {"capabilities": ["vision"], "remote_host": "https://example.com"}])
def test_unusable_vision_model(monkeypatch, details):
    fake_http(monkeypatch, json.dumps(details).encode())
    with pytest.raises(ValueError):
        observe_llm.check_vision_model("qwen3-vl:4b-instruct", "http://localhost:11434", 10)


def test_vision_preflight(monkeypatch):
    calls = fake_http(monkeypatch, b'{"capabilities":["completion","vision"]}')
    assert "vision" in observe_llm.check_vision_model("local", "http://localhost:11434", 10)["capabilities"]
    assert calls[0][0].endswith("/api/show")


@pytest.mark.parametrize("kwargs", [{"endpoint": "http://remote"}, {"endpoint": "http://localhost/path"},
    {"model": "model-cloud"}, {"timeout": float("nan")}, {"timeout": 601}, {"timeout": True}])
def test_bad_service_options(document, kwargs):
    with pytest.raises(ValueError):
        observe_llm.parse_instruction("去厨房看看椅子", document, **kwargs)


def test_stillness_requires_continuous_fresh_low_speed():
    motion = Stillness()
    motion.update(1_000_000_000, 0, 0, 1_000_000_000, 1)
    assert not motion.ready(1_000_000_000, 1)
    motion.update(1_300_000_000, 0, 0, 1_300_000_000, 1.3)
    motion.update(1_600_000_000, 0, 0, 1_600_000_000, 1.6)
    assert motion.ready(1_600_000_000, 1.6)
    assert not motion.ready(2_500_000_000, 2.5)
    motion.update(1_700_000_000, 0.2, 0, 1_700_000_000, 1.7)
    assert not motion.ready(1_700_000_000, 1.7)


@pytest.mark.parametrize("stamp,speed", [(0, 0), (1_000_000_000, float("nan")),
                                        (3_000_000_000, 0)])
def test_invalid_motion(stamp, speed):
    motion = Stillness()
    motion.update(stamp, speed, 0, 1_000_000_000, 1)
    assert not motion.ready(1_000_000_000, 1)


def test_duplicate_and_gapped_odometry_resets_stability():
    motion = Stillness()
    for stamp in (1_000_000_000, 1_300_000_000, 1_600_000_000):
        motion.update(stamp, 0, 0, stamp, stamp / 1e9)
    motion.update(1_600_000_000, 0, 0, 1_600_000_000, 1.6)
    assert not motion.ready(1_600_000_000, 1.6)
    motion.update(3_000_000_000, 0, 0, 3_000_000_000, 3)
    assert not motion.ready(3_000_000_000, 3)


def test_pose_mismatch():
    transform = SimpleNamespace(transform=SimpleNamespace(
        translation=SimpleNamespace(x=1, y=2), rotation=SimpleNamespace(x=0, y=0, z=0, w=1)))
    assert near_goal(transform, {"x": 1, "y": 2, "yaw": 0})
    assert not near_goal(transform, {"x": 3, "y": 2, "yaw": 0})
    assert not near_goal(transform, {"x": 1, "y": 2, "yaw": 1})


@pytest.mark.parametrize("mode", ["success", "nav_failed", "unsafe", "capture_failed",
                                  "vision_failed", "preflight_failed", "canceled", "capture_canceled", "vision_canceled"])
def test_execution_gates_and_journal(tmp_path, document, mode):
    options = language_observe.parser().parse_args(["去厨房看看椅子", "--log-dir", str(tmp_path / "logs"),
                                                   "--observations-dir", str(tmp_path / "images")])
    calls = []
    def preflight():
        calls.append("preflight")
        if mode == "preflight_failed":
            raise RuntimeError("missing model")
        return {"capabilities": ["vision"]}
    def navigate(*_args):
        calls.append("navigation")
        return NavigationResult("aborted" if mode == "nav_failed" else "succeeded",
                                safe_to_continue=mode != "unsafe")
    def capture(*_args):
        calls.append("capture")
        if mode == "capture_canceled":
            raise KeyboardInterrupt()
        if mode == "capture_failed":
            raise RuntimeError("no fresh odom")
        _, image = cv2.imencode(".png", np.zeros((3, 4, 3), dtype=np.uint8))
        return image.tobytes(), {"observation_kind": "rgb", "image_size": {"width": 4, "height": 3}}
    def infer(*_args):
        calls.append("vision")
        if mode == "vision_canceled":
            raise KeyboardInterrupt()
        if mode == "vision_failed":
            raise ValueError("bad model output")
        return {"summary": "当前视角没有确认椅子", "objects": [], "uncertainties": ["桌后被遮挡"]}, 1
    code, path = observe_task.execute({"goal": "厨房", "question": "是否看到椅子？"},
        document, "goals.yaml", options, navigate, capture, infer, preflight,
        lambda: mode == "canceled", lambda *_: None, {"confirmed": True})
    record = json.loads(path.read_text())
    assert code == (0 if mode == "success" else 130 if "canceled" in mode else 1)
    if mode in ("nav_failed", "unsafe", "preflight_failed", "canceled"):
        assert "capture" not in calls and "vision" not in calls
    if mode == "capture_failed":
        assert "vision" not in calls
    if mode == "capture_canceled":
        assert "vision" not in calls and record["status"] == "canceled"
    if mode == "vision_canceled":
        assert record["status"] == "canceled" and record["observation"]["directory"]
    if mode == "success":
        assert record["status"] == "completed" and record["observation"]["report"]
        assert calls == ["preflight", "navigation", "capture", "vision"]
    if mode == "vision_failed":
        assert record["status"] == "vision_failed" and record["observation"]["directory"]
        assert record["observation"]["result"] is None


@pytest.mark.parametrize("mode", ["preview", "decline", "confirm", "non_tty", "clarify"])
def test_cli_confirmation(monkeypatch, document, mode):
    monkeypatch.setattr(language_observe, "load", lambda *_: document)
    monkeypatch.setattr(observe_llm, "parse_instruction", lambda *_: (
        parsed(), None if mode == "clarify" else {"goal": "厨房", "question": "看到椅子吗？"}, 1))
    monkeypatch.setattr(language_observe.sys, "stdin", SimpleNamespace(isatty=lambda: mode != "non_tty"))
    monkeypatch.setattr("builtins.input", lambda *_: "y" if mode == "confirm" else "n")
    calls = []
    monkeypatch.setattr(language_observe, "run_execution", lambda *args: calls.append(args) or 0)
    options = language_observe.parser().parse_args(["去厨房看看椅子", *(["--preview"] if mode == "preview" else [])])
    if mode == "non_tty":
        with pytest.raises(ValueError):
            language_observe.run(options, ["language_observe"])
    else:
        assert language_observe.run(options, ["language_observe"]) == (1 if mode == "clarify" else 0)
    assert len(calls) == (1 if mode == "confirm" else 0)
