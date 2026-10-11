import io
import json
import sys
from types import SimpleNamespace
from urllib.error import URLError

import pytest

from wheel_arm_navigation import language_goal, llm


@pytest.fixture
def document():
    return {"frame_id": "map", "goals": {
        "卧室": {"x": 1, "y": 2, "yaw": 0},
        "厨房": {"x": 3, "y": 4, "yaw": 0.5}}}


def ready(**changes):
    return {"status": "ready", "message": "依次导航", "goals": ["卧室", "厨房"],
            "stay": 0, "repeat": 1, **changes}


def test_valid_plan_preserves_order_and_duplicates(document):
    plan = llm.validate_response(ready(goals=["厨房", "卧室", "厨房"], repeat=2), document)
    assert plan["goals"] == ["厨房", "卧室", "厨房"]
    assert plan["repeat"] == 2 and plan["on_failure"] == "stop"
    assert plan["timeout"] == 180


@pytest.mark.parametrize("changes", [
    {"goals": ["厨房入口"]}, {"goals": []}, {"goals": [1]}, {"goals": "卧室"},
    {"goals": ["卧室"] * 11}, {"stay": True}, {"stay": -1}, {"stay": 61},
    {"stay": float("nan")}, {"stay": float("inf")}, {"repeat": True},
    {"repeat": 0}, {"repeat": 4}, {"repeat": 1.0}, {"status": "execute"},
    {"message": ""}, {"message": "x" * 301}, {"x": 1}, {"on_failure": "skip"},
])
def test_invalid_model_output_rejected(document, changes):
    with pytest.raises(ValueError):
        llm.validate_response(ready(**changes), document)


@pytest.mark.parametrize("status", ["clarify", "unsupported"])
def test_non_ready_has_no_plan(document, status):
    assert llm.validate_response(ready(status=status, goals=[]), document) is None
    with pytest.raises(ValueError):
        llm.validate_response(ready(status=status), document)


@pytest.mark.parametrize("value", ['{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}', '```json\n{}\n```'])
def test_strict_json(value):
    with pytest.raises(ValueError):
        llm.strict_json(value)


def fake_transport(monkeypatch, body):
    captured = {}

    class Opener:
        def open(self, request, timeout):
            captured.update(url=request.full_url, payload=llm.strict_json(request.data),
                            timeout=timeout)
            return io.BytesIO(body)

    def build(*handlers):
        captured["handlers"] = handlers
        return Opener()

    monkeypatch.setattr(llm, "build_opener", build)
    return captured


def test_http_payload_and_response(monkeypatch, document):
    captured = fake_transport(monkeypatch, json.dumps({
        "done": True, "done_reason": "stop", "message": {
            "content": json.dumps(ready(), ensure_ascii=False)}}).encode())
    result, plan, elapsed = llm.parse_instruction("先去卧室再去厨房", document)
    assert result["status"] == "ready" and plan["goals"] == ["卧室", "厨房"]
    payload = captured["payload"]
    assert captured["url"] == "http://127.0.0.1:11434/api/chat"
    assert payload["think"] is False and payload["stream"] is False
    assert payload["keep_alive"] == 0
    assert payload["format"]["properties"]["goals"]["items"]["enum"] == ["卧室", "厨房"]
    assert captured["handlers"][0].proxies == {}
    assert elapsed >= 0


@pytest.mark.parametrize("body", [
    b"x" * 65537, b"not json", b'{"done":false}',
    b'{"done":true,"message":{}}', b'{"done":true,"message":[]}',
    b'{"done":true,"done_reason":"length"}',
])
def test_bad_http_response(monkeypatch, document, body):
    fake_transport(monkeypatch, body)
    with pytest.raises(ValueError):
        llm.parse_instruction("去卧室", document)


@pytest.mark.parametrize("endpoint", [
    "http://example.com", "https://localhost:11434", "http://127.0.0.1/path",
    "http://user:pass@localhost", "http://localhost?x=1", "http://localhost/#x",
])
def test_reject_nonlocal_endpoint(document, endpoint):
    with pytest.raises(ValueError):
        llm.parse_instruction("去卧室", document, endpoint=endpoint)


@pytest.mark.parametrize("timeout", [0, -1, 601, float("nan"), True])
def test_timeout_bounds(document, timeout):
    with pytest.raises(ValueError):
        llm.parse_instruction("去卧室", document, timeout=timeout)


def test_no_redirect():
    with pytest.raises(ValueError, match="重定向"):
        llm.NoRedirect().redirect_request(None, None, 302, "", {}, "http://example.com")


def test_transport_failure(monkeypatch, document):
    class Opener:
        def open(self, *_args, **_kwargs):
            raise URLError("connection refused")
    monkeypatch.setattr(llm, "build_opener", lambda *_: Opener())
    with pytest.raises(RuntimeError, match="Ollama"):
        llm.parse_instruction("去卧室", document)


@pytest.mark.parametrize("instruction", ["", " ", "x" * 1001])
def test_instruction_limits(document, instruction):
    with pytest.raises(ValueError):
        llm.parse_instruction(instruction, document)


def test_empty_goals():
    with pytest.raises(ValueError, match="尚未记录"):
        llm.parse_instruction("去卧室", {"frame_id": "map", "goals": {}})


@pytest.mark.parametrize("mode", ["preview", "decline", "non_tty", "clarify", "confirm"])
def test_cli_confirmation_gate(monkeypatch, document, mode):
    monkeypatch.setattr(language_goal, "load", lambda _: document)
    result = ready()
    plan = llm.validate_response(result, document)
    if mode == "clarify":
        result = ready(status="clarify", goals=[])
        plan = None
    monkeypatch.setattr(language_goal, "parse_instruction", lambda *_: (result, plan, 1))
    monkeypatch.setattr(language_goal.sys, "stdin", SimpleNamespace(isatty=lambda: mode != "non_tty"))
    monkeypatch.setattr("builtins.input", lambda _: "y" if mode == "confirm" else "n")
    calls = []
    monkeypatch.setitem(sys.modules, "wheel_arm_navigation.mission_runner",
                        SimpleNamespace(run_plan=lambda *args, **kwargs:
                                        calls.append((args, kwargs)) or 0))
    args = ["去卧室", *(["--preview"] if mode == "preview" else [])]
    options = language_goal.parser().parse_args(args)
    if mode == "non_tty":
        with pytest.raises(ValueError, match="交互终端"):
            language_goal.run(options, ["language_goal"])
    else:
        assert language_goal.run(options, ["language_goal"]) == (1 if mode == "clarify" else 0)
    assert len(calls) == (1 if mode == "confirm" else 0)
    if calls:
        assert calls[0][1]["source"]["confirmed"] is True
