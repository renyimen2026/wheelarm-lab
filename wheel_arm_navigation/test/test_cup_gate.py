import json

import pytest

from wheel_arm_navigation import chat_llm, cup_gate


DOCUMENT = {"frame_id": "map", "goals": {"客厅": {"x": 0., "y": 0., "yaw": 0.},
                                           "厨房": {"x": 1., "y": 0., "yaw": 0.}}}


@pytest.mark.parametrize("status,count", [("not_found", 0), ("uncertain", 1), ("found", 2)])
def test_no_grasp_without_unique_positive_confirmation(status, count):
    with pytest.raises(RuntimeError, match="禁止抓取"):
        cup_gate.require_cup({"status": status, "count": count, "evidence": "测试画面"})


def test_positive_confirmation():
    cup_gate.require_cup({"status": "found", "count": 1, "evidence": "桌上可见白色无柄杯"})


@pytest.mark.parametrize("result", [None, {}, {"status": "found", "count": True, "evidence": "杯子"},
    {"status": "found", "count": 0, "evidence": "杯子"},
    {"status": "not_found", "count": 1, "evidence": "杯子"}])
def test_malformed_confirmation_rejected(result):
    with pytest.raises(ValueError):
        cup_gate.require_cup(result)


def test_real_image_is_sent_to_vision_model(monkeypatch):
    image = b"\x89PNG\r\n\x1a\nimage-test"
    calls = []
    def request(*args):
        calls.append(args)
        return {"done": True, "message": {"content": json.dumps({"status": "found", "count": 1, "evidence": "杯子"})}}
    monkeypatch.setattr(cup_gate, "request", request)
    result, _ = cup_gate.confirm(image, "vision", "http://127.0.0.1:11434", 60)
    assert calls[0][4]["messages"][1]["images"]
    assert calls[0][1] == "vision"
    cup_gate.require_cup(result)


def test_task_intents_and_destination_allowlist():
    response = {"status": "ready", "message": "拿杯子", "goal": "客厅", "question": "是否可见杯子？", "action": "pick_cup"}
    assert chat_llm.validate(response, DOCUMENT)["action"] == "pick_cup"
    with pytest.raises(ValueError, match="客厅"):
        chat_llm.validate({**response, "goal": "厨房"}, DOCUMENT)
    with pytest.raises(ValueError):
        chat_llm.validate({**response, "action": "run_code"}, DOCUMENT)
