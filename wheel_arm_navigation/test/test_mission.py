import json
from types import SimpleNamespace

import pytest

from wheel_arm_navigation.mission import execute, load_task, validate_task


def outcome(status="succeeded", safe=True):
    return SimpleNamespace(status=status, reason="test", action_status=None,
                           safe_to_continue=safe)


def document():
    return {"frame_id": "map", "goals": {
        "客厅": {"x": 1.0, "y": 0.0, "yaw": 0.0},
        "卧室": {"x": 2.0, "y": 0.0, "yaw": 0.0},
    }}


def run(tmp_path, plan, navigate, stopping=lambda: False, pause=lambda _: True):
    code, path = execute(plan, document(), "goals.yaml", tmp_path,
                         navigate, stopping, pause, lambda _: None)
    return code, json.loads(path.read_text(encoding="utf-8"))


def test_order_repeat_and_stay(tmp_path):
    calls, rests = [], []

    def navigate(name, pose, frame, wait, timeout):
        current = json.loads(next(tmp_path.glob("*.json")).read_text(encoding="utf-8"))
        assert current["steps"][len(calls)]["status"] == "running"
        assert pose == document()["goals"][name]
        assert frame == "map" and wait == 10 and timeout == 180
        calls.append(name)
        return outcome()

    code, log = run(tmp_path, {"goals": ["客厅", "卧室"], "repeat": 2, "stay": 0.1},
                    navigate, pause=lambda seconds: rests.append(seconds) is None)
    assert code == 0
    assert calls == ["客厅", "卧室", "客厅", "卧室"]
    assert rests == [0.1] * 4
    assert log["status"] == "completed"
    assert [step["round"] for step in log["steps"]] == [1, 1, 2, 2]
    assert log["finished_at"] and log["elapsed_sec"] >= 0
    assert all(step["status"] == "succeeded" for step in log["steps"])
    assert log["summary"]["succeeded"] == 4 and log["summary"]["failed"] == 0


@pytest.mark.parametrize("policy,expected,status", [
    ("stop", ["客厅"], "failed"),
    ("skip", ["客厅", "卧室"], "completed_with_failures"),
])
def test_failure_policy(tmp_path, policy, expected, status):
    calls = []

    def navigate(name, *_):
        calls.append(name)
        return outcome("aborted" if name == "客厅" else "succeeded")

    code, log = run(tmp_path, {"goals": ["客厅", "卧室"], "on_failure": policy}, navigate)
    assert code == 1
    assert calls == expected
    assert log["status"] == status
    assert log["steps"][0]["reason"] == "test"
    assert log["summary"]["failed"] == 1


def test_unsafe_result_never_skipped(tmp_path):
    code, log = run(tmp_path, {"goals": ["客厅", "卧室"], "on_failure": "skip"},
                    lambda *_: outcome("timeout", safe=False))
    assert code == 1
    assert log["status"] == "unsafe_stop"
    assert log["steps"][1]["status"] == "not_run"


def test_safe_timeout_can_be_skipped(tmp_path):
    code, log = run(tmp_path, {"goals": ["客厅", "卧室"], "on_failure": "skip"},
                    lambda name, *_: outcome("timeout" if name == "客厅" else "succeeded"))
    assert code == 1 and log["status"] == "completed_with_failures"
    assert log["steps"][1]["status"] == "succeeded"


def test_cancel_during_navigation(tmp_path):
    code, log = run(tmp_path, {"goals": ["客厅", "卧室"], "on_failure": "skip"},
                    lambda *_: outcome("canceled"))
    assert code == 130 and log["status"] == "canceled"
    assert log["steps"][1]["status"] == "not_run"


def test_cancel_during_stay(tmp_path):
    code, log = run(tmp_path, {"goals": ["客厅", "卧室"], "stay": 2},
                    lambda *_: outcome(), pause=lambda _: False)
    assert code == 130 and log["status"] == "canceled"
    assert log["steps"][0]["status"] == "succeeded"
    assert log["steps"][1]["status"] == "not_run"


def test_unknown_goal_prevents_entire_task(tmp_path):
    with pytest.raises(ValueError, match="missing"):
        run(tmp_path, {"goals": ["客厅", "missing"]},
            lambda *_: pytest.fail("must not send any goal"))
    assert not list(tmp_path.glob("*.json"))


def test_navigation_exception_is_logged(tmp_path):
    def navigate(*_):
        raise RuntimeError("transport disconnected")

    code, log = run(tmp_path, {"goals": ["客厅", "卧室"]}, navigate)
    assert code == 1 and log["status"] == "unsafe_stop"
    assert log["steps"][0]["reason"] == "transport disconnected"


def test_no_writable_log_prevents_motion(tmp_path):
    file_path = tmp_path / "not_a_directory"
    file_path.write_text("x", encoding="utf-8")
    with pytest.raises(OSError):
        run(file_path, {"goals": ["客厅"]}, lambda *_: pytest.fail("must not move"))


@pytest.mark.parametrize("changes", [
    {"goals": []}, {"goals": "客厅"}, {"goals": [""]}, {"repeat": 0},
    {"repeat": True}, {"repeat": 1.5}, {"stay": -1}, {"stay": float("nan")},
    {"timeout": 0}, {"wait": float("inf")}, {"name": ""}, {"on_failure": "retry"},
    {"typo": 1},
])
def test_invalid_task(changes):
    with pytest.raises(ValueError):
        validate_task({"goals": ["客厅"], **changes})


def test_load_task(tmp_path):
    path = tmp_path / "task.yaml"
    path.write_text("name: patrol\ngoals: [客厅, 卧室]\nstay: 1\n", encoding="utf-8")
    plan = load_task(path)
    assert plan["goals"] == ["客厅", "卧室"] and plan["stay"] == 1
    assert plan["on_failure"] == "stop"
