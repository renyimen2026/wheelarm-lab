from types import SimpleNamespace

import pytest

from wheel_arm_navigation import language_observe, observe_llm, robot_chat


@pytest.fixture(autouse=True)
def terminal(monkeypatch):
    monkeypatch.setattr(robot_chat.sys, "stdin", SimpleNamespace(isatty=lambda: True))


def reader(*items):
    values = iter(items)
    def read(prompt):
        value = next(values)
        if isinstance(value, BaseException):
            raise value
        return value
    return read


def test_parser_uses_cuda_service_and_no_positional_task():
    options = robot_chat.parser().parse_args([])
    assert options.endpoint.endswith(":11435")
    assert not hasattr(options, "instruction")
    assert language_observe.parser().parse_args(["task"]).endpoint.endswith(":11434")


def test_repeated_tasks_reuse_one_session_and_keep_options_unmodified():
    options = robot_chat.parser().parse_args([])
    calls, output = [], []
    def execute(args, argv, on_stage, session_id):
        calls.append((args.instruction, session_id))
        for stage in ("parsing", "parsed", "navigation_requested", "navigating", "arrived", "recognizing", "completed"):
            on_stage(stage, {"goal": "厨房", "question": "有没有杯子？", "result": {
                "summary": "当前画面没有确认杯子", "uncertainties": ["部分区域被遮挡"]}})
        on_stage("finished", {"status": "completed"})
        return 0
    assert robot_chat.run_session(options, ["robot_chat"],
        reader("去厨房看看杯子", "去卧室看看家具", "/quit"), output.append, execute) == 0
    assert [item[0] for item in calls] == ["去厨房看看杯子", "去卧室看看家具"]
    assert calls[0][1] == calls[1][1]
    assert not hasattr(options, "instruction")
    assert sum("识别完成" in item for item in output) == 2
    assert any("部分区域被遮挡" in item for item in output)
    positions = [next(index for index, item in enumerate(output) if word in item)
                 for word in ("任务解析中", "指令解析完成", "前往目标", "已到达", "场景识别中", "识别完成")]
    assert positions == sorted(positions)


def test_empty_help_and_quit_do_not_call_model():
    def fail(*args, **kwargs):
        pytest.fail("Unexpected model call")
    assert robot_chat.run_session(robot_chat.parser().parse_args([]), [],
        reader("", "/help", "/exit"), lambda *_: None, fail) == 0


@pytest.mark.parametrize("error", [ValueError("bad task"), RuntimeError("model unavailable"), KeyboardInterrupt(), EOFError()])
def test_error_before_navigation_allows_another_task(error):
    calls = []
    def execute(args, argv, **kwargs):
        calls.append(args.instruction)
        raise error
    assert robot_chat.run_session(robot_chat.parser().parse_args([]), [],
        reader("first", "second", "/quit"), lambda *_: None, execute) == 0
    assert calls == ["first", "second"]


@pytest.mark.parametrize("mode", ["unsafe_stop", "communication_error", "interrupt"])
def test_unknown_navigation_state_stops_session(mode):
    calls, output = [], []
    def execute(args, argv, on_stage, **kwargs):
        calls.append(args.instruction)
        on_stage("navigation_requested", {"goal": "厨房"})
        if mode == "communication_error":
            raise RuntimeError("lost connection")
        if mode == "interrupt":
            raise KeyboardInterrupt()
        on_stage("finished", {"status": "unsafe_stop"})
        return 1
    assert robot_chat.run_session(robot_chat.parser().parse_args([]), [],
        reader("first", "must not execute"), output.append, execute) == 1
    assert calls == ["first"]
    assert any("停止接收" in item for item in output)


def test_confirmed_cancel_allows_another_task():
    calls = []
    def execute(args, argv, on_stage, **kwargs):
        calls.append(args.instruction)
        on_stage("navigation_requested", {})
        on_stage("finished", {"status": "canceled"})
        return 130
    assert robot_chat.run_session(robot_chat.parser().parse_args([]), [],
        reader("first", "second", "/quit"), lambda *_: None, execute) == 0
    assert calls == ["first", "second"]


@pytest.mark.parametrize("interrupt,code", [(EOFError(), 0), (KeyboardInterrupt(), 130)])
def test_idle_interrupt_exits(interrupt, code):
    assert robot_chat.run_session(robot_chat.parser().parse_args([]), [],
        reader(interrupt), lambda *_: None) == code


def test_noninteractive_input_rejected(monkeypatch):
    monkeypatch.setattr(robot_chat.sys, "stdin", SimpleNamespace(isatty=lambda: False))
    with pytest.raises(ValueError):
        robot_chat.run_session(robot_chat.parser().parse_args([]), [])


@pytest.mark.parametrize("mode", ["preview", "decline", "confirm", "clarify"])
def test_chat_still_uses_real_confirmation_gate(monkeypatch, mode):
    document = {"frame_id": "map", "goals": {"厨房": {"x": 1, "y": 2, "yaw": 0}}}
    response = {"status": "ready", "message": "观察", "goal": "厨房", "question": "看到杯子吗？"}
    monkeypatch.setattr(language_observe, "load", lambda *_: document)
    monkeypatch.setattr(observe_llm, "parse_instruction", lambda *_: (
        response, None if mode == "clarify" else {"goal": "厨房", "question": "看到杯子吗？"}, 1))
    monkeypatch.setattr("builtins.input", lambda *_: "y" if mode == "confirm" else "n")
    calls = []
    monkeypatch.setattr(language_observe, "run_execution", lambda *args, **kwargs: calls.append((args, kwargs)) or 0)
    options = robot_chat.parser().parse_args(["--preview"] if mode == "preview" else [])
    assert robot_chat.run_session(options, ["robot_chat"], reader("去厨房看看杯子", "/quit"), lambda *_: None) == 0
    assert len(calls) == (1 if mode == "confirm" else 0)
    if calls:
        source = calls[0][0][-1]
        assert source["confirmed"] and source["interaction"] == "terminal_chat"
        assert source["session_id"]
