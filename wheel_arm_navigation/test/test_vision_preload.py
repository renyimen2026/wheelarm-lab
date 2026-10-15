import threading

import pytest

from wheel_arm_navigation import observe_llm, vision_preload


def make():
    progress, stop = [], {"value": False}
    preload = vision_preload.VisionPreload("vision-model", "http://localhost:11435", 30,
                                         progress.append, lambda: stop["value"])
    return preload, progress, stop


def test_disabled_preload_never_calls_service(monkeypatch):
    monkeypatch.setattr(vision_preload, "preload_model", lambda *args: pytest.fail("Unexpected load"))
    preload, _, _ = make()
    preload.start()
    assert preload.wait(1) is False
    assert preload.snapshot()["status"] == "disabled"


def test_loading_is_nonblocking_and_only_starts_once(monkeypatch):
    started, release = threading.Event(), threading.Event()
    calls = []
    def load(*args):
        calls.append(args)
        started.set()
        assert release.wait(3)
    monkeypatch.setattr(vision_preload, "preload_model", load)
    preload, _, _ = make()
    preload.enable(True)
    try:
        preload.start()
        assert started.wait(1)
        assert preload.snapshot()["status"] == "loading"
        preload.start()
        assert len(calls) == 1
    finally:
        release.set()
    assert preload.wait(2) is True
    assert preload.snapshot()["status"] == "ready"


def test_preload_failure_remains_optional(monkeypatch):
    def fail(*args):
        raise RuntimeError("GPU allocation failed")
    monkeypatch.setattr(vision_preload, "preload_model", fail)
    preload, progress, _ = make()
    preload.enable(True)
    preload.start()
    assert preload.wait(2) is False
    assert preload.snapshot()["reason"] == "GPU allocation failed"
    assert any("导航继续" in message for message in progress)


@pytest.mark.parametrize("cancel", [True, False])
def test_wait_can_be_canceled_or_times_out(monkeypatch, cancel):
    release = threading.Event()
    monkeypatch.setattr(vision_preload, "preload_model", lambda *args: release.wait(3))
    preload, _, stop = make()
    preload.enable(True)
    preload.start()
    stop["value"] = cancel
    try:
        with pytest.raises(KeyboardInterrupt if cancel else RuntimeError):
            preload.wait(0.1)
    finally:
        preload.close()
        release.set()
        assert preload.done.wait(2)


def test_stopped_or_closed_client_never_starts(monkeypatch):
    monkeypatch.setattr(vision_preload, "preload_model", lambda *args: pytest.fail("Unexpected load"))
    for closed in (True, False):
        preload, _, stop = make()
        preload.enable(True)
        if closed:
            preload.close()
        else:
            stop["value"] = True
        preload.start()
        assert preload.thread is None


@pytest.mark.parametrize("features,backend,expected", [
    (["preload"], "pytorch-cuda", True), (["preload"], "ollama", False),
    (None, "pytorch-cuda", False), ([], "pytorch-cuda", False)])
def test_capability_negotiation(monkeypatch, features, backend, expected):
    monkeypatch.setattr(observe_llm, "request", lambda *args, **kwargs: {
        "capabilities": ["vision"], "features": features, "backend": backend})
    info = observe_llm.check_vision_model("vision", "http://localhost:11435", 30)
    assert info.get("preload_supported", False) == expected


@pytest.mark.parametrize("response", [{}, {"model": "wrong", "status": "ready", "backend": "pytorch-cuda"},
    {"model": "vision", "status": "loading", "backend": "pytorch-cuda"}])
def test_invalid_preload_response_rejected(monkeypatch, response):
    monkeypatch.setattr(observe_llm, "request", lambda *args, **kwargs: response)
    with pytest.raises(ValueError):
        observe_llm.preload_model("vision", "http://localhost:11435", 30)
