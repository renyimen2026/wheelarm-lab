import json
from types import SimpleNamespace

import pytest

from wheel_arm_manipulation.task_metadata import TaskMetadataRecorder, run_with_metadata, task_event


def episode(path):
    (path/"frames.jsonl").write_text(json.dumps({
        "clock": "ros_sim_time", "joints_stamp_ns": 10_000_000_000})+"\n")
    return path


def read(path):
    return json.loads((path/"task_metadata.json").read_text())


def harness(path, finish="place"):
    node = SimpleNamespace(snapshot=lambda: None, get_clock=lambda: SimpleNamespace(
        now=lambda: SimpleNamespace(nanoseconds=10_100_000_000)))
    options = SimpleNamespace(episode_dir=str(episode(path)), finish=finish)
    return node, options


def verified(node, options, confirm):
    task_event(node, "execution_started")
    task_event(node, "phase_started", phase="lift", motion_skipped=False)
    task_event(node, "lift_verified", height_delta_m=.08)
    task_event(node, "phase_finished", phase="lift", motion_skipped=False)
    if options.finish == "place":
        task_event(node, "placement_verified", position_error_m=.001)
    task_event(node, "procedure_completed")
    return 0


@pytest.mark.parametrize("finish", ["hold", "place"])
def test_success_is_explicitly_linked_and_preserves_source_data(tmp_path, finish):
    node, options = harness(tmp_path, finish)
    before = (tmp_path/"frames.jsonl").read_bytes()
    assert run_with_metadata(node, options, verified) == 0
    metadata = read(tmp_path)
    events = [json.loads(line) for line in (tmp_path/"task_events.jsonl").read_text().splitlines()]
    assert metadata["task_success"] is True
    assert metadata["episode_dir"] == str(tmp_path.resolve())
    assert metadata["placement_verified"] == (finish == "place")
    assert metadata["active_phase"] is None
    assert {e["task_id"] for e in events} == {metadata["task_id"]}
    assert [e["event_index"] for e in events] == list(range(len(events)))
    assert all(e["stamp_ns"] == 10_100_000_000 for e in events)
    assert events[-1]["event"] == "task_finished"
    assert (tmp_path/"frames.jsonl").read_bytes() == before


def test_unconfirmed_execution_is_cancelled_not_success(tmp_path):
    node, options = harness(tmp_path)
    run_with_metadata(node, options, lambda *_: 0)
    assert read(tmp_path)["execution_status"] == "cancelled"
    assert read(tmp_path)["task_success"] is None


@pytest.mark.parametrize("error,status,success", [
    (RuntimeError("motion failed"), "failed", False),
    (KeyboardInterrupt(), "interrupted", None),
])
def test_failed_or_interrupted_stage_stays_open_and_keeps_lift_evidence(tmp_path, error, status, success):
    node, options = harness(tmp_path)

    def fail(node, *_):
        task_event(node, "execution_started")
        task_event(node, "lift_verified", height_delta_m=.08)
        task_event(node, "phase_started", phase="lower")
        raise error

    with pytest.raises(type(error)):
        run_with_metadata(node, options, fail)
    metadata = read(tmp_path)
    assert metadata["execution_status"] == status
    assert metadata["task_success"] is success
    assert metadata["lift_verified"] is True
    assert metadata["active_phase"] == "lower"


def test_cleanup_failure_after_procedure_does_not_claim_success(tmp_path):
    node, options = harness(tmp_path)

    def fail(*args):
        verified(*args)
        raise RuntimeError("scene cleanup failed")

    with pytest.raises(RuntimeError):
        run_with_metadata(node, options, fail)
    assert read(tmp_path)["procedure_completed"] is True
    assert read(tmp_path)["task_success"] is False


@pytest.mark.parametrize("missing", ["lift_verified", "placement_verified", "procedure_completed"])
def test_zero_exit_without_required_evidence_cannot_label_success(tmp_path, missing):
    node, options = harness(tmp_path)

    def incomplete(node, *_):
        for kind in ("execution_started", "lift_verified", "placement_verified", "procedure_completed"):
            if kind != missing:
                task_event(node, kind)
        return 0

    run_with_metadata(node, options, incomplete)
    assert read(tmp_path)["task_success"] is False
    assert read(tmp_path)["execution_status"] == "failed"


def test_no_terminal_result_leaves_unknown_and_reuse_cannot_overwrite(tmp_path):
    episode(tmp_path)
    recorder = TaskMetadataRecorder(tmp_path, "place", lambda: 10_100_000_000)
    recorder.event("execution_started")
    recorder.stream.close()  # Simulate termination before finish().
    before = {file.name: file.read_bytes() for file in tmp_path.iterdir()}
    assert read(tmp_path)["task_success"] is None
    with pytest.raises(FileExistsError):
        TaskMetadataRecorder(tmp_path, "place", lambda: 10_100_000_000)
    assert {file.name: file.read_bytes() for file in tmp_path.iterdir()} == before


@pytest.mark.parametrize("invalid", ["missing_frames", "finished", "stale", "future", "clock"])
def test_invalid_episode_is_rejected_before_metadata_is_created(tmp_path, invalid):
    episode(tmp_path)
    now = 10_100_000_000
    if invalid == "missing_frames":
        (tmp_path/"frames.jsonl").unlink()
    elif invalid == "finished":
        (tmp_path/"summary.json").write_text("{}")
    elif invalid == "stale":
        now = 13_000_000_000
    elif invalid == "future":
        now = 9_000_000_000
    else:
        (tmp_path/"frames.jsonl").write_text('{"clock": "ros_time"}\n')
    with pytest.raises(ValueError):
        TaskMetadataRecorder(tmp_path, "place", lambda: now)
    assert not (tmp_path/"task_metadata.json").exists()


def test_clock_reset_rejects_events_and_leaves_result_unknown(tmp_path):
    episode(tmp_path)
    stamp = [10_100_000_000]
    recorder = TaskMetadataRecorder(tmp_path, "place", lambda: stamp[0])
    stamp[0] = 1_000_000_000
    with pytest.raises(ValueError, match="clock moved backwards"):
        recorder.finish("finished", "returned")
    assert read(tmp_path)["task_success"] is None
