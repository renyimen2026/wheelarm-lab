import json
from types import SimpleNamespace

from control_msgs.msg import JointTrajectoryControllerState
import pytest

from wheel_arm_manipulation.controller_reference import (
    ARM_JOINTS, FINGER_JOINTS, ControllerReferenceRecorder, controller_reference,
)
from wheel_arm_manipulation.observation_capture import EpisodeRecorder


def message(group="arm", field="reference"):
    result = JointTrajectoryControllerState()
    result.header.stamp.sec = 10
    result.joint_names = list(ARM_JOINTS if group == "arm" else FINGER_JOINTS)[::-1]
    values = [0., -.6, .9, -.3] if group == "arm" else [.03, .02]
    getattr(result, field).positions = values[::-1]
    # Actual position deliberately differs from the commanded reference.
    result.actual.positions = [0.]*len(values)
    return result


@pytest.mark.parametrize("field", ["reference", "desired"])
def test_reference_uses_named_targets_and_supports_legacy_messages(field):
    data = controller_reference(message(field=field), "arm", 10_100_000_000, "ros_sim_time")
    assert data["positions"] == [0., -.6, .9, -.3]
    assert data["names"] == list(ARM_JOINTS)
    assert data["source_field"] == field
    assert data["semantics"] == "current_interpolated_position_reference"


def test_older_message_without_reference_field_is_supported():
    msg = message(field="desired")
    old = SimpleNamespace(header=msg.header, joint_names=msg.joint_names, desired=msg.desired)
    assert controller_reference(old, "arm", 10_100_000_000, "ros_sim_time")["source_field"] == "desired"


def test_gripper_reference_preserves_both_fingers_and_total_opening():
    data = controller_reference(message("gripper"), "gripper", 10_100_000_000, "ros_sim_time")
    assert data["positions"] == [.03, .02]
    assert data["gripper_width_m"] == pytest.approx(.05)
    assert data["units"] == ["m", "m"]


@pytest.mark.parametrize("bad", ["empty", "nonfinite", "duplicate", "stale", "future"])
def test_invalid_references_are_counted_and_not_written(tmp_path, bad):
    recorder = ControllerReferenceRecorder(tmp_path, {"arm": "/arm", "gripper": "/gripper"})
    msg = message()
    if bad == "empty":
        msg.reference.positions = []
    elif bad == "nonfinite":
        msg.reference.positions[0] = float("nan")
    elif bad == "duplicate":
        msg.joint_names[0] = msg.joint_names[1]
    elif bad == "stale":
        msg.header.stamp.sec = 9
    else:
        msg.header.stamp.sec = 11
    recorder.append(msg, "arm", 10_100_000_000, "ros_sim_time")
    result = recorder.close()
    assert result["rejected"]["arm"] == 1
    assert result["saved"]["arm"] == 0
    assert not (tmp_path/"controller_references.jsonl").read_text()


def test_stream_is_readable_and_duplicate_timestamps_are_not_repeated(tmp_path):
    recorder = ControllerReferenceRecorder(tmp_path, {"arm": "/arm", "gripper": "/gripper"})
    for group in ("arm", "arm", "gripper"):
        recorder.append(message(group), group, 10_100_000_000, "ros_sim_time")
    rows = [json.loads(line) for line in (tmp_path/"controller_references.jsonl").read_text().splitlines()]
    assert len(rows) == 2
    assert rows[0]["topic"] == "/arm"
    result = recorder.close()
    assert result["both_controllers_recorded"]
    assert result["aligned_action_labels"] is False


def test_missing_controller_marks_collection_incomplete(tmp_path):
    episode = EpisodeRecorder(tmp_path, 10.)
    episode.references = ControllerReferenceRecorder(episode.path, {"arm": "/arm", "gripper": "/gripper"})
    episode.references.append(message(), "arm", 10_100_000_000, "ros_sim_time")
    result = episode.close("ctrl_c")
    assert result["collection_status"] == "failed"
    assert result["controller_references"]["saved"] == {"arm": 1, "gripper": 0}
    assert episode.references.stream.closed


def test_controller_reference_clock_reset_is_rejected(tmp_path):
    recorder = ControllerReferenceRecorder(tmp_path, {"arm": "/arm", "gripper": "/gripper"})
    try:
        recorder.append(message(), "arm", 10_100_000_000, "ros_sim_time")
        msg = message()
        msg.header.stamp.sec = 1
        with pytest.raises(ValueError, match="backwards"):
            recorder.append(msg, "arm", 1_100_000_000, "ros_sim_time")
    finally:
        recorder.close()
