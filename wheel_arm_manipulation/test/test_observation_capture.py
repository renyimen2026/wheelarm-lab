import json

import cv2
import numpy as np
import pytest
from sensor_msgs.msg import Image, JointState

from wheel_arm_manipulation.observation_capture import (
    ARM_JOINTS, FINGER_JOINTS, CaptureDiagnostics, EpisodeRecorder, ObservationCapture,
    extract_state, observation, save_observation,
)


def messages():
    rgb = Image(width=4, height=3, encoding="rgb8")
    rgb.header.stamp.sec = 10
    rgb.header.frame_id = "camera_optical_frame"
    joints = JointState(name=["left_wheel_joint", *ARM_JOINTS, *FINGER_JOINTS],
                        position=[999., .1, -1., 2., -1., .03, .02])
    joints.header.stamp.sec = 10
    joints.header.stamp.nanosec = 20_000_000
    return rgb, joints


def test_state_uses_joint_names_and_sums_fingers():
    _, joints = messages()
    pairs = list(zip(joints.name, joints.position))[::-1]
    joints.name = [name for name, _ in pairs]
    joints.position = [value for _, value in pairs]
    joints.effort = [float("nan")]*len(pairs)
    assert extract_state(joints) == pytest.approx([.1, -1., 2., -1., .05])


@pytest.mark.parametrize("invalid", ["missing", "duplicate", "length", "nan", "finger"])
def test_invalid_joint_state_is_rejected(invalid):
    _, joints = messages()
    if invalid == "missing":
        joints.name[-1] = "unknown"
    elif invalid == "duplicate":
        joints.name[0] = joints.name[1]
    elif invalid == "length":
        joints.position.pop()
    elif invalid == "nan":
        joints.position[1] = float("nan")
    else:
        joints.position[-1] = .05
    with pytest.raises(ValueError):
        extract_state(joints)


def test_observation_has_explicit_units_timestamps_and_no_action_label():
    rgb, joints = messages()
    data = observation(rgb, joints, 10_100_000_000)
    assert data["timestamp_difference_ms"] == 20.
    assert data["state"]["units"] == ["rad"]*4+["m"]
    assert data["kind"] == "observation_only"
    assert "action" not in data


@pytest.mark.parametrize("stamp,now", [(0, 10_100_000_000), (9_000_000_000, 10_100_000_000),
                                      (10_200_000_000, 10_100_000_000),
                                      (10_080_000_000, 10_100_000_000)])
def test_zero_stale_future_or_unsynchronized_images_are_rejected(stamp, now):
    rgb, joints = messages()
    rgb.header.stamp.sec, rgb.header.stamp.nanosec = divmod(stamp, 1_000_000_000)
    with pytest.raises(ValueError):
        observation(rgb, joints, now)


def test_saved_png_preserves_color_and_json_and_never_overwrites(tmp_path):
    rgb, joints = messages()
    metadata = observation(rgb, joints, 10_100_000_000)
    bgr = np.zeros((3, 4, 3), dtype=np.uint8)
    bgr[:, :, 2] = 255
    first = save_observation(tmp_path, bgr, metadata)
    second = save_observation(tmp_path, bgr, metadata)
    assert first != second
    assert np.array_equal(cv2.imread(str(first/"rgb.png")), bgr)
    assert json.loads((first/"observation.json").read_text()) == metadata


def test_wrong_image_shape_is_not_saved(tmp_path):
    rgb, joints = messages()
    with pytest.raises(ValueError):
        save_observation(tmp_path, np.zeros((2, 4, 3), dtype=np.uint8),
                         observation(rgb, joints, 10_100_000_000))
    assert not list(tmp_path.iterdir())


def frame(stamp):
    rgb, joints = messages()
    for message in (rgb, joints):
        message.header.stamp.sec, message.header.stamp.nanosec = divmod(stamp, 1_000_000_000)
    return observation(rgb, joints, stamp+10_000_000)


def test_episode_streams_numbered_frames_and_reports_actual_simulation_rate(tmp_path):
    recorder = EpisodeRecorder(tmp_path, 10.)
    bgr = np.zeros((3, 4, 3), dtype=np.uint8)
    for stamp in (10_000_000_000, 10_030_000_000, 10_100_000_000, 10_200_000_000):
        if recorder.due(stamp):
            recorder.append(bgr, frame(stamp))
    # Records are already readable before the episode is closed.
    lines = (recorder.path/"frames.jsonl").read_text().splitlines()
    assert len(lines) == 3
    for index, line in enumerate(lines):
        data = json.loads(line)
        assert data["frame_index"] == index
        assert "action" not in data
        assert np.array_equal(cv2.imread(str(recorder.path/data["image"]["file"])), bgr)
    summary = recorder.close("ctrl_c")
    assert recorder.stream.closed
    assert summary["collection_status"] == "finished"
    assert summary["actual_hz"] == pytest.approx(10.)
    assert summary["sample_span_sim_seconds"] == pytest.approx(.2)
    assert summary["max_frame_gap_sim_seconds"] == pytest.approx(.1)
    assert json.loads((recorder.path/"summary.json").read_text()) == summary


def test_missing_slots_are_not_filled_with_duplicate_frames(tmp_path):
    recorder = EpisodeRecorder(tmp_path, 10.)
    try:
        bgr = np.zeros((3, 4, 3), dtype=np.uint8)
        for stamp in (10_000_000_000, 10_350_000_000):
            assert recorder.due(stamp)
            recorder.append(bgr, frame(stamp))
        assert recorder.count == 2
        assert not recorder.due(10_350_000_000)
        assert not recorder.due(10_360_000_000)
        assert recorder.due(10_400_000_000)
    finally:
        summary = recorder.close("data_timeout", failed=True)
    assert summary["actual_hz"] == pytest.approx(1/.35)
    assert summary["collection_status"] == "failed"


def test_clock_reset_stops_the_episode(tmp_path):
    recorder = EpisodeRecorder(tmp_path, 10.)
    try:
        assert recorder.due(10_000_000_000)
        with pytest.raises(ValueError, match="backwards"):
            recorder.due(1_000_000_000)
    finally:
        recorder.close("clock_reset", failed=True)


@pytest.mark.parametrize("hz", [0., -1., float("nan"), float("inf"), 101.])
def test_invalid_recording_rate_does_not_create_files(tmp_path, hz):
    with pytest.raises(ValueError):
        EpisodeRecorder(tmp_path, hz)
    assert not list(tmp_path.iterdir())


def test_one_frame_has_no_fabricated_frequency(tmp_path):
    recorder = EpisodeRecorder(tmp_path, 10.)
    recorder.append(np.zeros((3, 4, 3), dtype=np.uint8), frame(10_000_000_000))
    summary = recorder.close("ctrl_c")
    assert summary["frames"] == 1
    assert summary["actual_hz"] is None


def test_ctrl_c_finishes_episode_without_shutting_ros_down_mid_write(tmp_path, monkeypatch):
    import signal
    from types import SimpleNamespace
    from wheel_arm_manipulation import observation_capture as capture

    recorder = EpisodeRecorder(tmp_path, 10.)
    node = SimpleNamespace(result=None, recorder=recorder, last_saved=capture.time.monotonic(),
                           reason="waiting", destroy_node=lambda: None)
    monkeypatch.setattr(capture, "ObservationCapture", lambda options: node)
    monkeypatch.setattr(capture.rclpy, "init", lambda **kwargs: None)
    monkeypatch.setattr(capture.rclpy, "ok", lambda: True)
    monkeypatch.setattr(capture.rclpy, "shutdown", lambda: None)

    def spin_once(node, **kwargs):
        recorder.append(np.zeros((3, 4, 3), dtype=np.uint8), frame(10_000_000_000))
        signal.getsignal(signal.SIGINT)(signal.SIGINT, None)

    monkeypatch.setattr(capture.rclpy, "spin_once", spin_once)
    old_handler = signal.getsignal(signal.SIGINT)
    assert capture.main(["observation_capture", "--record"]) == 0
    assert signal.getsignal(signal.SIGINT) == old_handler
    assert recorder.stream.closed
    summary = json.loads((recorder.path/"summary.json").read_text())
    assert summary["frames"] == 1
    assert summary["end_reason"] == "ctrl_c"
    assert summary["collection_status"] == "finished"


def test_diagnostics_distinguish_received_pairs_skipped_invalid_and_saved(tmp_path):
    from types import SimpleNamespace

    diagnostics = CaptureDiagnostics()
    bgr = np.zeros((3, 4, 3), dtype=np.uint8)
    options = SimpleNamespace(record=True, record_references=False, output_dir=tmp_path, hz=10., instruction="拿起杯子",
                              rgb_topic="/camera/image", joint_states_topic="/wheel_arm/joint_states")
    node = SimpleNamespace(result=None, options=options, diagnostics=diagnostics, recorder=None,
                           bridge=SimpleNamespace(imgmsg_to_cv2=lambda image, encoding: bgr),
                           get_clock=lambda: SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=10_300_000_000)),
                           get_parameter=lambda name: SimpleNamespace(value=True))
    for offset in (0, 30_000_000, 100_000_000, 200_000_000):
        rgb, joints = messages()
        rgb.header.stamp.nanosec = offset
        joints.header.stamp.nanosec = offset
        if offset == 200_000_000:
            joints.name[-1] = "missing"
        diagnostics.received_rgb(rgb)
        diagnostics.received_joints(joints)
        ObservationCapture.receive(node, rgb, joints)

    summary = node.recorder.close("ctrl_c")
    diag = summary["diagnostics"]
    assert diag["counts"] == {"rgb_received": 4, "joint_states_received": 4, "paired": 4,
                              "invalid_pairs": 1, "sampling_skipped": 1,
                              "conversion_failed": 0, "frames_saved": 2}
    assert all(entry["count"] == 2 for entry in diag["timings"].values())
    assert all(entry["mean_ms"] >= 0 and entry["max_ms"] >= entry["mean_ms"]
               for entry in diag["timings"].values())
    assert json.loads((node.recorder.path/"summary.json").read_text())["diagnostics"] == diag


def test_rgb_callback_rate_uses_wall_arrival_intervals(monkeypatch):
    from wheel_arm_manipulation import observation_capture as capture

    times = iter((1_000_000_000, 1_100_000_000, 1_400_000_000))
    monkeypatch.setattr(capture.time, "monotonic_ns", lambda: next(times))
    stats = CaptureDiagnostics()
    for _ in range(3):
        stats.received_rgb(None)
    report = stats.report()
    assert report["rgb_callback_wall_hz"] == pytest.approx(5.)
    assert report["rgb_max_callback_wall_gap_ms"] == pytest.approx(300.)


@pytest.mark.parametrize("reliability", ["best_effort", "reliable"])
def test_rgb_reliability_option_does_not_change_joint_or_reference_subscriptions(monkeypatch, reliability):
    from types import SimpleNamespace
    from wheel_arm_manipulation import observation_capture as capture
    from wheel_arm_manipulation.image_qos import image_qos
    from rclpy.qos import qos_profile_sensor_data

    subscriptions, references = [], []
    monkeypatch.setattr(capture.Node, "__init__", lambda *_a, **_kw: None)

    def subscriber(node, message_type, topic, **kwargs):
        subscriptions.append((message_type, topic, kwargs["qos_profile"]))
        return SimpleNamespace(registerCallback=lambda *_: None)

    monkeypatch.setattr(capture.message_filters, "Subscriber", subscriber)
    monkeypatch.setattr(capture.message_filters, "ApproximateTimeSynchronizer",
                        lambda *_a, **_kw: SimpleNamespace(registerCallback=lambda *_: None))
    monkeypatch.setattr(capture.Node, "create_subscription",
                        lambda self, kind, topic, callback, qos: references.append(qos))
    options = SimpleNamespace(rgb_topic="/camera/image", joint_states_topic="/wheel_arm/joint_states",
                              rgb_reliability=reliability, record_references=True,
                              arm_reference_topic="arm_reference", gripper_reference_topic="gripper_reference")
    node = ObservationCapture(options)
    assert subscriptions[0][2] == image_qos(reliability)
    assert subscriptions[1][2] == qos_profile_sensor_data
    assert references == [qos_profile_sensor_data]*2
    assert node.diagnostics.report()["rgb_subscription_qos"]["reliability"] == reliability


def test_valid_pair_ages_preserve_freshness_measurement_and_summary_configuration(tmp_path):
    diagnostics = CaptureDiagnostics("reliable")
    rgb, joints = messages()
    metadata = observation(rgb, joints, 10_100_000_000)
    assert metadata["image_age_ms"] == 100.
    assert metadata["joints_age_ms"] == 80.
    diagnostics.valid_pair(metadata)
    diagnostics.valid_pair(observation(rgb, joints, 10_200_000_000))
    recorder = EpisodeRecorder(tmp_path, 15., diagnostics)
    summary = recorder.close("duration_reached")
    diag = summary["diagnostics"]
    assert diag["rgb_subscription_qos"]["reliability"] == "reliable"
    assert diag["valid_pair_age_relative_to_node_ros_clock"]["image_age_ms"] == {
        "count": 2, "mean_ms": 150., "max_ms": 200.}
    with pytest.raises(ValueError, match="stale"):
        observation(rgb, joints, 10_600_000_000)
