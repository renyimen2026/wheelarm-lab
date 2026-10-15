from types import SimpleNamespace

import pytest

from wheel_arm_navigation import live_image, scene_describe, vision
from wheel_arm_navigation.observation import save_rgb_capture
from test_vision import observation


def frame(stamp=10_000_000_000, width=640):
    return SimpleNamespace(header=SimpleNamespace(stamp=SimpleNamespace(
        sec=stamp // 1_000_000_000, nanosec=stamp % 1_000_000_000)), width=width, height=480)


def test_fresh_frame():
    assert live_image.validate_frame(frame(), 10_100_000_000, 9_900_000_000, 0.5) == 10_000_000_000


@pytest.mark.parametrize("stamp,now,requested,width", [
    (0, 10_000_000_000, 9_000_000_000, 640),
    (8_000_000_000, 10_000_000_000, 9_000_000_000, 640),
    (9_000_000_000, 10_000_000_000, 8_000_000_000, 640),
    (11_000_000_000, 10_000_000_000, 9_000_000_000, 640),
    (10_000_000_000, 10_000_000_000, None, 640),
    (10_000_000_000, 10_000_000_000, 9_000_000_000, 3000),
])
def test_bad_frames(stamp, now, requested, width):
    with pytest.raises(ValueError):
        live_image.validate_frame(frame(stamp, width), now, requested, 0.5)


def test_live_default_and_offline_option():
    assert scene_describe.parser().parse_args([]).observation is None
    assert scene_describe.parser().parse_args(["observations/id"]).observation == "observations/id"


def test_rgb_capture_can_be_replayed(observation, tmp_path):
    original, metadata, image = vision.read_observation(observation)
    root, record = save_rgb_capture(tmp_path / "live", image, {
        "observation_kind": "rgb", "image_size": metadata["camera_info"],
        "capture_mode": "live_once"})
    replay_root, replay_meta, replay_image = vision.read_observation(root)
    assert replay_image == image and replay_root == root and replay_meta == record
    assert {file.name for file in root.iterdir()} == {"metadata.json", "rgb.png"}
