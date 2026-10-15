import json
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from wheel_arm_navigation import observation
from wheel_arm_navigation.scene_snapshot import parser, validate_headers


@pytest.fixture
def sample():
    rgb = np.zeros((3, 4, 3), dtype=np.uint8)
    rgb[:, :, 2] = 255
    depth = np.full((3, 4), 2.0, dtype=np.float32)
    metadata = {"camera_info": {"width": 4, "height": 3,
                               "k": [2, 0, 2, 0, 2, 1.5, 0, 0, 1]}, "label": "餐厅"}
    return rgb, depth, metadata


def header(stamp=1000000000, frame="camera_optical_frame"):
    return SimpleNamespace(header=SimpleNamespace(
        stamp=SimpleNamespace(sec=stamp // 1000000000, nanosec=stamp % 1000000000),
        frame_id=frame))


def test_metric_depth_conversion():
    meters = observation.depth_meters(np.array([[1500, 0]], dtype=np.uint16), "16UC1")
    assert meters.dtype == np.float32 and meters[0, 0] == 1.5
    values = np.array([[2, np.inf, np.nan]], dtype=np.float32)
    np.testing.assert_equal(observation.depth_meters(values, "32FC1"), values)


@pytest.mark.parametrize("encoding,dtype", [("mono8", np.uint8), ("16UC1", np.float32),
                                            ("32FC1", np.uint16)])
def test_unsupported_depth(encoding, dtype):
    with pytest.raises(ValueError):
        observation.depth_meters(np.ones((3, 4), dtype=dtype), encoding)


def test_save_complete_observation(tmp_path, sample):
    rgb, depth, metadata = sample
    depth[0, :3] = [np.inf, np.nan, 0]
    path = observation.save_observation(tmp_path, rgb, depth, metadata)
    assert sorted(file.name for file in path.iterdir()) == [
        "depth.npy", "depth_preview.png", "metadata.json", "rgb.png"]
    np.testing.assert_equal(np.load(path / "depth.npy", allow_pickle=False), depth)
    np.testing.assert_equal(cv2.imread(str(path / "rgb.png")), rgb)
    record = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    assert record["depth_unit"] == "meter" and record["label"] == "餐厅"
    assert record["depth_statistics"]["valid_fraction"] == 0.75
    assert record["depth_statistics"]["median_m"] == 2
    assert not list(tmp_path.glob(".observation-*"))


def test_empty_depth_is_recorded_not_invented(sample):
    _, depth, _ = sample
    depth[:] = np.inf
    preview, stats = observation.depth_preview(depth)
    assert not preview.any() and stats["valid_fraction"] == 0
    assert stats["median_m"] is None and stats["min_m"] is None


def test_failed_write_leaves_no_partial_record(tmp_path, sample, monkeypatch):
    monkeypatch.setattr(observation.cv2, "imwrite", lambda *_: False)
    with pytest.raises(OSError):
        observation.save_observation(tmp_path, *sample)
    assert not list(tmp_path.iterdir())


def test_existing_observations_are_not_overwritten(tmp_path, sample):
    first = observation.save_observation(tmp_path, *sample)
    second = observation.save_observation(tmp_path, *sample)
    assert first != second and first.exists() and second.exists()


@pytest.mark.parametrize("change", ["rgb_type", "shape", "uncalibrated", "info_size", "nan_k"])
def test_invalid_sample(tmp_path, sample, change):
    rgb, depth, metadata = sample
    if change == "rgb_type":
        rgb = rgb.astype(np.float32)
    elif change == "shape":
        depth = depth[:2]
    elif change == "uncalibrated":
        metadata["camera_info"]["k"][0] = 0
    elif change == "info_size":
        metadata["camera_info"]["width"] = 1
    elif change == "nan_k":
        metadata["camera_info"]["k"][0] = float("nan")
    with pytest.raises(ValueError):
        observation.save_observation(tmp_path, rgb, depth, metadata)
    assert not list(tmp_path.iterdir())


def test_fresh_synchronized_headers():
    options = parser().parse_args([])
    validate_headers(header(), header(), header(), 1100000000, 900000000, options)


@pytest.mark.parametrize("kind", ["old", "future", "before_request", "zero", "clock_zero",
                                  "unsynchronized", "wrong_frame"])
def test_bad_headers(kind):
    options = parser().parse_args([])
    rgb, depth, info = header(), header(), header()
    now, started = 1100000000, 900000000
    if kind == "old":
        now = 2000000000
    elif kind == "future":
        now = 999999999
    elif kind == "before_request":
        started = 1050000000
    elif kind == "zero":
        rgb = header(0)
    elif kind == "clock_zero":
        now = 0
    elif kind == "unsynchronized":
        depth = header(1030000000)
    elif kind == "wrong_frame":
        info = header(frame="camera_link")
    with pytest.raises(ValueError):
        validate_headers(rgb, depth, info, now, started, options)


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf"])
def test_positive_arguments(value):
    with pytest.raises(SystemExit):
        parser().parse_args(["--wait", value])
