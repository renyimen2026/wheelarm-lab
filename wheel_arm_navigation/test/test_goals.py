import math

import pytest

from wheel_arm_navigation.goals import load, save_pose, validate


def test_save_reload_and_overwrite(tmp_path):
    path = tmp_path / "maps" / "goals.yaml"
    save_pose(path, "客厅", 1.0, -2.0, math.pi)
    assert load(path)["goals"]["客厅"]["y"] == -2
    with pytest.raises(ValueError, match="overwrite"):
        save_pose(path, "客厅", 0, 0, 0)
    save_pose(path, "厨房", 3, 2, 0)
    save_pose(path, "客厅", 0, 0, 0, overwrite=True)
    assert set(load(path)["goals"]) == {"客厅", "厨房"}
    assert load(path)["goals"]["客厅"]["x"] == 0


@pytest.mark.parametrize("value", [None, True, "1.0", math.nan, math.inf])
def test_invalid_coordinates(value):
    with pytest.raises(ValueError):
        validate({"frame_id": "map", "goals": {"room": {"x": value, "y": 0, "yaw": 0}}})


@pytest.mark.parametrize("document", [None, [], {}, {"frame_id": "odom", "goals": {}},
                                       {"frame_id": "map", "goals": []},
                                       {"frame_id": "map", "goals": {"": {}}}])
def test_invalid_schema(document):
    with pytest.raises(ValueError):
        validate(document)


def test_failed_update_preserves_original(tmp_path):
    path = tmp_path / "goals.yaml"
    save_pose(path, "room", 1, 2, 0)
    before = path.read_bytes()
    with pytest.raises(ValueError):
        save_pose(path, "bad", math.inf, 0, 0)
    assert path.read_bytes() == before


def test_invalid_yaml_has_clear_error(tmp_path):
    path = tmp_path / "goals.yaml"
    path.write_text("goals: [", encoding="utf-8")
    with pytest.raises(ValueError, match="YAML"):
        load(path)
