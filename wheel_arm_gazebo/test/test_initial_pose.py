import importlib.util
import math
from pathlib import Path

import pytest
import yaml
from launch import LaunchContext


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("localization_launch", ROOT / "launch/nav_localization.launch.py")
localization = importlib.util.module_from_spec(spec)
spec.loader.exec_module(localization)


def test_no_prior_preserves_manual_mode():
    assert localization.resolve_prior("", ["", "", ""]) is None


def test_explicit_prior():
    assert localization.resolve_prior("", ["1.5", "-2", "0.6"]) == {
        "frame_id": "map", "x": 1.5, "y": -2.0, "yaw": 0.6}


def test_prior_file(tmp_path):
    path = tmp_path / "pose.yaml"
    path.write_text("frame_id: map\nx: 1.5\ny: -2\nyaw: 0.6\n")
    assert localization.resolve_prior(str(path), ["", "", ""])["y"] == -2
    with pytest.raises(ValueError, match="not both"):
        localization.resolve_prior(str(path), ["1", "2", "3"])


@pytest.mark.parametrize("coordinates", [["1", "", "0"], ["", "2", ""],
                                           ["nan", "0", "0"], ["inf", "0", "0"],
                                           ["word", "0", "0"]])
def test_invalid_coordinates(coordinates):
    with pytest.raises(ValueError):
        localization.resolve_prior("", coordinates)


@pytest.mark.parametrize("value", [None, [], {"x": 1},
    {"frame_id": "odom", "x": 0, "y": 0, "yaw": 0},
    {"frame_id": "map", "x": True, "y": 0, "yaw": 0},
    {"frame_id": "map", "x": "1", "y": 0, "yaw": 0},
    {"frame_id": "map", "x": 0, "y": 0, "yaw": math.nan},
    {"frame_id": "map", "x": 0, "y": 0, "yaw": 0, "spawn_x": 3.5}])
def test_invalid_prior_files(value):
    with pytest.raises(ValueError):
        localization.validate_prior(value)


@pytest.mark.parametrize("filename", ["nav2_params.yaml", "nav2_params_raw_odom.yaml"])
def test_only_amcl_is_overridden(filename):
    path = ROOT / "config" / filename
    original_bytes = path.read_bytes()
    original = yaml.safe_load(original_bytes)
    prior = localization.resolve_prior("", ["1.5", "-2", "0.6"])
    changed = localization.parameters_with_prior(path, prior)
    parameters = changed["amcl"]["ros__parameters"]
    assert parameters["set_initial_pose"] is True
    assert parameters["initial_pose"] == {"x": 1.5, "y": -2.0, "z": 0.0, "yaw": 0.6}
    original["amcl"]["ros__parameters"].update(
        set_initial_pose=True, initial_pose=parameters["initial_pose"])
    assert changed == original and path.read_bytes() == original_bytes


@pytest.mark.parametrize("ekf", ["true", "false"])
def test_launch_selects_correct_parameters_and_cleans_up(tmp_path, ekf):
    map_path = tmp_path / "map.yaml"
    map_path.write_text("image: map.pgm\n")
    context = LaunchContext()
    context.launch_configurations.update({
        "map": str(map_path), "use_ekf": ekf, "initial_pose_file": "",
        "initial_pose_x": "1", "initial_pose_y": "2", "initial_pose_yaw": "0.3",
        "params_file": str(ROOT / "config/nav2_params.yaml"),
        "raw_odom_params_file": str(ROOT / "config/nav2_params_raw_odom.yaml")})
    actions = localization.launch_localization(context)
    args = dict(actions[-1].launch_arguments)
    selected = "params_file" if ekf == "true" else "raw_odom_params_file"
    assert selected in args and args["slam"] == "False"
    path = Path(args[selected])
    try:
        parameters = yaml.safe_load(path.read_text())
        assert parameters["amcl"]["ros__parameters"]["set_initial_pose"] is True
        expected_odom = "/odometry/filtered" if ekf == "true" else "/wheel_arm/diff_drive_base_controller/odom"
        assert parameters["bt_navigator"]["ros__parameters"]["odom_topic"] == expected_odom
    finally:
        localization.remove_parameters(context, str(path))
    assert not path.exists()
