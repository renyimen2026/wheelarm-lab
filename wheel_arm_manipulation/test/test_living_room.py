from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from wheel_arm_manipulation.living_room import layout, write_world
from wheel_arm_manipulation.living_room_fixture import fixture_scene


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "wheel_arm_manipulation/config/living_room.yaml"
GAZEBO = ROOT / "wheel_arm_gazebo"


def test_dock_faces_the_cup_and_preserves_existing_house(tmp_path):
    settings, original, _, bounds, cup, base = layout(CONFIG, GAZEBO)
    local = base[:3, :3].T @ (cup-base[:3, 3])
    assert settings["name"] == "客厅"
    offset = np.array([settings["cup_world_x_offset"], 0., 0.])
    assert np.allclose(local, np.array([.64, 0, .415947])+base[:3, :3].T @ offset)
    baseline = tmp_path / "baseline.yaml"
    baseline.write_text(CONFIG.read_text().replace("cup_world_x_offset: 0.05", "cup_world_x_offset: 0.0"))
    _, _, _, _, original_cup, original_base = layout(baseline, GAZEBO)
    assert np.allclose(cup-original_cup, [.05, 0., 0.])
    assert np.allclose(base, original_base)
    source = (GAZEBO / "worlds/indoor_house.sdf").read_bytes()
    write_world(CONFIG, GAZEBO, tmp_path / "house.sdf")
    new = ET.parse(tmp_path / "house.sdf").getroot().find("world")
    old = original.getroot().find("world")
    assert len(list(new)) == len(list(old))+2
    assert new.find("include/name").text == "test_cup"
    assert new.find("model[@name='living_room_dock']//collision") is None
    assert (GAZEBO / "worlds/indoor_house.sdf").read_bytes() == source
    scene = fixture_scene(CONFIG, GAZEBO)
    assert len(scene.world.collision_objects) == len(bounds)+1
    assert all(obj.header.frame_id == "base_footprint" for obj in scene.world.collision_objects)
    assert any(obj.id == "grasp_fixture_table" for obj in scene.world.collision_objects)


def test_cup_cannot_be_placed_outside_table(tmp_path):
    path = tmp_path / "invalid.yaml"
    path.write_text(CONFIG.read_text().replace("cup_offset: [0.35, 0.20]", "cup_offset: [2.0, 0.20]"))
    with pytest.raises(ValueError, match="inside the tabletop"):
        layout(path, GAZEBO)


def test_navigation_uses_explicit_map_reference_and_actual_footprint():
    from wheel_arm_manipulation.house_navigation import reference, planar, parameters, validate_goal
    alignment = ROOT / "wheel_arm_manipulation/config/house_map_reference.yaml"
    map_world, _, prior = reference(alignment)
    dock = planar(map_world @ layout(CONFIG, GAZEBO)[-1])
    assert dock["x"] == pytest.approx(-1.64611636)
    assert dock["y"] == pytest.approx(-1.88998006)
    validate_goal(CONFIG, alignment, GAZEBO, dock)
    with pytest.raises(ValueError, match="停靠点"):
        validate_goal(CONFIG, alignment, GAZEBO, {**dock, "x": dock["x"]+.3})
    document = parameters(GAZEBO / "config/nav2_params.yaml", prior)
    assert document["amcl"]["ros__parameters"]["initial_pose"]["x"] == 0.
    assert document["controller_server"]["ros__parameters"]["general_goal_checker"]["xy_goal_tolerance"] == .04
    assert "0.265" in document["local_costmap"]["local_costmap"]["ros__parameters"]["footprint"]


def test_fixture_updates_for_current_base_pose():
    from wheel_arm_manipulation.living_room import pose_matrix
    dock = layout(CONFIG, GAZEBO)[-1]
    moved = dock @ pose_matrix([.03, 0., 0., 0., 0., 0.])
    original = fixture_scene(CONFIG, GAZEBO)
    updated = fixture_scene(CONFIG, GAZEBO, moved)
    for a, b in zip(original.world.collision_objects[:-1], updated.world.collision_objects[:-1]):
        assert b.pose.position.x == pytest.approx(a.pose.position.x-.03)
