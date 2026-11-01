import copy
from dataclasses import replace

from moveit_msgs.msg import AllowedCollisionEntry, AllowedCollisionMatrix, CollisionObject, PlanningScene
import numpy as np
import pytest

from wheel_arm_manipulation.cup_grasp import parser, straight_trajectory, unchanged, verify_lift
from wheel_arm_manipulation.grasp_kinematics import GraspKinematics
from wheel_arm_manipulation.grasp_scene import GraspScene, contact_matrix
from wheel_arm_manipulation.planning import load_model


@pytest.mark.parametrize("position", [[.62, .04, .60], [.7, -.04, .63], [.668, 0, .68]])
def test_horizontal_grasp_ik_uses_urdf_and_measured_target(position):
    model = replace(load_model(), tip="gripper_tcp")
    solver = GraspKinematics(model)
    goal = solver.solve(position, model.named["ready"])
    model.check(goal)
    tf = solver.forward(goal)
    assert np.linalg.norm(tf[:3, 3]-position) < .001
    assert np.allclose(tf[2, :3], [0, 0, 1], atol=.015)


def test_unreachable_grasp_is_rejected():
    model = load_model()
    with pytest.raises(ValueError):
        GraspKinematics(model).solve([5, 0, .6], model.named["ready"])


@pytest.mark.parametrize("delta", [[.08, 0, 0], [0, 0, .08], [-.08, 0, 0]])
def test_straight_segments_follow_measured_line_with_zero_endpoint_speed(delta):
    model = replace(load_model(), tip="gripper_tcp")
    solver = GraspKinematics(model)
    origin = np.array([.64, .02, .60])
    start = solver.solve(origin, model.named["ready"])
    direction = np.array(delta)
    trajectory, _, _ = straight_trajectory(solver, model, start, origin+direction, .2, .2)
    for point in trajectory.joint_trajectory.points:
        tf = solver.forward(dict(zip(model.joints, point.positions)))
        offset = tf[:3, 3]-origin
        fraction = np.clip(offset @ direction/(direction @ direction), 0, 1)
        assert np.linalg.norm(offset-fraction*direction) < .001
        assert np.allclose(tf[2, :3], [0, 0, 1], atol=.015)
    for point in (trajectory.joint_trajectory.points[0], trajectory.joint_trajectory.points[-1]):
        assert max(abs(v) for v in point.velocities) < 1e-8


def test_only_target_finger_contacts_are_allowed_and_restored():
    names = ["left_finger_link", "right_finger_link", "tool0", "upper_arm_link"]
    original = AllowedCollisionMatrix(entry_names=names,
        entry_values=[AllowedCollisionEntry(enabled=[False]*4) for _ in names])
    original.entry_values[0].enabled[1] = original.entry_values[1].enabled[0] = True
    guarded = contact_matrix(original, "owned_cup")
    assert not any(guarded.entry_values[-1].enabled)
    changed = contact_matrix(original, "owned_cup", fingers=True)
    assert changed.entry_values[-1].enabled == [True, True, False, False, False]
    assert original.entry_names == names
    assert contact_matrix(changed, "owned_cup", remove=True) == original


def test_preview_default_and_no_forced_confirmation_switch():
    options = parser().parse_args([])
    assert not options.execute and not options.detect_only
    with pytest.raises(SystemExit):
        parser().parse_args(["--execute", "--detect-only"])
    with pytest.raises(SystemExit):
        parser().parse_args(["--yes"])


def test_support_contact_does_not_disable_robot_table_collisions():
    names = ["left_finger_link", "right_finger_link", "tool0", "unrelated_table"]
    original = AllowedCollisionMatrix(entry_names=names,
        entry_values=[AllowedCollisionEntry(enabled=[False]*4) for _ in names])
    changed = contact_matrix(original, "cup", support="owned_table")
    assert changed.entry_values[-1].enabled == [False, False, False, False, True, False]
    support_index = changed.entry_names.index("owned_table")
    assert changed.entry_values[support_index].enabled == [False, False, False, False, False, True]
    restored = contact_matrix(contact_matrix(changed, "cup", remove=True), "owned_table", remove=True)
    assert restored == original


def test_moved_cup_and_failed_visual_lift_are_rejected():
    from types import SimpleNamespace
    observed = SimpleNamespace(center=np.array([.65, 0, .60]), radius=.025, top=.65,
                               table_bounds=np.array([[.5, -.2, .55], [.8, .2, .55]]))
    unchanged(observed, copy.deepcopy(observed))
    moved = copy.deepcopy(observed)
    moved.center[1] += .02
    with pytest.raises(RuntimeError):
        unchanged(observed, moved)
    with pytest.raises(RuntimeError):
        verify_lift(observed, observed, .08)
    lifted = copy.deepcopy(observed)
    lifted.top += .08
    assert verify_lift(observed, lifted, .08) == pytest.approx(.08)


def test_scene_refreshes_stale_fixture_and_limits_finger_contact_to_approach():
    from types import SimpleNamespace
    names = ["left_finger_link", "right_finger_link", "tool0"]
    current = PlanningScene()
    current.allowed_collision_matrix = AllowedCollisionMatrix(entry_names=names,
        entry_values=[AllowedCollisionEntry(enabled=[False]*3) for _ in names])
    current.world.collision_objects = [CollisionObject(id="grasp_fixture_table")]
    node = SimpleNamespace(model=SimpleNamespace(frame="base_footprint"))
    cup = SimpleNamespace(center=np.array([.65, .12, .60]), radius=.025, height=.1,
                          table_bounds=np.array([[.5, -.08, .55], [.8, .32, .55]]))
    scene = GraspScene(node, cup)
    scene.get = lambda: current
    applied = []
    scene.apply = applied.append
    scene.install()
    update = applied[-1]
    fixture = next(obj for obj in update.world.collision_objects if obj.id == "grasp_fixture_table")
    assert fixture.pose.position.y == pytest.approx(.12)
    assert fixture.primitives[0].dimensions[2] == pytest.approx(.55)
    current.allowed_collision_matrix = update.allowed_collision_matrix
    index = current.allowed_collision_matrix.entry_names.index(scene.identifier)
    assert not current.allowed_collision_matrix.entry_values[index].enabled[0]
    scene.finger_contacts(True)
    assert applied[-1].allowed_collision_matrix.entry_values[index].enabled[:3] == [True, True, False]
    scene.finger_contacts(False)
    assert applied[-1].allowed_collision_matrix.entry_values[index].enabled[:3] == [False, False, False]
