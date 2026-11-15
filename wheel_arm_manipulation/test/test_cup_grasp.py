import copy
from dataclasses import replace

from moveit_msgs.msg import AllowedCollisionEntry, AllowedCollisionMatrix, CollisionObject, PlanningScene
import numpy as np
import pytest
from types import SimpleNamespace
from unittest.mock import Mock

from wheel_arm_manipulation.cup_grasp import parser, straight_trajectory, unchanged, verify_lift
from wheel_arm_manipulation.grasp_kinematics import GraspKinematics
from wheel_arm_manipulation.grasp_scene import GraspScene, contact_matrix
from wheel_arm_manipulation.planning import load_model


@pytest.mark.parametrize("hold_ok", [True, False])
def test_extended_hold_stays_inside_lift_phase_and_rechecks_height(monkeypatch, hold_ok):
    from wheel_arm_manipulation import cup_grasp
    events = []
    monkeypatch.setattr(cup_grasp, "task_event", lambda node, kind, **data: events.append((kind, data)))
    monkeypatch.setattr(cup_grasp.rclpy, "spin_once", lambda *a, **kw: None)
    ticks = iter(range(100))
    monkeypatch.setattr(cup_grasp.time, "monotonic", lambda: next(ticks))
    stamps = iter([1, 2, 6, 12])
    cup = SimpleNamespace(center=np.array([.65, 0, .60]), radius=.025, top=.65)
    raised = copy.deepcopy(cup)
    raised.top += .08
    held = copy.deepcopy(raised if hold_ok else cup)
    joint = SimpleNamespace(name=[], position=[])
    arm = SimpleNamespace(joints=[], start_tolerance=.01, check=lambda _: None)
    node = Mock(arm=arm, gripper=arm, joints=joint, stopping=False,
                options=SimpleNamespace(finish="hold", lift_height=.08, hold_seconds=10.))
    node.snapshot.return_value = joint
    node.get_clock.return_value.now.side_effect = lambda: SimpleNamespace(nanoseconds=next(stamps)*10**9)
    node.observer.capture.side_effect = [raised, held]
    plans = [("lift", arm, object(), {}, {}, joint, None)]
    if hold_ok:
        cup_grasp.CupGraspClient.perform(node, plans, cup, Mock())
        assert [kind for kind, _ in events] == [
            "phase_started", "lift_verified", "hold_started", "hold_verified", "phase_finished"]
    else:
        with pytest.raises(RuntimeError, match="杯口升高验证"):
            cup_grasp.CupGraspClient.perform(node, plans, cup, Mock())
        assert events[-1][0] == "hold_started"


@pytest.mark.parametrize("lift_ok", [True, False])
def test_execution_emits_phase_boundaries_and_only_verified_evidence(monkeypatch, lift_ok):
    from wheel_arm_manipulation import cup_grasp
    events = []
    monkeypatch.setattr(cup_grasp, "task_event", lambda node, kind, **data: events.append((kind, data)))
    ticks = iter(range(100))
    monkeypatch.setattr(cup_grasp.time, "monotonic", lambda: next(ticks))
    cup = SimpleNamespace(center=np.array([.65, 0, .60]), radius=.025, top=.65,
                         table_bounds=np.array([[.5, -.2, .55], [.8, .2, .55]]))
    lifted = copy.deepcopy(cup)
    lifted.top += .08 if lift_ok else 0.
    joint = SimpleNamespace(name=[], position=[])
    arm = SimpleNamespace(joints=[], start_tolerance=.01, check=lambda _: None)
    gripper = copy.copy(arm)
    node = Mock(arm=arm, gripper=gripper, joints=joint,
                options=SimpleNamespace(finish="place", lift_height=.08))
    node.snapshot.return_value = joint
    node.observer.capture.side_effect = lambda **kw: lifted if kw.get("label") == "lift_verification" else cup
    labels = ["open", "pregrasp", "approach", "close", "lift", "lower", "release", "retreat"]
    plans = [(label, gripper if label in ("open", "close", "release") else arm,
              None if label == "open" else object(), {}, {}, joint, None) for label in labels]
    if lift_ok:
        cup_grasp.CupGraspClient.perform(node, plans, cup, Mock())
        assert [data["phase"] for kind, data in events if kind == "phase_finished"] == labels
        assert any(kind == "lift_verified" for kind, _ in events)
        assert any(kind == "placement_verified" for kind, _ in events)
        assert events[0] == ("phase_started", {"phase": "open", "motion_skipped": True})
    else:
        with pytest.raises(RuntimeError, match="杯口升高验证"):
            cup_grasp.CupGraspClient.perform(node, plans, cup, Mock())
        assert events[-1] == ("phase_started", {"phase": "lift", "motion_skipped": False})
        assert not any(kind == "lift_verified" for kind, _ in events)


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
