import copy
from types import SimpleNamespace as NS

from moveit_msgs.msg import RobotTrajectory
import pytest
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectoryPoint

from wheel_arm_manipulation import arm_goal
from wheel_arm_manipulation.planning import load_model, validate_trajectory, values_from_state


@pytest.fixture
def model():
    return load_model()


def trajectory(model):
    result = RobotTrajectory()
    result.joint_trajectory.joint_names = list(model.joints)
    for t, values in ((0, model.named["home"]), (10, model.named["ready"])):
        point = JointTrajectoryPoint(positions=[values[n] for n in model.joints],
                                     velocities=[0.0] * 4, accelerations=[0.0] * 4)
        point.time_from_start.sec = t
        result.joint_trajectory.points.append(point)
    return result


def test_model_reuses_srdf_and_urdf(model):
    assert len(model.joints) == 4 and model.tip == "tool0"
    assert model.named["ready"]["elbow_joint"] == 0.9
    assert model.bounds["elbow_joint"] == (-2.2, 2.2)
    assert model.velocities["elbow_joint"] == 0.6
    assert model.accelerations["elbow_joint"] == 0.8


def test_gripper_model_uses_linear_limits_and_named_openings():
    model = load_model("gripper")
    assert model.joints == ("left_finger_joint", "right_finger_joint")
    assert model.tip == "gripper_tcp" and model.group == "gripper"
    assert model.bounds == {name: (0, 0.04) for name in model.joints}
    assert sum(model.named["open"].values()) == 0.08
    assert sum(model.named["closed"].values()) == 0
    assert model.start_tolerance == 0.002 and model.reached_tolerance == 0.002
    assert arm_goal.parser().parse_args(["gripper", "open"]).execute is False


@pytest.mark.parametrize("command", ["close", "half", "nan"])
def test_gripper_rejects_unknown_state(command):
    with pytest.raises(SystemExit):
        arm_goal.parser().parse_args(["gripper", command])


@pytest.mark.parametrize("width", ["-0.001", "0.081", "nan", "inf"])
def test_gripper_width_limits(width):
    with pytest.raises(SystemExit):
        arm_goal.parser().parse_args(["gripper", "width", "--width", width])


@pytest.mark.parametrize("arguments", [["gripper", "width"], ["gripper", "open", "--width", "0.04"]])
def test_gripper_width_mode_requires_matching_arguments(arguments, monkeypatch):
    monkeypatch.setattr(arm_goal.rclpy, "init", lambda **_: pytest.fail("Must reject before ROS initialization"))
    with pytest.raises(SystemExit):
        arm_goal.main(arguments)


def test_fixture_scene_matches_known_test_world():
    from wheel_arm_manipulation.grasp_fixture import fixture_scene
    scene = fixture_scene()
    assert scene.is_diff and scene.robot_state.is_diff
    table, floor = scene.world.collision_objects
    assert table.header.frame_id == "base_footprint"
    assert table.pose.position.x == 0.7
    assert abs(table.pose.position.z + table.primitives[0].dimensions[2] / 2 - 0.5487) < 1e-8
    assert floor.pose.position.z < 0


def test_gripper_trajectory_rejects_millimeter_endpoint_error():
    from trajectory_msgs.msg import JointTrajectoryPoint
    model = load_model("gripper")
    result = RobotTrajectory()
    result.joint_trajectory.joint_names = list(model.joints)
    for t, positions in ((0, [0.0, 0.0]), (10, [0.04, 0.04])):
        p = JointTrajectoryPoint(positions=positions, velocities=[0.0] * 2, accelerations=[0.0] * 2)
        p.time_from_start.sec = t
        result.joint_trajectory.points.append(p)
    assert validate_trajectory(result, model, model.named["closed"], model.named["open"], 0.2, 0.2) == 10
    result.joint_trajectory.points[-1].positions[0] -= 0.003
    with pytest.raises(ValueError):
        validate_trajectory(result, model, model.named["closed"], model.named["open"], 0.2, 0.2)


@pytest.mark.parametrize("mode", ["valid", "missing", "duplicate", "mismatch", "nan", "limit"])
def test_joint_state_validation(model, mode):
    msg = JointState(name=list(model.joints), position=[0.0] * 4)
    if mode == "missing":
        msg.name.pop()
        msg.position.pop()
    elif mode == "duplicate":
        msg.name[0] = msg.name[1]
    elif mode == "mismatch":
        msg.position.pop()
    elif mode == "nan":
        msg.position[0] = float("nan")
    elif mode == "limit":
        msg.position[0] = 10.0
    if mode == "valid":
        assert values_from_state(msg, model) == model.named["home"]
    else:
        with pytest.raises(ValueError):
            values_from_state(msg, model)


@pytest.mark.parametrize("mode", ["valid", "base", "duplicate", "missing", "one_point", "too_many",
    "position_nan", "position_limit", "velocity_nan", "velocity_limit", "acceleration_limit",
    "effort", "duration", "position_jump", "negative_time", "bad_nanoseconds", "delayed_header", "start", "end", "missing_velocities"])
def test_trajectory_validation(model, mode):
    result = trajectory(model)
    path = result.joint_trajectory
    last = path.points[-1]
    if mode == "base":
        result.multi_dof_joint_trajectory.joint_names = ["base"]
    elif mode == "duplicate":
        path.joint_names[0] = path.joint_names[1]
    elif mode == "missing":
        path.joint_names.pop()
    elif mode == "one_point":
        path.points.pop()
    elif mode == "too_many":
        path.points = [copy.deepcopy(last) for _ in range(1001)]
    elif mode == "position_nan":
        last.positions[0] = float("nan")
    elif mode == "position_limit":
        last.positions[0] = 10.0
    elif mode == "velocity_nan":
        last.velocities[0] = float("nan")
    elif mode == "velocity_limit":
        last.velocities[0] = 0.2
    elif mode == "acceleration_limit":
        last.accelerations[0] = 0.3
    elif mode == "effort":
        last.effort = [1.0] * 4
    elif mode == "duration":
        last.time_from_start.sec = 0
    elif mode == "position_jump":
        last.time_from_start.sec = 1
    elif mode == "negative_time":
        path.points[0].time_from_start.sec = -1
    elif mode == "bad_nanoseconds":
        last.time_from_start.nanosec = 10**9
    elif mode == "delayed_header":
        path.header.stamp.sec = 10
    elif mode == "start":
        path.points[0].positions[0] = 0.1
    elif mode == "end":
        last.positions[0] = 0.1
    elif mode == "missing_velocities":
        last.velocities = []
    if mode == "valid":
        assert validate_trajectory(result, model, model.named["home"], model.named["ready"], 0.2, 0.2) == 10
    else:
        with pytest.raises(ValueError):
            validate_trajectory(result, model, model.named["home"], model.named["ready"], 0.2, 0.2)


@pytest.mark.parametrize("arguments", [["position", "--x", "nan", "--y", "0", "--z", "1"],
    ["named", "ready", "--velocity", "0"], ["named", "ready", "--acceleration", "1.1"],
    ["named", "home", "--timeout", "inf"], ["state", "--execute"]])
def test_parser_rejects_unsafe_arguments(arguments):
    with pytest.raises(SystemExit):
        arm_goal.parser().parse_args(arguments)


@pytest.mark.parametrize("mode", ["preview", "decline", "execute", "pipe"])
def test_execution_confirmation_gate(monkeypatch, mode):
    calls = []
    node = NS(snapshot=lambda: None, plan=lambda _: (1, 2, 3), execute=lambda *args: calls.append(args))
    options = arm_goal.parser().parse_args(["named", "ready"] + ([] if mode == "preview" else ["--execute"]))
    monkeypatch.setattr(arm_goal.sys, "stdin", NS(isatty=lambda: mode != "pipe"))
    monkeypatch.setattr("builtins.input", lambda *_: "y" if mode == "execute" else "n")
    monkeypatch.setattr(arm_goal.rclpy, "spin_once", lambda *args, **kwargs: None)
    if mode == "pipe":
        with pytest.raises(ValueError):
            arm_goal.run(node, options)
    else:
        assert arm_goal.run(node, options) == 0
    assert calls == ([(1, 2, 3)] if mode == "execute" else [])


@pytest.mark.parametrize("mode", ["canceled", "aborted", "succeeded", "active", "no_result", "not_stopped", "unknown_acceptance"])
def test_cancellation_requires_terminal_result_and_stopped_feedback(mode, capsys):
    from action_msgs.msg import GoalStatus
    from unique_identifier_msgs.msg import UUID
    identity = UUID(uuid=list(range(16)))
    status = {"canceled": GoalStatus.STATUS_CANCELED, "aborted": GoalStatus.STATUS_ABORTED,
              "succeeded": GoalStatus.STATUS_SUCCEEDED}.get(mode, GoalStatus.STATUS_EXECUTING)
    if mode in ("not_stopped", "unknown_acceptance"):
        status = GoalStatus.STATUS_CANCELED
    result = NS(status=status)
    future = NS(done=lambda: mode != "no_result", result=lambda: result)
    checked = []

    def snapshot(**kwargs):
        checked.append(True)
        if mode == "not_stopped":
            raise RuntimeError("feedback still moving")

    def call(kind, name, request, *args):
        if name.endswith("cancel_goal"):
            assert request.goal_info.goal_id == identity
            return NS()
        assert request.goal_id == identity
        return result

    node = NS(execution_pending=True, wait=lambda *args, **kwargs: mode != "no_result",
              snapshot=snapshot, call=call, get_logger=lambda: NS(error=lambda *_: None))
    handle = NS(cancel_goal_async=lambda: NS())
    if mode == "unknown_acceptance":
        handle = future = None
    safe = arm_goal.ArmGoalClient.cancel_and_confirm(node, None, handle, future, identity)
    expected = mode in ("canceled", "aborted", "succeeded", "unknown_acceptance")
    assert safe == expected
    assert node.execution_pending != expected
    assert ("[安全]" in capsys.readouterr().out) != expected
