"""Real MoveIt planning/execution against a mock arm in an isolated DDS domain."""

import os
from pathlib import Path
import re
import signal
import subprocess
import threading
import time

import pytest


pytestmark = pytest.mark.skipif(os.environ.get("WHEELARM_ROS_TEST") != "1",
                                reason="Requires an unused ROS domain and installed package")
JOINTS = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_pitch_joint"]
READY = [0.0, -0.6, 0.9, -0.3]


@pytest.mark.parametrize("backend", ["mock", pytest.param("gazebo", marks=pytest.mark.skipif(
    os.environ.get("WHEELARM_GAZEBO_TEST") != "1", reason="Opt-in isolated Gazebo rendering test"))])
def test_real_planning_collision_ik_and_trajectory_execution(tmp_path, backend):
    if backend == "gazebo" and os.environ.get("ROS_DOMAIN_ID", "0") == "0":
        pytest.fail("Gazebo execution tests require an explicit non-default unused ROS_DOMAIN_ID")
    import rclpy
    from control_msgs.action import FollowJointTrajectory
    from controller_manager_msgs.srv import ListControllers
    from moveit_msgs.action import ExecuteTrajectory
    from moveit_msgs.msg import Constraints, JointConstraint, RobotState, MoveItErrorCodes
    from moveit_msgs.srv import GetMotionPlan, GetStateValidity, GetPositionFK, GetPositionIK
    from rclpy.action import ActionClient, ActionServer
    from rclpy.callback_groups import ReentrantCallbackGroup
    from rclpy.executors import MultiThreadedExecutor
    from sensor_msgs.msg import JointState

    rclpy.init()
    node = rclpy.create_node("mock_moveit_arm")
    positions = [0.0] * 4
    received = []
    actual_state_received = threading.Event()
    publisher = node.create_publisher(JointState, "/wheel_arm/joint_states", 10)

    def read_actual(msg):
        if all(name in msg.name for name in JOINTS):
            positions[:] = [msg.position[msg.name.index(name)] for name in JOINTS]
            actual_state_received.set()

    def publish():
        msg = JointState()
        msg.header.stamp = node.get_clock().now().to_msg()
        msg.name = JOINTS + ["left_wheel_joint", "right_wheel_joint"]
        msg.position = positions[:] + [0.0, 0.0]
        msg.velocity = [0.0] * 6
        publisher.publish(msg)

    def follow(handle):
        trajectory = handle.request.trajectory
        received.append(trajectory)
        assert set(trajectory.joint_names) == set(JOINTS)
        start, previous = time.monotonic(), [positions[JOINTS.index(j)] for j in trajectory.joint_names]
        previous_time = 0.0
        for point in trajectory.points:
            end_time = point.time_from_start.sec + point.time_from_start.nanosec / 1e9
            while time.monotonic() - start < end_time:
                ratio = min(1.0, max(0.0, (time.monotonic() - start - previous_time)
                                      / max(1e-9, end_time - previous_time)))
                for index, name in enumerate(trajectory.joint_names):
                    positions[JOINTS.index(name)] = previous[index] + ratio * (point.positions[index] - previous[index])
                time.sleep(0.01)
            for index, name in enumerate(trajectory.joint_names):
                positions[JOINTS.index(name)] = point.positions[index]
            previous, previous_time = list(point.positions), end_time
        handle.succeed()
        result = FollowJointTrajectory.Result()
        result.error_code = FollowJointTrajectory.Result.SUCCESSFUL
        return result

    group = ReentrantCallbackGroup()
    action = None
    if backend == "mock":
        action = ActionServer(node, FollowJointTrajectory,
                              "/wheel_arm/arm_controller/follow_joint_trajectory",
                              execute_callback=follow, callback_group=group)
        node.create_timer(0.02, publish, callback_group=group)
    else:
        node.create_subscription(JointState, "/wheel_arm/joint_states", read_actual, 10,
                                 callback_group=group)
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(node)
    thread = threading.Thread(target=executor.spin)
    thread.start()
    process = None
    logfile = (tmp_path / "move_group.log").open("w+")

    def wait(future, timeout=20):
        deadline = time.monotonic() + timeout
        while not future.done() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert future.done(), "MoveIt request timed out"
        return future.result()

    def call(kind, name, request):
        client = node.create_client(kind, name, callback_group=group)
        try:
            assert client.wait_for_service(timeout_sec=20), name
            return wait(client.call_async(request))
        finally:
            node.destroy_client(client)

    def state(values):
        result = RobotState()
        result.joint_state.name = JOINTS
        result.joint_state.position = list(values)
        return result

    try:
        command = ["ros2", "launch", "wheel_arm_moveit_config"]
        show_rviz = backend == "mock" and os.environ.get("WHEELARM_GUI_TEST") == "1"
        if backend == "mock":
            command += ["moveit.launch.py", f"rviz:={str(show_rviz).lower()}", "use_sim_time:=false"]
        else:
            command += ["sim.launch.py", "rviz:=false", "gui:=false"]
        environment = os.environ.copy()
        environment["IGN_PARTITION"] = environment["GZ_PARTITION"] = f"wheelarm-moveit-test-{os.getpid()}"
        process = subprocess.Popen(command, env=environment,
            stdout=logfile, stderr=subprocess.STDOUT, start_new_session=True)
        if backend == "gazebo":
            controller = node.create_client(ListControllers, "/wheel_arm/controller_manager/list_controllers")
            try:
                assert controller.wait_for_service(timeout_sec=60)
                deadline = time.monotonic() + 30
                while time.monotonic() < deadline:
                    controllers = wait(controller.call_async(ListControllers.Request())).controller
                    if any(c.name == "arm_controller" and c.state == "active" for c in controllers):
                        break
                    time.sleep(0.2)
                else:
                    pytest.fail("Gazebo arm_controller did not activate")
            finally:
                node.destroy_client(controller)
            assert actual_state_received.wait(5)
        for values in ([0.0] * 4, READY):
            request = GetStateValidity.Request(robot_state=state(values), group_name="arm")
            response = call(GetStateValidity, "/check_state_validity", request)
            assert response.valid, str(response.contacts)
        response = call(GetStateValidity, "/check_state_validity",
                        GetStateValidity.Request(robot_state=state([0.0, 1.5, 0.0, 0.0]), group_name="arm"))
        assert not response.valid, "Non-adjacent arm/base collisions must remain enabled"

        fk = GetPositionFK.Request(robot_state=state(READY), fk_link_names=["tool0"])
        fk.header.frame_id = "base_footprint"
        pose = call(GetPositionFK, "/compute_fk", fk)
        assert pose.error_code.val == MoveItErrorCodes.SUCCESS
        ik = GetPositionIK.Request()
        ik.ik_request.group_name = "arm"
        ik.ik_request.ik_link_name = "tool0"
        ik.ik_request.pose_stamped = pose.pose_stamped[0]
        # An incompatible orientation must not turn position-only IK into six-DOF IK.
        ik.ik_request.pose_stamped.pose.orientation.w = 0.0
        ik.ik_request.pose_stamped.pose.orientation.z = 1.0
        ik.ik_request.robot_state = state([0.0] * 4)
        ik.ik_request.avoid_collisions = True
        ik.ik_request.timeout.sec = 1
        result = call(GetPositionIK, "/compute_ik", ik)
        assert result.error_code.val == MoveItErrorCodes.SUCCESS
        checked = GetPositionFK.Request(robot_state=result.solution, fk_link_names=["tool0"])
        checked.header.frame_id = "base_footprint"
        solved = call(GetPositionFK, "/compute_fk", checked).pose_stamped[0].pose.position
        target = ik.ik_request.pose_stamped.pose.position
        assert sum((getattr(solved, a) - getattr(target, a)) ** 2 for a in ("x", "y", "z")) < 1e-6

        request = GetMotionPlan.Request()
        plan = request.motion_plan_request
        plan.group_name, plan.pipeline_id = "arm", "ompl"
        plan.planner_id = "RRTConnectkConfigDefault"
        # Gazebo feedback includes small gravity/servo offsets; plan from measured state.
        start_positions = positions[:]
        plan.start_state = state(start_positions)
        plan.allowed_planning_time, plan.num_planning_attempts = 5.0, 3
        plan.max_velocity_scaling_factor = plan.max_acceleration_scaling_factor = 0.5
        constraints = Constraints()
        constraints.joint_constraints = [JointConstraint(joint_name=name, position=value,
            tolerance_above=0.001, tolerance_below=0.001, weight=1.0) for name, value in zip(JOINTS, READY)]
        plan.goal_constraints = [constraints]
        response = call(GetMotionPlan, "/plan_kinematic_path", request).motion_plan_response
        assert response.error_code.val == MoveItErrorCodes.SUCCESS
        trajectory = response.trajectory.joint_trajectory
        assert set(trajectory.joint_names) == set(JOINTS)
        assert len(trajectory.points) >= 2
        times = [p.time_from_start.sec + p.time_from_start.nanosec / 1e9 for p in trajectory.points]
        assert all(a < b for a, b in zip(times, times[1:])) and times[-1] > 0
        assert all(abs(v) <= 0.6 + 1e-6 for p in trajectory.points for v in p.velocities)
        assert not received, "Planning alone must not execute"

        client = ActionClient(node, ExecuteTrajectory, "/execute_trajectory", callback_group=group)
        try:
            assert client.wait_for_server(timeout_sec=10)
            handle = wait(client.send_goal_async(ExecuteTrajectory.Goal(trajectory=response.trajectory)))
            assert handle.accepted
            result = wait(handle.get_result_async(), timeout=40)
            assert result.result.error_code.val == MoveItErrorCodes.SUCCESS
            if backend == "mock":
                assert len(received) == 1
            assert all(abs(value - target) < 0.03 for value, target in zip(positions, READY))
            plan.start_state = state(positions[:])
            for constraint in plan.goal_constraints[0].joint_constraints:
                constraint.position = 0.0
            response = call(GetMotionPlan, "/plan_kinematic_path", request).motion_plan_response
            assert response.error_code.val == MoveItErrorCodes.SUCCESS
            handle = wait(client.send_goal_async(ExecuteTrajectory.Goal(trajectory=response.trajectory)))
            assert handle.accepted
            result = wait(handle.get_result_async(), timeout=40)
            assert result.result.error_code.val == MoveItErrorCodes.SUCCESS
            assert all(abs(value) < 0.05 for value in positions)
            if backend == "mock":
                assert len(received) == 2
            if show_rviz:
                text = (tmp_path / "move_group.log").read_text()
                assert "Ready to take commands for planning group arm" in text
                assert "Error loading" not in text and "PluginlibFactory" not in text
        finally:
            client.destroy()
    finally:
        if process is not None and process.poll() is None:
            process.send_signal(signal.SIGINT)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)
        # Gazebo's Ruby launcher can exit before its native child; reap our whole group.
        if process is not None:
            for signum in (signal.SIGTERM, signal.SIGKILL):
                try:
                    os.killpg(process.pid, signum)
                    time.sleep(0.2)
                except ProcessLookupError:
                    break
        executor.shutdown(timeout_sec=5)
        thread.join(timeout=5)
        if action is not None:
            action.destroy()
        node.destroy_node()
        rclpy.shutdown()
        logfile.close()
    text = (tmp_path / "move_group.log").read_text()
    assert "Segmentation fault" not in text and "exit code -11" not in text, text
    assert re.search(r"\[move_group-\d+\]: process has finished cleanly", text), text
    if show_rviz:
        assert re.search(r"\[rviz2-\d+\]: process has finished cleanly", text), text
