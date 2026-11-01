"""Exercise the installed CLI against real MoveIt and an isolated mock robot."""

import os
from pathlib import Path
import pty
import select
import signal
import subprocess
import threading
import time

import pytest


pytestmark = pytest.mark.skipif(os.environ.get("WHEELARM_ROS_TEST") != "1",
                                reason="Requires an unused ROS domain and installed packages")
JOINTS = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_pitch_joint"]
CONTROL_JOINTS = JOINTS + ["left_finger_joint", "right_finger_joint"]


@pytest.fixture
def robot(tmp_path):
    import rclpy
    from control_msgs.action import FollowJointTrajectory
    from nav_msgs.msg import Odometry
    from rclpy.action import ActionServer, CancelResponse, GoalResponse
    from rclpy.callback_groups import ReentrantCallbackGroup
    from rclpy.executors import MultiThreadedExecutor
    from sensor_msgs.msg import JointState

    rclpy.init()
    node = rclpy.create_node("mock_position_arm")
    state = {"position": [0.0] * 4 + [0.03, 0.03], "mode": "normal", "goals": [], "active": False,
             "accepted": threading.Event(), "stop": threading.Event()}
    joints = node.create_publisher(JointState, "/wheel_arm/joint_states", 10)
    odom = node.create_publisher(Odometry, "/odometry/filtered", 10)

    def publish():
        stamp = node.get_clock().now().to_msg()
        message = JointState(name=CONTROL_JOINTS + ["left_wheel_joint", "right_wheel_joint"],
                             position=state["position"][:] + [0.0, 0.0], velocity=[0.0] * 8)
        message.header.stamp = stamp
        if state["mode"] == "stale":
            message.header.stamp.sec -= 5
        if state["mode"] == "unsafe" and state["active"]:
            message.velocity[0] = 0.1
        joints.publish(message)
        message = Odometry(child_frame_id="base_footprint")
        message.header.stamp = stamp
        message.twist.twist.linear.x = 0.2 if state["mode"] == "moving_base" else 0.0
        odom.publish(message)

    def accept(goal):
        state["goals"].append(goal.trajectory)
        return GoalResponse.REJECT if state["mode"] == "reject" else GoalResponse.ACCEPT

    def follow(handle):
        state["active"] = True
        state["accepted"].set()
        trajectory = handle.request.trajectory
        start, previous_time = time.monotonic(), 0.0
        previous = [state["position"][CONTROL_JOINTS.index(name)] for name in trajectory.joint_names]
        try:
            for point in trajectory.points:
                end = point.time_from_start.sec + point.time_from_start.nanosec / 1e9
                while time.monotonic() - start < end or state["mode"] == "unsafe":
                    if state["stop"].is_set():
                        handle.abort()
                        return FollowJointTrajectory.Result(error_code=-1)
                    if handle.is_cancel_requested:
                        handle.canceled()
                        return FollowJointTrajectory.Result()
                    if state["mode"] != "no_motion":
                        ratio = min(1.0, max(0.0, (time.monotonic() - start - previous_time) / max(1e-9, end - previous_time)))
                        for index, name in enumerate(trajectory.joint_names):
                            state["position"][CONTROL_JOINTS.index(name)] = previous[index] + ratio * (point.positions[index] - previous[index])
                    time.sleep(0.01)
                if state["mode"] != "no_motion":
                    for index, name in enumerate(trajectory.joint_names):
                        state["position"][CONTROL_JOINTS.index(name)] = point.positions[index]
                previous, previous_time = list(point.positions), end
            handle.succeed()
            return FollowJointTrajectory.Result(error_code=FollowJointTrajectory.Result.SUCCESSFUL)
        finally:
            state["active"] = False

    group = ReentrantCallbackGroup()
    action = ActionServer(node, FollowJointTrajectory,
        "/wheel_arm/arm_controller/follow_joint_trajectory", execute_callback=follow,
        goal_callback=accept, cancel_callback=lambda _: (
            CancelResponse.REJECT if state["mode"] == "unsafe" else CancelResponse.ACCEPT), callback_group=group)
    gripper_action = ActionServer(node, FollowJointTrajectory,
        "/wheel_arm/gripper_controller/follow_joint_trajectory", execute_callback=follow,
        goal_callback=accept, cancel_callback=lambda _: CancelResponse.ACCEPT, callback_group=group)
    node.create_timer(0.02, publish, callback_group=group)
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    thread = threading.Thread(target=executor.spin)
    thread.start()
    log_path = tmp_path / "moveit.log"
    log = log_path.open("w+")
    process = subprocess.Popen(["ros2", "launch", "wheel_arm_moveit_config", "moveit.launch.py",
        "rviz:=false", "use_sim_time:=false"], stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        yield state
    finally:
        state["stop"].set()
        deadline = time.monotonic() + 3
        while state["active"] and time.monotonic() < deadline:
            time.sleep(0.02)
        if process.poll() is None:
            process.send_signal(signal.SIGINT)
            try:
                process.wait(timeout=8)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        executor.shutdown(timeout_sec=5)
        thread.join(timeout=5)
        action.destroy()
        gripper_action.destroy()
        node.destroy_node()
        rclpy.shutdown()
        log.close()
        assert "exit code -11" not in log_path.read_text()


def invoke(arguments, robot, mode, tmp_path, sim_time=False, wait=2):
    command = ["ros2", "run", "wheel_arm_manipulation", "arm_goal", *arguments,
               "--wait", str(wait), "--ros-args", "-p", f"use_sim_time:={str(sim_time).lower()}"]
    if "--execute" not in arguments or mode == "pipe":
        return subprocess.run(command, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=25)
    master, slave = pty.openpty()
    output = bytearray()
    process = subprocess.Popen(command, stdin=slave, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, start_new_session=True)
    os.close(slave)
    try:
        deadline = time.monotonic() + 20
        while "[y/N]".encode() not in output:
            assert time.monotonic() < deadline, output.decode(errors="replace")
            ready, _, _ = select.select([process.stdout], [], [], 0.1)
            if ready:
                data = os.read(process.stdout.fileno(), 65536)
                assert data, output.decode(errors="replace")
                output.extend(data)
        if mode == "changed":
            robot["position"][0] = 0.2
        os.write(master, b"n\n" if mode == "decline" else b"y\n")
        if mode == "cancel":
            assert robot["accepted"].wait(15)
            os.killpg(process.pid, signal.SIGINT)
        data, _ = process.communicate(timeout=35)
        output.extend(data)
        return subprocess.CompletedProcess(command, process.returncode, output.decode(errors="replace"), "")
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGINT)
            try:
                process.wait(timeout=12)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)
        os.close(master)


@pytest.mark.parametrize("mode", ["state", "preview", "unreachable", "execute", "position_execute",
    "decline", "pipe", "changed", "moving_base", "stale", "cancel", "timeout", "reject", "no_motion", "unsafe"])
def test_installed_cli(robot, tmp_path, mode):
    robot["mode"] = mode if mode in ("moving_base", "stale", "reject", "no_motion", "unsafe") else "normal"
    arguments = ["named", "ready", "--velocity", "0.5", "--acceleration", "0.5"]
    if mode == "state":
        arguments = ["state"]
    elif mode in ("preview", "position_execute"):
        arguments = ["position", "--x", "0.55", "--y", "0", "--z", "0.65",
                     "--velocity", "0.5", "--acceleration", "0.5"]
    elif mode == "unreachable":
        arguments = ["position", "--x", "5", "--y", "0", "--z", "1"]
    if mode in ("execute", "position_execute", "decline", "pipe", "changed", "cancel", "timeout", "reject", "no_motion", "unsafe"):
        arguments += ["--execute"]
    if mode in ("timeout", "unsafe"):
        arguments += ["--timeout", "0.2"]
    result = invoke(arguments, robot, mode, tmp_path)
    text = result.stdout + result.stderr
    expected = 0 if mode in ("state", "preview", "execute", "position_execute", "decline") else 130 if mode == "cancel" else 1
    assert result.returncode == expected, text
    if mode in ("execute", "position_execute"):
        assert len(robot["goals"]) == 1 and "[完成]" in text
    elif mode in ("cancel", "timeout"):
        assert "[取消] 已确认目标结束" in text and "[完成]" not in text
    elif mode == "unsafe":
        assert "[安全]" in text and "[完成]" not in text
    elif mode in ("reject", "no_motion"):
        assert len(robot["goals"]) == 1 and "[完成]" not in text
    else:
        assert not robot["goals"], text


@pytest.mark.parametrize("mode", ["preview", "execute", "decline", "no_motion", "cancel"])
def test_gripper_cli(robot, tmp_path, mode):
    robot["mode"] = "no_motion" if mode == "no_motion" else "normal"
    arguments = ["gripper", "closed", "--velocity", "0.5", "--acceleration", "0.5"]
    if mode != "preview":
        arguments += ["--execute"]
    result = invoke(arguments, robot, mode, tmp_path)
    text = result.stdout + result.stderr
    expected = 1 if mode == "no_motion" else 130 if mode == "cancel" else 0
    assert result.returncode == expected, text
    if mode == "execute":
        assert "[夹爪] 反馈内侧开口" in text and "[完成]" in text
        assert all(abs(p) <= 0.002 for p in robot["position"][4:])
    else:
        assert "[完成]" not in text
    if mode in ("preview", "decline"):
        assert not robot["goals"]
    else:
        assert len(robot["goals"]) == 1
        assert set(robot["goals"][0].joint_names) == set(CONTROL_JOINTS[4:])
    if mode == "cancel":
        assert "[取消] 已确认目标结束" in text


@pytest.mark.skipif(os.environ.get("WHEELARM_GAZEBO_TEST") != "1", reason="Opt-in actual Gazebo test")
@pytest.mark.parametrize("fixture", ["empty", "cup", "cup_mast", "mug_mast", "ceramic_cup_mast"])
def test_xyz_and_home_with_gazebo(tmp_path, fixture):
    if os.environ.get("ROS_DOMAIN_ID", "0") == "0":
        pytest.fail("Use an explicit unused non-default ROS domain")
    import rclpy
    from controller_manager_msgs.srv import ListControllers
    from geometry_msgs.msg import PoseStamped
    from moveit_msgs.msg import PlanningSceneComponents
    from moveit_msgs.srv import GetPlanningScene
    from sensor_msgs.msg import CameraInfo, Image
    from tf2_ros import Buffer, TransformListener

    rclpy.init()
    node = rclpy.create_node("arm_goal_gazebo_check")
    cup_poses = []
    images = {}
    camera_tf = Buffer()
    camera_listener = TransformListener(camera_tf, node)
    node.create_subscription(CameraInfo, "/camera/camera_info", lambda msg: images.update(info=msg), 10)
    for topic, key in (("/camera/image", "rgb"), ("/camera/depth_image", "depth")):
        node.create_subscription(Image, topic, lambda msg, key=key: images.update({key: msg}), 10)
    node.create_subscription(PoseStamped, "/grasp_test/cup_pose",
                             lambda msg: cup_poses.append((time.monotonic(), msg.pose.position)), 100)

    def observe(seconds):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            rclpy.spin_once(node, timeout_sec=0.02)
        return [p for t, p in cup_poses if t > end - min(1, seconds / 2)]

    def command(arguments):
        result = invoke([*arguments, "--velocity", "0.3", "--acceleration", "0.3", "--execute"],
                        None, "execute", tmp_path, sim_time=True, wait=10)
        with (tmp_path / "commands.log").open("a") as transcript:
            transcript.write(str(arguments) + "\n" + result.stdout + result.stderr + "\n")
        assert result.returncode == 0 and "[完成]" in result.stdout, result.stdout + result.stderr
        return result

    def capture(label):
        import cv2
        import numpy as np
        from cv_bridge import CvBridge
        observe(1)
        assert {"rgb", "depth"} <= images.keys()
        bridge = CvBridge()
        rgb = bridge.imgmsg_to_cv2(images["rgb"], desired_encoding="rgb8")
        depth = bridge.imgmsg_to_cv2(images["depth"], desired_encoding="passthrough")
        assert rgb.shape == (480, 640, 3) and depth.shape == (480, 640)
        assert images["rgb"].header.frame_id == images["depth"].header.frame_id == "camera_optical_frame"
        assert cv2.imwrite(str(tmp_path / (label + ".png")), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        np.save(tmp_path / (label + "_depth.npy"), depth)
        if fixture == "cup_mast":
            # Fixture-only visibility check, not a production cup detector.
            r, g, b = rgb.astype(float).transpose(2, 0, 1)
            cup = (g > r * 1.5) & (b > r * 1.5) & (g > 40) & (abs(g - b) < 25)
            assert cup.sum() >= 20, f"Cup not visible at {label}: {cup.sum()} pixels"
            assert np.count_nonzero(cup & np.isfinite(depth) & (depth > 0.1) & (depth < 2)) >= 20
        elif fixture in ("mug_mast", "ceramic_cup_mast"):
            from geometry_msgs.msg import PointStamped
            from rclpy.time import Time
            from tf2_geometry_msgs import do_transform_point

            assert "info" in images and cup_poses
            transform = camera_tf.lookup_transform("camera_optical_frame", "base_footprint",
                                                    Time.from_msg(images["rgb"].header.stamp))
            # This fixture keeps the base at the world origin; truth is used only by the test.
            center = cup_poses[-1][1]
            corners = []
            for x in (-.03, .03):
                for y in (-.03, .03):
                    for z in (-.055, .055):
                        point = PointStamped()
                        point.point.x, point.point.y, point.point.z = center.x+x, center.y+y, center.z+z
                        corners.append(do_transform_point(point, transform).point)
            k = images["info"].k
            assert min(p.z for p in corners) > 0
            u = [k[0] * p.x / p.z + k[2] for p in corners]
            v = [k[4] * p.y / p.z + k[5] for p in corners]
            x0, x1 = max(0, int(min(u))-3), min(rgb.shape[1], int(max(u))+4)
            y0, y1 = max(0, int(min(v))-3), min(rgb.shape[0], int(max(v))+4)
            roi = rgb[y0:y1, x0:x1].astype(float)
            distance = depth[y0:y1, x0:x1]
            neutral = (roi.max(axis=2)-roi.min(axis=2) < 25) & (roi.mean(axis=2) > 80)
            visible = neutral & np.isfinite(distance) & (distance >= min(p.z for p in corners)) & (
                distance <= max(p.z for p in corners))
            assert visible.sum() >= 20, f"Neutral cup not visible at {label}: {visible.sum()} pixels"
    log_path = tmp_path / "gazebo.log"
    log = log_path.open("w+")
    environment = os.environ.copy()
    environment["IGN_PARTITION"] = environment["GZ_PARTITION"] = f"arm-goal-test-{os.getpid()}"
    package, launch = (("wheel_arm_manipulation", "cup_test.launch.py") if fixture != "empty" else
                       ("wheel_arm_moveit_config", "sim.launch.py"))
    cup_model = {"mug_mast": "ceramic_mug", "ceramic_cup_mast": "ceramic_cup"}.get(fixture)
    process = subprocess.Popen(["ros2", "launch", package, launch,
        "gui:=false", "rviz:=false", "camera_mount:=" + ("mast" if fixture.endswith("_mast") else "low"),
        *(["cup_model:=" + cup_model] if cup_model else [])],
        stdout=log, stderr=subprocess.STDOUT, env=environment, start_new_session=True)
    try:
        client = node.create_client(ListControllers, "/wheel_arm/controller_manager/list_controllers")
        assert client.wait_for_service(timeout_sec=60)
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            future = client.call_async(ListControllers.Request())
            rclpy.spin_until_future_complete(node, future, timeout_sec=5)
            if not future.done():
                future.cancel()
                continue
            active = {c.name for c in future.result().controller if c.state == "active"}
            if {"arm_controller", "gripper_controller"} <= active:
                break
            time.sleep(0.1)
        else:
            pytest.fail("arm/gripper controllers not active")
        if fixture == "empty":
            for arguments in (["gripper", "open"], ["gripper", "closed"], ["gripper", "open"],
                              ["position", "--x", "0.55", "--y", "0", "--z", "0.65"], ["named", "home"]):
                result = command(arguments)
                if arguments[0] == "gripper":
                    assert "[夹爪] 反馈内侧开口" in result.stdout
        else:
            scene_client = node.create_client(GetPlanningScene, "/get_planning_scene")
            assert scene_client.wait_for_service(timeout_sec=15)
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline:
                query = GetPlanningScene.Request(components=PlanningSceneComponents(
                    components=PlanningSceneComponents.WORLD_OBJECT_NAMES))
                future = scene_client.call_async(query)
                rclpy.spin_until_future_complete(node, future, timeout_sec=3)
                if future.done() and {"grasp_fixture_table", "grasp_fixture_floor"} <= {
                        obj.id for obj in future.result().scene.world.collision_objects}:
                    break
            else:
                pytest.fail("Fixture collision scene was not installed")
            poses = observe(2)
            assert poses, subprocess.run(["ign", "topic", "-l"], env=environment,
                                         capture_output=True, text=True, timeout=5).stdout
            initial = poses[-1]
            assert abs(initial.z - 0.5987) < 0.005
            capture("pregrasp")
            command(["gripper", "open"])
            command(["named", "cup_grasp"])
            approach = observe(1)
            assert approach and abs(approach[-1].z - initial.z) < 0.005
            assert abs(approach[-1].x - initial.x) < 0.01 and abs(approach[-1].y - initial.y) < 0.01
            capture("grasp")
            command(["gripper", "width", "--width", "0.049"])
            command(["named", "cup_lift"])
            lifted = observe(3)
            assert len(lifted) >= 5
            assert all(p.z - initial.z > 0.05 for p in lifted), str(lifted)
            assert max(p.z for p in lifted) - min(p.z for p in lifted) < 0.01
            assert all(abs(p.x - initial.x) < 0.04 and abs(p.y - initial.y) < 0.02 for p in lifted)
            capture("lift")
            command(["named", "cup_grasp"])
            command(["gripper", "open"])
            released = observe(3)
            assert released and abs(released[-1].z - initial.z) < 0.005
            assert abs(released[-1].x - initial.x) < 0.02 and abs(released[-1].y - initial.y) < 0.02
    finally:
        if process.poll() is None:
            process.send_signal(signal.SIGINT)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)
        for signum in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(process.pid, signum)
                time.sleep(0.2)
            except ProcessLookupError:
                break
        node.destroy_node()
        rclpy.shutdown()
        log.close()
        assert "exit code -11" not in log_path.read_text()
