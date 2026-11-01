"""Opt-in real RGB-D/MoveIt/Gazebo grasp; truth is only an independent test oracle."""

import json
import os
from pathlib import Path
import pty
import select
import signal
import subprocess
import time

import numpy as np
import pytest


pytestmark = pytest.mark.skipif(os.environ.get("WHEELARM_GAZEBO_TEST") != "1",
                                reason="Opt-in RGB-D grasp against Gazebo")


@pytest.mark.parametrize("placement", ["default", "shifted", "relocated_base"])
def test_rgbd_grasp_without_simulator_pose_input(tmp_path, placement):
    if os.environ.get("ROS_DOMAIN_ID", "0") == "0":
        pytest.fail("Use an explicit unused ROS domain")
    import rclpy
    from controller_manager_msgs.srv import ListControllers
    from geometry_msgs.msg import PoseStamped
    from sensor_msgs.msg import JointState

    rclpy.init()
    node = rclpy.create_node("rgbd_grasp_independent_check")
    poses, joints = [], []
    node.create_subscription(PoseStamped, "/grasp_test/cup_pose", lambda msg: poses.append(msg.pose.position), 100)
    node.create_subscription(JointState, "/wheel_arm/joint_states", joints.append, 10)
    environment = dict(os.environ, IGN_PARTITION=f"rgbd-grasp-test-{os.getpid()}-{placement}",
                       GZ_PARTITION=f"rgbd-grasp-test-{os.getpid()}-{placement}")
    log_path = tmp_path / "simulation.log"
    log = log_path.open("w")
    simulation = subprocess.Popen(["ros2", "launch", "wheel_arm_manipulation", "cup_test.launch.py",
        "camera_mount:=mast", "cup_model:=ceramic_cup", "gui:="+os.environ.get("WHEELARM_GAZEBO_GUI", "false"),
        "rviz:=false"],
        env=environment, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)

    def observe(seconds):
        until = time.monotonic()+seconds
        while time.monotonic() < until:
            rclpy.spin_once(node, timeout_sec=.02)

    def cli(arguments, answer=None):
        master, slave = pty.openpty() if answer is not None else (None, None)
        process = subprocess.Popen(["ros2", "run", "wheel_arm_manipulation", "cup_grasp",
                                    *arguments, "--velocity", "0.3", "--acceleration", "0.3", "--wait", "15",
                                    "--output-dir", str(tmp_path / "observations")],
                                   env=environment, stdin=slave if slave is not None else subprocess.DEVNULL,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True)
        if slave is not None:
            os.close(slave)
        output = bytearray()
        answered = False
        try:
            deadline = time.monotonic()+150
            while process.poll() is None and time.monotonic() < deadline:
                rclpy.spin_once(node, timeout_sec=.01)
                if select.select([process.stdout], [], [], 0)[0]:
                    output.extend(os.read(process.stdout.fileno(), 65536))
                if master is not None and not answered and b"[y/N]" in output:
                    os.write(master, answer.encode()+b"\n")
                    answered = True
            assert process.poll() is not None, output.decode(errors="replace")
            output.extend(process.stdout.read())
            text = output.decode(errors="replace")
            (tmp_path / ("cli-"+str(len(list(tmp_path.glob("cli-*"))))+".log")).write_text(text)
            assert process.returncode == 0, text
            if master is not None:
                assert answered, text
            return text
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGINT)
                try:
                    process.wait(timeout=12)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            if master is not None:
                os.close(master)

    try:
        client = node.create_client(ListControllers, "/wheel_arm/controller_manager/list_controllers")
        assert client.wait_for_service(timeout_sec=60), log_path.read_text()
        deadline = time.monotonic()+25
        while time.monotonic() < deadline:
            future = client.call_async(ListControllers.Request())
            rclpy.spin_until_future_complete(node, future, timeout_sec=3)
            if future.done() and {"arm_controller", "gripper_controller"} <= {
                    c.name for c in future.result().controller if c.state == "active"}:
                break
        else:
            pytest.fail("Controllers not active")
        observe(2)
        assert poses and joints
        if placement == "shifted":
            result = subprocess.run(["ign", "service", "-s", "/world/grasp_test/set_pose",
                "--reqtype", "ignition.msgs.Pose", "--reptype", "ignition.msgs.Boolean", "--timeout", "5000",
                "--req", 'name: "test_cup", position: {x: 0.70, y: 0.035, z: 0.5988}, orientation: {w: 1}'],
                env=environment, capture_output=True, text=True, timeout=10)
            assert result.returncode == 0 and "true" in result.stdout, result.stdout+result.stderr
            observe(2)
        if placement == "relocated_base":
            result = subprocess.run(["ign", "service", "-s", "/world/grasp_test/set_pose",
                "--reqtype", "ignition.msgs.Pose", "--reptype", "ignition.msgs.Boolean", "--timeout", "5000",
                "--req", 'name: "wheel_arm", position: {x: -0.01, y: -0.06, z: 0.003}, '
                         'orientation: {z: -0.0499791693, w: 0.9987502604}'],
                env=environment, capture_output=True, text=True, timeout=10)
            assert result.returncode == 0 and "true" in result.stdout, result.stdout+result.stderr
            observe(3)
        initial = poses[-1]
        joint_names = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_pitch_joint",
                       "left_finger_joint", "right_finger_joint"]

        def positions():
            return np.array([joints[-1].position[joints[-1].name.index(n)] for n in joint_names])

        before = positions()
        text = cli(["--detect-only"])
        record = json.loads(next(line[len("[测量] "):] for line in text.splitlines() if line.startswith("[测量] ")))
        expected = np.array([initial.x, initial.y, initial.z])
        if placement == "relocated_base":
            from scipy.spatial.transform import Rotation
            expected = Rotation.from_euler("z", .1).apply(expected-np.array([-.01, -.06, 0.]))
        assert np.linalg.norm(np.array(record["center"])-expected) < .004
        assert abs(record["diameter"]-.05) < .0015 and abs(record["height"]-.1) < .004
        text = cli([])
        assert "[预览]" in text and "[阶段执行]" not in text
        text = cli(["--execute"], answer="n")
        assert "[取消] 未执行" in text and "[阶段执行]" not in text
        observe(.5)
        assert np.max(np.abs(positions()-before)) < .01
        poses.clear()
        text = cli(["--execute"], answer="y")
        assert "[视觉确认]" in text and "[完成] RGB-D" in text
        assert max(p.z for p in poses)-initial.z > .05
        assert max(np.hypot(p.x-initial.x, p.y-initial.y) for p in poses) < .005
        observe(1)
        assert np.linalg.norm(np.array([poses[-1].x, poses[-1].y, poses[-1].z])-
                              [initial.x, initial.y, initial.z]) < .015
        text = cli([])
        assert "[跳过] open" in text and "[预览]" in text
    finally:
        try:
            os.killpg(simulation.pid, signal.SIGINT)
            simulation.wait(timeout=12)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            pass
        for signum in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(simulation.pid, signum)
                time.sleep(.2)
            except ProcessLookupError:
                break
        simulation.wait()
        log.close()
        node.destroy_node()
        rclpy.shutdown()
