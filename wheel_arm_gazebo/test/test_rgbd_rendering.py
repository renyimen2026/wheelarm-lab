"""Opt-in Fortress RGB-D regression at the previously failing mast/yaw poses."""

import os
from pathlib import Path
import signal
import subprocess
import time
import xml.etree.ElementTree as ET

import numpy as np
import pytest
import xacro


pytestmark = pytest.mark.skipif(os.environ.get("WHEELARM_GAZEBO_TEST") != "1",
                                reason="Opt-in isolated Gazebo rendering regression")


def test_mast_rgbd_and_lidar_remain_valid_when_rotated(tmp_path):
    import rclpy
    from cv_bridge import CvBridge
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Image, LaserScan

    if os.environ.get("ROS_DOMAIN_ID", "0") == "0":
        pytest.fail("Use an explicit unused ROS domain")
    repo = Path(__file__).resolve().parents[2]
    robot_xml = xacro.process_file(str(repo / "wheel_arm_description/urdf/wheel_arm.urdf.xacro"),
        mappings={"use_gazebo_control": "false", "camera_mount": "mast"}).toxml()
    (tmp_path / "robot.urdf").write_text(robot_xml)
    converted = subprocess.run(["ign", "sdf", "-p", str(tmp_path / "robot.urdf")],
                               capture_output=True, text=True, check=True)
    robot = ET.fromstring(converted.stdout).find("model")
    robot.set("name", "render_probe")
    ET.SubElement(robot, "static").text = "true"
    world = ET.parse(repo / "wheel_arm_gazebo/worlds/grasp_cup.sdf")
    world.getroot().find("world").append(robot)
    world.write(tmp_path / "world.sdf", encoding="utf-8", xml_declaration=True)
    environment = dict(os.environ, IGN_PARTITION=f"rgbd-render-{os.getpid()}",
        IGN_GAZEBO_RESOURCE_PATH=str(repo / "wheel_arm_gazebo/models"))
    processes, logs = [], []
    rclpy.init()
    node = rclpy.create_node("rgbd_render_regression")
    bridge, depths, rgbs, scans = CvBridge(), [], [], []

    def depth(message):
        data = bridge.imgmsg_to_cv2(message, "passthrough")
        depths.append(float((np.isfinite(data) & (data > 0)).mean()))

    node.create_subscription(Image, "/camera/depth_image", depth, qos_profile_sensor_data)
    node.create_subscription(Image, "/camera/image",
        lambda msg: rgbs.append(float(bridge.imgmsg_to_cv2(msg, "bgr8").std())), qos_profile_sensor_data)
    node.create_subscription(LaserScan, "/scan",
        lambda msg: scans.append(int(np.isfinite(msg.ranges).sum())), qos_profile_sensor_data)

    def start(command, name):
        log = (tmp_path / name).open("w")
        logs.append(log)
        processes.append(subprocess.Popen(command, env=environment, stdout=log,
            stderr=subprocess.STDOUT, start_new_session=True))

    def observe(seconds):
        end = time.monotonic()+seconds
        while time.monotonic() < end:
            rclpy.spin_once(node, timeout_sec=.01)

    try:
        start(["ros2", "run", "ros_gz_bridge", "parameter_bridge",
               "/camera/depth_image@sensor_msgs/msg/Image[ignition.msgs.Image",
               "/camera/image@sensor_msgs/msg/Image[ignition.msgs.Image",
               "/scan@sensor_msgs/msg/LaserScan[ignition.msgs.LaserScan"], "bridge.log")
        start(["ign", "gazebo", "-r", "-s", str(tmp_path / "world.sdf")], "simulation.log")
        end = time.monotonic()+30
        while (not depths or not rgbs or not scans) and time.monotonic() < end:
            observe(.1)
        assert depths and rgbs and scans, "Camera/lidar failed to start"
        records = []
        for yaw in [0, .01, -.01, .1, -.1, .25, -.25, 0]:
            a, b, c = len(depths), len(rgbs), len(scans)
            result = subprocess.run(["ign", "service", "-s", "/world/grasp_test/set_pose",
                "--reqtype", "ignition.msgs.Pose", "--reptype", "ignition.msgs.Boolean", "--timeout", "2000",
                "--req", f'name: "render_probe", position: {{z: 0.001}}, '
                         f'orientation: {{z: {np.sin(yaw/2)}, w: {np.cos(yaw/2)}}}'],
                env=environment, capture_output=True, text=True, timeout=5)
            assert result.returncode == 0 and "true" in result.stdout
            observe(1.5)
            assert depths[a:] and rgbs[b:] and scans[c:], f"No new data at yaw={yaw}"
            # The floor fills most of this fixed camera view. This also catches
            # half-frame invalid renders, not just completely black frames.
            assert min(depths[a:]) > .70, (yaw, depths[a:])
            assert min(rgbs[b:]) > 5., (yaw, rgbs[b:])
            assert min(scans[c:]) > 20, (yaw, scans[c:])
            records.append((yaw, len(depths[a:]), min(depths[a:]), min(rgbs[b:]), min(scans[c:])))
        (tmp_path / "measurements.txt").write_text(repr(records))
    finally:
        for process in processes:
            try:
                os.killpg(process.pid, signal.SIGINT)
                process.wait(timeout=8)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                pass
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
        for log in logs:
            log.close()
        node.destroy_node()
        rclpy.shutdown()
