import importlib.util
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest
import xacro
import yaml


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
ARM_JOINTS = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_pitch_joint"]


def read_yaml(name):
    return yaml.safe_load((ROOT / "config" / name).read_text())


@pytest.fixture
def model():
    xml = xacro.process_file(str(REPO / "wheel_arm_description/urdf/wheel_arm.urdf.xacro"),
                             mappings={"use_gazebo_control": "false"}).toxml()
    return ET.fromstring(xml), ET.parse(ROOT / "config/wheel_arm.srdf").getroot()


def test_chain_matches_existing_four_joint_arm(model):
    urdf, srdf = model
    chain = srdf.find("group/chain")
    parents = {joint.find("child").get("link"): joint for joint in urdf.findall("joint")}
    child, joints = chain.get("tip_link"), []
    while child != chain.get("base_link"):
        joint = parents[child]
        if joint.get("type") != "fixed":
            joints.append(joint.get("name"))
        child = joint.find("parent").get("link")
    assert joints[::-1] == ARM_JOINTS
    assert chain.get("tip_link") == "tool0"
    assert srdf.find("virtual_joint") is None
    assert {p.get("name") for p in srdf.findall("passive_joint")} == {
        "left_wheel_joint", "right_wheel_joint"}


def test_named_states_and_planning_limits_respect_urdf(model):
    urdf, srdf = model
    joints = {joint.get("name"): joint for joint in urdf.findall("joint")}
    limits = read_yaml("joint_limits.yaml")
    assert 0 < limits["default_velocity_scaling_factor"] <= 0.2
    assert 0 < limits["default_acceleration_scaling_factor"] <= 0.2
    assert {state.get("name") for state in srdf.findall("group_state[@group='arm']")} == {
        "home", "ready", "cup_pregrasp", "cup_grasp", "cup_lift"}
    for state in srdf.findall("group_state[@group='arm']"):
        assert [joint.get("name") for joint in state] == ARM_JOINTS
        for joint in state:
            limit = joints[joint.get("name")].find("limit")
            assert float(limit.get("lower")) <= float(joint.get("value")) <= float(limit.get("upper"))
    for name, limit in limits["joint_limits"].items():
        assert 0 < limit["max_velocity"] <= float(joints[name].find("limit").get("velocity"))
        assert limit["has_acceleration_limits"] and limit["max_acceleration"] > 0


def test_collision_exemptions_are_only_neighbors_and_rigid_fixture(model):
    urdf, srdf = model
    links = {link.get("name") for link in urdf.findall("link")}
    neighbors = {frozenset((joint.find("parent").get("link"), joint.find("child").get("link")))
                 for joint in urdf.findall("joint")}
    extras = {frozenset(("arm_mount_link", "imu_link")),
              frozenset(("left_finger_link", "right_finger_link"))}
    for pair in srdf.findall("disable_collisions"):
        names = frozenset((pair.get("link1"), pair.get("link2")))
        assert names <= links
        assert names in neighbors or names in extras


def test_controller_targets_existing_namespaced_action():
    config = read_yaml("moveit_controllers.yaml")
    manager = config["moveit_simple_controller_manager"]
    assert manager["controller_names"] == ["wheel_arm/arm_controller", "wheel_arm/gripper_controller"]
    controller = manager[manager["controller_names"][0]]
    assert controller["action_ns"] == "follow_joint_trajectory"
    assert controller["type"] == "FollowJointTrajectory"
    assert controller["joints"] == ARM_JOINTS
    for name in ("controllers.yaml", "controllers_raw_odom.yaml"):
        existing = yaml.safe_load((REPO / "wheel_arm_control/config" / name).read_text())
        assert controller["joints"] == existing["/**/arm_controller"]["ros__parameters"]["joints"]


def test_gripper_geometry_limits_and_controller_agree(model):
    urdf, srdf = model
    names = ["left_finger_joint", "right_finger_joint"]
    assert [j.get("name") for j in srdf.findall("group[@name='gripper']/joint")] == names
    for name, sign in zip(names, (1, -1)):
        joint = urdf.find(f"joint[@name='{name}']")
        assert joint.get("type") == "prismatic"
        assert joint.find("parent").get("link") == "tool0"
        assert float(joint.find("axis").get("xyz").split()[1]) == sign
        assert float(joint.find("limit").get("lower")) == 0
        assert float(joint.find("limit").get("upper")) == 0.04
        assert urdf.find(f"ros2_control/joint[@name='{name}']/command_interface").get("name") == "position"
    assert urdf.find("joint[@name='gripper_tcp_joint']/origin").get("xyz") == "0.12 0 0"
    assert srdf.find("end_effector").get("group") == "gripper"
    for state, value in (("open", 0.04), ("closed", 0.0)):
        assert [float(j.get("value")) for j in srdf.findall(f"group_state[@name='{state}']/joint")] == [value] * 2
    controller = read_yaml("moveit_controllers.yaml")["moveit_simple_controller_manager"]["wheel_arm/gripper_controller"]
    assert controller["joints"] == names
    for filename in ("controllers.yaml", "controllers_raw_odom.yaml"):
        config = yaml.safe_load((REPO / "wheel_arm_control/config" / filename).read_text())
        assert config["/**/gripper_controller"]["ros__parameters"]["joints"] == names
        assert not config["/**/gripper_controller"]["ros__parameters"]["allow_partial_joints_goal"]


def test_optional_camera_mast_preserves_low_model_and_arm_geometry():
    path = str(REPO / "wheel_arm_description/urdf/wheel_arm.urdf.xacro")
    low = ET.fromstring(xacro.process_file(path).toxml())
    explicit = ET.fromstring(xacro.process_file(path, mappings={"camera_mount": "low"}).toxml())
    mast = ET.fromstring(xacro.process_file(path, mappings={"camera_mount": "mast"}).toxml())
    assert ET.tostring(low) == ET.tostring(explicit)
    assert low.find("link[@name='camera_mast_link']") is None
    assert low.find("joint[@name='camera_joint']/parent").get("link") == "base_link"
    assert low.find("joint[@name='camera_joint']/origin").get("xyz") == "0.34 0 0.13999999999999999"
    assert mast.find("joint[@name='camera_mast_joint']/origin").get("xyz") == "-0.18 0.19 0.18"
    assert mast.find("joint[@name='camera_joint']/origin").get("rpy") == "0 0.35 -0.22"
    # The sensor origin must not coincide with the mast's top face.
    origin = mast.find("joint[@name='camera_joint']/origin")
    assert float(origin.get("xyz").split()[2]) == pytest.approx(.63+.0275)
    assert mast.find("link[@name='camera_mast_link']/collision") is not None
    assert mast.find("link[@name='camera_mast_link']/inertial") is not None
    for element in low:
        if element.tag == "joint" and element.get("name") == "camera_joint":
            continue
        if element.get("name") is not None:
            other = mast.find(f"{element.tag}[@name='{element.get('name')}']")
            assert ET.tostring(element) == ET.tostring(other)


@pytest.mark.parametrize("mount", ["low", "mast"])
def test_semantic_camera_variant_only_adds_rigid_mount_exemptions(mount):
    srdf = ET.fromstring(xacro.process_file(str(ROOT / "config/wheel_arm.srdf.xacro"),
                                           mappings={"camera_mount": mount}).toxml())
    original = ET.parse(ROOT / "config/wheel_arm.srdf").getroot()
    exemptions = lambda tree: {(p.get("link1"), p.get("link2")) for p in tree.findall("disable_collisions")}
    extra = exemptions(srdf) - exemptions(original)
    assert extra == (set() if mount == "low" else {
        ("base_link", "camera_mast_link"), ("camera_mast_link", "camera_link")})
    for p in srdf.findall("disable_collisions"):
        if "camera_mast_link" in (p.get("link1"), p.get("link2")):
            assert p.get("link1") not in {"shoulder_link", "upper_arm_link", "forearm_link", "wrist_link"}


def test_position_only_ik_and_rviz_group():
    assert read_yaml("kinematics.yaml")["arm"]["position_only_ik"] is True
    rviz = read_yaml("moveit.rviz")["Visualization Manager"]
    assert rviz["Global Options"]["Fixed Frame"] == "base_footprint"
    display = next(d for d in rviz["Displays"] if d["Name"] == "MotionPlanning")
    assert display["Planning Request"]["Planning Group"] == "arm"


def test_rviz_depth_uses_original_image_topic():
    displays = read_yaml("moveit.rviz")["Visualization Manager"]["Displays"]
    depth = next(display for display in displays if display["Name"] == "Depth Image")
    assert depth["Class"] == "rviz_default_plugins/Image"
    assert depth["Enabled"] and depth["Topic"]["Value"] == "/camera/depth_image"
    assert depth["Transport Hint"] == "raw"
    assert depth["Normalize Range"] is False
    assert depth["Min Value"] == 0.1 and depth["Max Value"] == 3.0


def test_installed_launch_loads_without_second_controller_manager():
    spec = importlib.util.spec_from_file_location("arm_moveit_launch", ROOT / "launch/moveit.launch.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from launch_ros.actions import Node
    description = module.generate_launch_description()
    nodes = [action for action in description.entities if isinstance(action, Node)]
    assert len(nodes) == 2  # move_group and RViz, no second state publisher or controller manager.
    from launch import LaunchContext
    from launch.utilities import perform_substitutions
    context = LaunchContext()
    context.launch_configurations["plugin_lifetime_workaround"] = "true"
    environments = [dict((perform_substitutions(context, key), perform_substitutions(context, value))
                         for key, value in node.additional_env) for node in nodes]
    assert "QT_USE_NATIVE_WINDOWS" not in environments[0]
    assert environments[1]["QT_USE_NATIVE_WINDOWS"] == "1"


@pytest.mark.parametrize("gui,expected", [("true", "-r test.sdf"), ("false", "-r -s test.sdf")])
def test_gazebo_gui_flag_preserves_default_and_can_run_server_only(gui, expected):
    from launch import LaunchContext
    from launch.actions import IncludeLaunchDescription
    from launch.utilities import normalize_to_list_of_substitutions, perform_substitutions
    spec = importlib.util.spec_from_file_location("arm_sim_launch", REPO / "wheel_arm_gazebo/launch/sim.launch.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    description = module.generate_launch_description()
    gazebo = next(action for action in description.entities if isinstance(action, IncludeLaunchDescription))
    context = LaunchContext()
    context.launch_configurations.update(gui=gui, world="test.sdf")
    arguments = dict(gazebo.launch_arguments)
    assert perform_substitutions(context, normalize_to_list_of_substitutions(arguments["gz_args"])) == expected
