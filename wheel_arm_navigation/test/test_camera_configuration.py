import ast
from pathlib import Path
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[2]


def test_camera_configuration_matches_capture_contract():
    robot = ET.parse(ROOT / "wheel_arm_description/urdf/wheel_arm.urdf.xacro")
    sensor = robot.find(".//gazebo[@reference='camera_link']/sensor")
    assert sensor.attrib["type"] == "rgbd_camera"
    assert sensor.findtext("topic") == "/camera"
    assert float(sensor.findtext("update_rate")) == 15
    assert sensor.findtext("camera/image/width") == "640"
    assert sensor.findtext("camera/image/height") == "480"
    assert float(sensor.findtext("camera/clip/near")) == 0.1
    assert float(sensor.findtext("camera/clip/far")) == 8
    joint = robot.find(".//joint[@name='camera_optical_joint']")
    assert joint.find("parent").attrib["link"] == "camera_link"
    assert joint.find("child").attrib["link"] == "camera_optical_frame"
    assert joint.find("origin").attrib["rpy"] == "${-pi/2} 0 ${-pi/2}"


def test_camera_bridge_and_description_parameter_types():
    launch = ast.parse((ROOT / "wheel_arm_gazebo/launch/sim.launch.py").read_text())
    typed = [node for node in ast.walk(launch) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name) and node.func.id == "ParameterValue"]
    assert len(typed) == 2
    assert all(any(keyword.arg == "value_type" and isinstance(keyword.value, ast.Name)
                   and keyword.value.id == "str" for keyword in node.keywords) for node in typed)
    camera = next(node.value for node in ast.walk(launch) if isinstance(node, ast.Assign)
                  and any(isinstance(target, ast.Name) and target.id == "camera_bridge"
                          for target in node.targets))
    parameters = next(keyword.value for keyword in camera.keywords if keyword.arg == "parameters")
    assert ast.literal_eval(parameters) == [{"override_frame_id": "camera_optical_frame"}]
    arguments = next(keyword.value for keyword in camera.keywords if keyword.arg == "arguments")
    topics = ast.literal_eval(arguments)
    assert len(topics) == 3 and not any("PointCloud" in topic for topic in topics)
    assert any(topic.startswith("/camera/image@") for topic in topics)
    assert any(topic.startswith("/camera/depth_image@") for topic in topics)
    assert any(topic.startswith("/camera/camera_info@") for topic in topics)
