import math
from pathlib import Path
import tempfile

import yaml

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, LogInfo, OpaqueFunction, RegisterEventHandler
from launch.event_handlers import OnShutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def validate_prior(value):
    if not isinstance(value, dict) or set(value) != {"frame_id", "x", "y", "yaw"}:
        raise ValueError("Initial pose must contain exactly frame_id, x, y and yaw")
    if value["frame_id"] != "map":
        raise ValueError("Initial pose must be in the map frame, not Gazebo world or odom")
    result = {"frame_id": "map"}
    for key in ("x", "y", "yaw"):
        number = value[key]
        if isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(number):
            raise ValueError(f"Initial pose {key} must be a finite number")
        result[key] = float(number)
    return result


def resolve_prior(filename, coordinates):
    provided = [value.strip() != "" for value in coordinates]
    if filename and any(provided):
        raise ValueError("Use initial_pose_file OR initial_pose_x/y/yaw, not both")
    if any(provided):
        if not all(provided):
            raise ValueError("Provide all three initial_pose_x, initial_pose_y and initial_pose_yaw")
        return validate_prior(dict(zip(("frame_id", "x", "y", "yaw"),
                                       ("map", *(float(value) for value in coordinates)))))
    if filename:
        path = Path(filename).expanduser().resolve()
        with path.open(encoding="utf-8") as stream:
            return validate_prior(yaml.safe_load(stream))
    return None


def parameters_with_prior(source, prior):
    with Path(source).expanduser().open(encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    if (not isinstance(document, dict) or not isinstance(document.get("amcl"), dict)
            or not isinstance(document["amcl"].get("ros__parameters"), dict)):
        raise ValueError("Nav2 parameters must contain amcl.ros__parameters")
    parameters = document["amcl"]["ros__parameters"]
    if parameters.get("global_frame_id", "map") != prior["frame_id"]:
        raise ValueError("Initial pose frame does not match AMCL global_frame_id")
    parameters["set_initial_pose"] = True
    parameters["initial_pose"] = {"x": prior["x"], "y": prior["y"], "z": 0.0, "yaw": prior["yaw"]}
    return document


def remove_parameters(context, path):
    Path(path).unlink(missing_ok=True)
    return []


def launch_localization(context):
    map_path = Path(LaunchConfiguration("map").perform(context)).expanduser().resolve()
    if not map_path.is_file():
        raise RuntimeError(
            f"Map YAML does not exist: {map_path}. "
            "Pass map:=maps/wheel_arm_map.yaml from your workspace root."
        )

    prior = resolve_prior(LaunchConfiguration("initial_pose_file").perform(context),
                          [LaunchConfiguration("initial_pose_" + key).perform(context)
                           for key in ("x", "y", "yaw")])
    arguments = {"slam": "False", "map": str(map_path)}
    actions = []
    if prior is not None:
        use_ekf = LaunchConfiguration("use_ekf").perform(context).lower()
        if use_ekf not in ("true", "false", "1", "0"):
            raise ValueError("use_ekf must be true or false")
        selected = "params_file" if use_ekf in ("true", "1") else "raw_odom_params_file"
        document = parameters_with_prior(LaunchConfiguration(selected).perform(context), prior)
        # Override AMCL only; preserve the selected EKF/raw-odom navigation configuration.
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".yaml",
                                         prefix="wheelarm-amcl-", delete=False) as stream:
            yaml.safe_dump(document, stream)
            path = stream.name
        arguments[selected] = path
        actions.extend([
            RegisterEventHandler(OnShutdown(on_shutdown=[
                OpaqueFunction(function=remove_parameters, kwargs={"path": path})])),
            LogInfo(msg=f"AMCL startup prior (map): x={prior['x']}, y={prior['y']}, "
                        f"yaw={prior['yaw']} rad. Verify scan/map alignment; RViz can reset it."),
        ])
    else:
        actions.append(LogInfo(msg="No startup prior override. AMCL uses its parameter file; "
                                   "with default parameters, set 2D Pose Estimate in RViz."))
    return actions + [
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution(
                    [FindPackageShare("wheel_arm_gazebo"), "launch", "nav.launch.py"]
                )
            ),
            launch_arguments=arguments.items(),
        )
    ]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("map", description="Saved occupancy map YAML path"),
            DeclareLaunchArgument("localization_debug", default_value="false"),
            DeclareLaunchArgument("use_ekf", default_value="true"),
            DeclareLaunchArgument("initial_pose_file", default_value="",
                                 description="YAML prior in map frame; mutually exclusive with coordinates"),
            DeclareLaunchArgument("initial_pose_x", default_value="", description="Map x in meters"),
            DeclareLaunchArgument("initial_pose_y", default_value="", description="Map y in meters"),
            DeclareLaunchArgument("initial_pose_yaw", default_value="", description="Map yaw in radians"),
            DeclareLaunchArgument("params_file", default_value=PathJoinSubstitution(
                [FindPackageShare("wheel_arm_gazebo"), "config", "nav2_params.yaml"])),
            DeclareLaunchArgument("raw_odom_params_file", default_value=PathJoinSubstitution(
                [FindPackageShare("wheel_arm_gazebo"), "config", "nav2_params_raw_odom.yaml"])),
            OpaqueFunction(function=launch_localization),
        ]
    )
