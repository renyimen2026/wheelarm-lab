from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def launch_localization(context):
    map_path = Path(LaunchConfiguration("map").perform(context)).expanduser().resolve()
    if not map_path.is_file():
        raise RuntimeError(
            f"Map YAML does not exist: {map_path}. "
            "Pass map:=maps/wheel_arm_map.yaml from your workspace root."
        )

    return [
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution(
                    [FindPackageShare("wheel_arm_gazebo"), "launch", "nav.launch.py"]
                )
            ),
            launch_arguments={"slam": "False", "map": str(map_path)}.items(),
        )
    ]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("map", description="Saved occupancy map YAML path"),
            DeclareLaunchArgument("localization_debug", default_value="false"),
            OpaqueFunction(function=launch_localization),
        ]
    )
