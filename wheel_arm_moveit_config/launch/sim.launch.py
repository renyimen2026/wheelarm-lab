from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    def launch(package, name, arguments):
        return IncludeLaunchDescription(PythonLaunchDescriptionSource(
            PathJoinSubstitution([FindPackageShare(package), "launch", name])),
            launch_arguments=arguments.items())

    return LaunchDescription([
        DeclareLaunchArgument("world", default_value=PathJoinSubstitution(
            [FindPackageShare("wheel_arm_gazebo"), "worlds", "empty.sdf"])),
        DeclareLaunchArgument("rviz", default_value="true"),
        DeclareLaunchArgument("gui", default_value="true"),
        DeclareLaunchArgument("camera_mount", default_value="low", choices=["low", "mast"]),
        DeclareLaunchArgument("initial_shoulder_lift", default_value="0.0"),
        DeclareLaunchArgument("initial_elbow", default_value="0.0"),
        DeclareLaunchArgument("initial_wrist_pitch", default_value="0.0"),
        DeclareLaunchArgument("allow_trajectory_execution", default_value="true"),
        DeclareLaunchArgument("plugin_lifetime_workaround", default_value="true"),
        launch("wheel_arm_gazebo", "sim.launch.py", {
            "world": LaunchConfiguration("world"), "spawn_x": "0", "spawn_y": "0",
            "spawn_z": "0.05", "spawn_yaw": "0", "localization_debug": "false",
            "use_sim_time": "true",
            "gui": LaunchConfiguration("gui"),
            "camera_mount": LaunchConfiguration("camera_mount"),
            "initial_shoulder_lift": LaunchConfiguration("initial_shoulder_lift"),
            "initial_elbow": LaunchConfiguration("initial_elbow"),
            "initial_wrist_pitch": LaunchConfiguration("initial_wrist_pitch"),
        }),
        launch("wheel_arm_moveit_config", "moveit.launch.py", {
            "rviz": LaunchConfiguration("rviz"), "use_sim_time": "true",
            "camera_mount": LaunchConfiguration("camera_mount"),
            "allow_trajectory_execution": LaunchConfiguration("allow_trajectory_execution"),
            "plugin_lifetime_workaround": LaunchConfiguration("plugin_lifetime_workaround"),
        }),
    ])
