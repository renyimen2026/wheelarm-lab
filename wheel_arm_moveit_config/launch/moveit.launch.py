from pathlib import Path

from ament_index_python.packages import get_package_prefix, get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.conditions import IfCondition
from launch.substitutions import EnvironmentVariable, LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():
    description = Path(get_package_share_directory("wheel_arm_description"))
    share = Path(get_package_share_directory("wheel_arm_moveit_config"))
    config = (
        MoveItConfigsBuilder("wheel_arm", package_name="wheel_arm_moveit_config")
        .robot_description(file_path=str(description / "urdf/wheel_arm.urdf.xacro"),
                           mappings={"use_gazebo_control": "false",
                                     "camera_mount": LaunchConfiguration("camera_mount")})
        .robot_description_semantic(file_path="config/wheel_arm.srdf.xacro",
                                    mappings={"camera_mount": LaunchConfiguration("camera_mount")})
        .robot_description_kinematics(file_path="config/kinematics.yaml")
        .joint_limits(file_path="config/joint_limits.yaml")
        .trajectory_execution(file_path="config/moveit_controllers.yaml",
                              moveit_manage_controllers=False)
        .planning_pipelines(pipelines=["ompl"], default_planning_pipeline="ompl")
        .planning_scene_monitor(publish_robot_description=True,
                                publish_robot_description_semantic=True)
        .to_moveit_configs()
    )
    use_sim_time = ParameterValue(LaunchConfiguration("use_sim_time"), value_type=bool)
    parameters = [config.to_dict(), {"use_sim_time": use_sim_time}]
    remappings = [("joint_states", LaunchConfiguration("joint_states_topic"))]
    guard = str(Path(get_package_prefix("wheel_arm_moveit_config")) /
                "lib/libwheel_arm_moveit_plugin_lifetime.so")
    environment = {"LD_PRELOAD": [PythonExpression([
        repr(guard + ":"), " if '", LaunchConfiguration("plugin_lifetime_workaround"),
        "'.lower() == 'true' else ''"]), EnvironmentVariable("LD_PRELOAD", default_value="")]}
    return LaunchDescription([
        DeclareLaunchArgument("use_sim_time", default_value="true"),
        DeclareLaunchArgument("camera_mount", default_value="low", choices=["low", "mast"]),
        DeclareLaunchArgument("joint_states_topic", default_value="/wheel_arm/joint_states"),
        DeclareLaunchArgument("rviz", default_value="true"),
        DeclareLaunchArgument("allow_trajectory_execution", default_value="true"),
        DeclareLaunchArgument("plugin_lifetime_workaround", default_value="true",
                              description="Keep Humble MoveIt plugin libraries mapped until exit"),
        LogInfo(msg="MoveIt arm only: stop base navigation and keyboard teleop. "
                    "Gazebo obstacles are NOT automatically in the planning scene."),
        Node(package="moveit_ros_move_group", executable="move_group", output="screen",
             parameters=parameters + [{"allow_trajectory_execution": ParameterValue(
                 LaunchConfiguration("allow_trajectory_execution"), value_type=bool)}],
             remappings=remappings, additional_env=environment),
        Node(package="rviz2", executable="rviz2", name="moveit_rviz", output="screen",
             arguments=["-d", str(share / "config/moveit.rviz")],
             parameters=parameters, remappings=remappings,
             # Keep native ancestors stable when Image docks embed an OGRE window on X11.
             additional_env={**environment, "QT_USE_NATIVE_WINDOWS": "1"},
             condition=IfCondition(LaunchConfiguration("rviz"))),
    ])
