from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, IncludeLaunchDescription, TimerAction
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node, SetRemap
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    use_sim_time = LaunchConfiguration("use_sim_time")
    world = LaunchConfiguration("world")
    params_file = LaunchConfiguration("params_file")
    autostart = LaunchConfiguration("autostart")
    rviz = LaunchConfiguration("rviz")
    rviz_config = LaunchConfiguration("rviz_config")
    use_ekf = LaunchConfiguration("use_ekf")
    localization_debug = LaunchConfiguration("localization_debug")
    raw_odom_params_file = LaunchConfiguration("raw_odom_params_file")

    sim_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [FindPackageShare("wheel_arm_gazebo"), "/launch/sim.launch.py"]
        ),
        launch_arguments={
            "use_sim_time": use_sim_time,
            "world": world,
            "use_ekf": use_ekf,
            "localization_debug": localization_debug,
        }.items(),
    )

    def make_nav2_bringup(selected_params_file, condition):
        return GroupAction(
            [
                SetRemap(
                    src="/cmd_vel",
                    dst="/wheel_arm/diff_drive_base_controller/cmd_vel_unstamped",
                ),
                SetRemap(
                    src="/cmd_vel_smoothed",
                    dst="/wheel_arm/diff_drive_base_controller/cmd_vel_unstamped",
                ),
                IncludeLaunchDescription(
                    PythonLaunchDescriptionSource(
                        [FindPackageShare("nav2_bringup"), "/launch/bringup_launch.py"]
                    ),
                    launch_arguments={
                        "namespace": "",
                        "use_namespace": "False",
                        "slam": "True",
                        "map": "",
                        "use_sim_time": use_sim_time,
                        "params_file": selected_params_file,
                        "autostart": autostart,
                    }.items(),
                ),
            ],
            condition=condition,
        )

    nav2_bringup = make_nav2_bringup(params_file, IfCondition(use_ekf))
    raw_odom_nav2_bringup = make_nav2_bringup(
        raw_odom_params_file,
        UnlessCondition(use_ekf),
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", rviz_config],
        parameters=[{"use_sim_time": use_sim_time}],
        condition=IfCondition(rviz),
    )

    delayed_nav_stack = TimerAction(
        period=10.0,
        actions=[nav2_bringup, raw_odom_nav2_bringup, rviz_node],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("use_sim_time", default_value="true"),
            DeclareLaunchArgument(
                "world",
                default_value=PathJoinSubstitution(
                    [FindPackageShare("wheel_arm_gazebo"), "worlds", "indoor_lab.sdf"]
                ),
            ),
            DeclareLaunchArgument(
                "params_file",
                default_value=PathJoinSubstitution(
                    [FindPackageShare("wheel_arm_gazebo"), "config", "nav2_params.yaml"]
                ),
            ),
            DeclareLaunchArgument(
                "raw_odom_params_file",
                default_value=PathJoinSubstitution(
                    [
                        FindPackageShare("wheel_arm_gazebo"),
                        "config",
                        "nav2_params_raw_odom.yaml",
                    ]
                ),
            ),
            DeclareLaunchArgument("autostart", default_value="true"),
            DeclareLaunchArgument("rviz", default_value="true"),
            DeclareLaunchArgument("use_ekf", default_value="true"),
            DeclareLaunchArgument("localization_debug", default_value="true"),
            DeclareLaunchArgument(
                "rviz_config",
                default_value=PathJoinSubstitution(
                    [FindPackageShare("wheel_arm_gazebo"), "rviz", "wheel_arm_nav.rviz"]
                ),
            ),
            sim_launch,
            delayed_nav_stack,
        ]
    )
