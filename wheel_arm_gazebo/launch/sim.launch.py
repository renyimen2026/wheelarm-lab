from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, RegisterEventHandler, SetEnvironmentVariable
from launch.conditions import IfCondition, UnlessCondition
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, EnvironmentVariable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    use_sim_time = LaunchConfiguration("use_sim_time")
    world = LaunchConfiguration("world")
    use_ekf = LaunchConfiguration("use_ekf")
    ekf_config = LaunchConfiguration("ekf_config")
    localization_debug = LaunchConfiguration("localization_debug")
    robot_model = PathJoinSubstitution(
        [FindPackageShare("wheel_arm_description"), "urdf", "wheel_arm.urdf.xacro"]
    )
    ekf_controllers_file = PathJoinSubstitution(
        [FindPackageShare("wheel_arm_control"), "config", "controllers.yaml"]
    )
    raw_odom_controllers_file = PathJoinSubstitution(
        [FindPackageShare("wheel_arm_control"), "config", "controllers_raw_odom.yaml"]
    )

    robot_description_content = Command(
        [
            "xacro ",
            robot_model,
            " controllers_file:=",
            ekf_controllers_file,
        ]
    )

    raw_odom_robot_description_content = Command(
        [
            "xacro ",
            robot_model,
            " controllers_file:=",
            raw_odom_controllers_file,
        ]
    )

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        namespace="wheel_arm",
        output="screen",
        parameters=[
            {"robot_description": robot_description_content},
            {"use_sim_time": use_sim_time},
        ],
        condition=IfCondition(use_ekf),
    )

    raw_odom_robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        namespace="wheel_arm",
        output="screen",
        parameters=[
            {"robot_description": raw_odom_robot_description_content},
            {"use_sim_time": use_sim_time},
        ],
        condition=UnlessCondition(use_ekf),
    )

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [FindPackageShare("ros_gz_sim"), "/launch/gz_sim.launch.py"]
        ),
        launch_arguments={"gz_args": ["-r ", world]}.items(),
    )

    spawn_robot = Node(
        package="ros_gz_sim",
        executable="create",
        output="screen",
        arguments=[
            "-topic",
            "/wheel_arm/robot_description",
            "-name",
            "wheel_arm",
            "-allow_renaming",
            "false",
            "-x",
            LaunchConfiguration("spawn_x"),
            "-y",
            LaunchConfiguration("spawn_y"),
            "-z",
            LaunchConfiguration("spawn_z"),
            "-Y",
            LaunchConfiguration("spawn_yaw"),
        ],
    )

    scan_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        output="screen",
        arguments=[
            "/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan",
        ],
        parameters=[
            {"override_frame_id": "lidar_link"},
        ],
    )

    imu_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        output="screen",
        arguments=[
            "/imu/raw@sensor_msgs/msg/Imu[gz.msgs.IMU",
        ],
        parameters=[
            {"override_frame_id": "imu_link"},
        ],
    )

    imu_covariance_node = Node(
        package="wheel_arm_gazebo",
        executable="imu_covariance.py",
        name="imu_covariance",
        output="screen",
        parameters=[
            {"use_sim_time": use_sim_time},
            {"input_topic": "/imu/raw"},
            {"output_topic": "/imu"},
            {"frame_id": "imu_link"},
        ],
    )

    bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        output="screen",
        arguments=[
            "/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock",
            "/camera/image@sensor_msgs/msg/Image[gz.msgs.Image",
            "/camera/image/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo",
        ],
    )

    ekf_node = Node(
        package="robot_localization",
        executable="ekf_node",
        name="ekf_filter_node",
        output="screen",
        parameters=[
            ekf_config,
            {"use_sim_time": use_sim_time},
        ],
        condition=IfCondition(use_ekf),
    )

    localization_debug_node = Node(
        package="wheel_arm_gazebo",
        executable="localization_debug.py",
        name="localization_debug",
        output="screen",
        parameters=[
            {"use_sim_time": use_sim_time},
        ],
        condition=IfCondition(localization_debug),
    )

    joint_state_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "joint_state_broadcaster",
            "--controller-manager",
            "/wheel_arm/controller_manager",
            "--controller-manager-timeout",
            "60",
            "--service-call-timeout",
            "60",
            "--switch-timeout",
            "60",
        ],
        output="screen",
    )

    diff_drive_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "diff_drive_base_controller",
            "--controller-manager",
            "/wheel_arm/controller_manager",
            "--controller-manager-timeout",
            "60",
            "--service-call-timeout",
            "60",
            "--switch-timeout",
            "60",
        ],
        output="screen",
    )

    arm_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "arm_controller",
            "--controller-manager",
            "/wheel_arm/controller_manager",
            "--controller-manager-timeout",
            "60",
            "--service-call-timeout",
            "60",
            "--switch-timeout",
            "60",
        ],
        output="screen",
    )

    start_controllers_after_spawn = RegisterEventHandler(
        OnProcessExit(
            target_action=spawn_robot,
            on_exit=[joint_state_broadcaster_spawner],
        )
    )

    start_diff_drive_after_joint_states = RegisterEventHandler(
        OnProcessExit(
            target_action=joint_state_broadcaster_spawner,
            on_exit=[diff_drive_spawner],
        )
    )

    start_arm_after_diff_drive = RegisterEventHandler(
        OnProcessExit(
            target_action=diff_drive_spawner,
            on_exit=[arm_controller_spawner],
        )
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("use_sim_time", default_value="true"),
            DeclareLaunchArgument(
                "world",
                default_value=PathJoinSubstitution(
                    [FindPackageShare("wheel_arm_gazebo"), "worlds", "indoor_house.sdf"]
                ),
            ),
            DeclareLaunchArgument("use_ekf", default_value="true"),
            DeclareLaunchArgument("spawn_x", default_value="3.5"),
            DeclareLaunchArgument("spawn_y", default_value="1.0"),
            DeclareLaunchArgument("spawn_z", default_value="0.05"),
            DeclareLaunchArgument("spawn_yaw", default_value="0.0"),
            DeclareLaunchArgument("localization_debug", default_value="false"),
            DeclareLaunchArgument(
                "ekf_config",
                default_value=PathJoinSubstitution(
                    [FindPackageShare("wheel_arm_gazebo"), "config", "ekf.yaml"]
                ),
            ),
            SetEnvironmentVariable(
                "IGN_GAZEBO_RESOURCE_PATH",
                [PathJoinSubstitution([FindPackageShare("wheel_arm_gazebo"), "models"]),
                 ":", EnvironmentVariable("IGN_GAZEBO_RESOURCE_PATH", default_value="")],
            ),
            SetEnvironmentVariable(
                "GZ_SIM_RESOURCE_PATH",
                [PathJoinSubstitution([FindPackageShare("wheel_arm_gazebo"), "models"]),
                 ":", EnvironmentVariable("GZ_SIM_RESOURCE_PATH", default_value="")],
            ),
            gazebo,
            robot_state_publisher,
            raw_odom_robot_state_publisher,
            spawn_robot,
            scan_bridge,
            imu_bridge,
            imu_covariance_node,
            bridge,
            ekf_node,
            localization_debug_node,
            start_controllers_after_spawn,
            start_diff_drive_after_joint_states,
            start_arm_after_diff_drive,
        ]
    )
