from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("gui", default_value="true"),
        DeclareLaunchArgument("rviz", default_value="true"),
        DeclareLaunchArgument("camera_mount", default_value="low", choices=["low", "mast"]),
        DeclareLaunchArgument("cup_model", default_value="test_cup",
                              choices=["test_cup", "ceramic_mug", "ceramic_cup"]),
        DeclareLaunchArgument("initial_shoulder_lift", default_value="-1.05"),
        DeclareLaunchArgument("initial_elbow", default_value="2.1"),
        DeclareLaunchArgument("initial_wrist_pitch", default_value="-1.05"),
        IncludeLaunchDescription(PythonLaunchDescriptionSource(PathJoinSubstitution([
            FindPackageShare("wheel_arm_moveit_config"), "launch", "sim.launch.py"])),
            launch_arguments={
                "gui": LaunchConfiguration("gui"), "rviz": LaunchConfiguration("rviz"),
                "camera_mount": LaunchConfiguration("camera_mount"),
                "initial_shoulder_lift": LaunchConfiguration("initial_shoulder_lift"),
                "initial_elbow": LaunchConfiguration("initial_elbow"),
                "initial_wrist_pitch": LaunchConfiguration("initial_wrist_pitch"),
                "world": PathJoinSubstitution([
                    FindPackageShare("wheel_arm_gazebo"), "worlds", PythonExpression([
                        "{'test_cup': 'grasp_test.sdf', 'ceramic_mug': 'grasp_mug.sdf', ",
                        "'ceramic_cup': 'grasp_cup.sdf'}['", LaunchConfiguration("cup_model"), "']"])]),
            }.items()),
        Node(package="ros_gz_bridge", executable="parameter_bridge", name="cup_pose_bridge",
             arguments=["/model/test_cup/pose@geometry_msgs/msg/PoseStamped[ignition.msgs.Pose"],
             remappings=[("/model/test_cup/pose", "/grasp_test/cup_pose")],
             parameters=[{"use_sim_time": True}], output="screen"),
        Node(package="wheel_arm_manipulation", executable="grasp_fixture",
             parameters=[{"use_sim_time": True}], output="screen"),
    ])
