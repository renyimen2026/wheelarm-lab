from pathlib import Path
import shutil
import tempfile

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, LogInfo, OpaqueFunction, RegisterEventHandler
from launch.event_handlers import OnShutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from scipy.spatial.transform import Rotation

from wheel_arm_manipulation.living_room import write_world


def start(context):
    package = Path(get_package_share_directory("wheel_arm_manipulation"))
    gazebo = Path(get_package_share_directory("wheel_arm_gazebo"))
    moveit = Path(get_package_share_directory("wheel_arm_moveit_config"))
    config = LaunchConfiguration("layout").perform(context)
    directory = tempfile.mkdtemp(prefix="wheelarm-living-room-")
    try:
        settings, cup, base = write_world(config, gazebo, Path(directory) / "world.sdf")
    except Exception:
        shutil.rmtree(directory)
        raise
    yaw = Rotation.from_matrix(base[:3, :3]).as_euler("xyz")[2]

    def cleanup(_):
        shutil.rmtree(directory, ignore_errors=True)
        return []

    return [
        RegisterEventHandler(OnShutdown(on_shutdown=[OpaqueFunction(function=cleanup)])),
        LogInfo(msg=f"{settings['name']} dock in Gazebo world: x={base[0,3]:.6f}, y={base[1,3]:.6f}, "
                    f"yaw={yaw:.6f}. Not map coordinates. Stop the base before grasping."),
        IncludeLaunchDescription(PythonLaunchDescriptionSource(str(gazebo / "launch/sim.launch.py")),
            launch_arguments={"world": str(Path(directory) / "world.sdf"),
                "spawn_x": str(base[0,3]), "spawn_y": str(base[1,3]), "spawn_yaw": str(yaw), "spawn_z": ".05",
                "camera_mount": "mast", "initial_shoulder_lift": "-1.05", "initial_elbow": "2.1",
                "initial_wrist_pitch": "-1.05", "gui": LaunchConfiguration("gui"),
                "localization_debug": "false"}.items()),
        IncludeLaunchDescription(PythonLaunchDescriptionSource(str(moveit / "launch/moveit.launch.py")),
            launch_arguments={"camera_mount": "mast", "rviz": LaunchConfiguration("rviz")}.items()),
        Node(package="wheel_arm_manipulation", executable="living_room_fixture",
             parameters=[{"use_sim_time": True, "layout": config}], output="screen"),
        Node(package="ros_gz_bridge", executable="parameter_bridge", name="cup_pose_bridge",
             arguments=["/model/test_cup/pose@geometry_msgs/msg/PoseStamped[ignition.msgs.Pose"],
             remappings=[("/model/test_cup/pose", "/grasp_test/cup_pose")], output="screen"),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("gui", default_value="true"),
        DeclareLaunchArgument("rviz", default_value="true"),
        DeclareLaunchArgument("layout", default_value=str(Path(get_package_share_directory(
            "wheel_arm_manipulation")) / "config/living_room.yaml")),
        OpaqueFunction(function=start),
    ])
