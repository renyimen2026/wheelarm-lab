from pathlib import Path
import shutil
import tempfile

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, LogInfo, OpaqueFunction, RegisterEventHandler
from launch.event_handlers import OnShutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
import yaml

from wheel_arm_manipulation.house_navigation import parameters, planar, reference, dock_reference
from wheel_arm_manipulation.living_room import write_world


def start(context):
    gazebo = Path(get_package_share_directory("wheel_arm_gazebo"))
    moveit = Path(get_package_share_directory("wheel_arm_moveit_config"))
    map_path = Path(LaunchConfiguration("map").perform(context)).expanduser().resolve()
    if not map_path.is_file():
        raise ValueError(f"Saved map does not exist: {map_path}")
    alignment = LaunchConfiguration("reference").perform(context)
    _, world_start, map_start = reference(alignment)
    directory = Path(tempfile.mkdtemp(prefix="wheelarm-house-nav-"))
    try:
        _, _, world_dock = write_world(LaunchConfiguration("layout").perform(context), gazebo, directory / "world.sdf")
        map_dock = dock_reference(LaunchConfiguration("layout").perform(context), alignment, gazebo)[0]
        if LaunchConfiguration("start_at_dock").perform(context).lower() == "true":
            world_start, map_start = world_dock, map_dock
        with (directory / "nav.yaml").open("w", encoding="utf-8") as stream:
            yaml.safe_dump(parameters(gazebo / "config/nav2_params.yaml", map_start), stream)
    except Exception:
        shutil.rmtree(directory)
        raise

    def cleanup(_):
        shutil.rmtree(directory, ignore_errors=True)
        return []

    spawn, dock = planar(world_start), planar(map_dock)
    return [
        RegisterEventHandler(OnShutdown(on_shutdown=[OpaqueFunction(function=cleanup)])),
        LogInfo(msg=f"House map reference: {alignment}. Expected living-room dock (map): {dock}. "
                    "Verify AMCL scan/map alignment before executing tasks."),
        IncludeLaunchDescription(PythonLaunchDescriptionSource(str(gazebo / "launch/nav.launch.py")),
            launch_arguments={"world": str(directory / "world.sdf"), "slam": "False", "map": str(map_path),
                "params_file": str(directory / "nav.yaml"), "use_ekf": "true", "localization_debug": "false",
                "camera_mount": "mast", "initial_shoulder_lift": "-1.4", "initial_elbow": "2.1",
                "initial_wrist_pitch": "-0.7", "spawn_x": str(spawn["x"]), "spawn_y": str(spawn["y"]),
                "spawn_yaw": str(spawn["yaw"]), "gui": LaunchConfiguration("gui"),
                "rviz": LaunchConfiguration("rviz")}.items()),
        IncludeLaunchDescription(PythonLaunchDescriptionSource(str(moveit / "launch/moveit.launch.py")),
            launch_arguments={"camera_mount": "mast", "rviz": LaunchConfiguration("moveit_rviz")}.items()),
    ]


def generate_launch_description():
    share = Path(get_package_share_directory("wheel_arm_manipulation"))
    return LaunchDescription([
        DeclareLaunchArgument("map", default_value="maps/wheel_arm_map.yaml"),
        DeclareLaunchArgument("layout", default_value=str(share / "config/living_room.yaml")),
        DeclareLaunchArgument("reference", default_value=str(share / "config/house_map_reference.yaml")),
        DeclareLaunchArgument("gui", default_value="true"),
        DeclareLaunchArgument("rviz", default_value="true"),
        DeclareLaunchArgument("moveit_rviz", default_value="false"),
        DeclareLaunchArgument("start_at_dock", default_value="false",
                             description="Calibration only: start physically at the cup dock"),
        OpaqueFunction(function=start),
    ])
