"""Explicit map/world reference and conservative docking-specific Nav2 overrides."""

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import yaml

from wheel_arm_manipulation.living_room import pose_matrix


def reference(path):
    with Path(path).expanduser().open(encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    if not isinstance(document, dict) or not {"world_pose", "map_pose"} <= set(document) or set(document)-{"world_pose", "map_pose", "dock_map_pose"}:
        raise ValueError("Map reference requires world_pose and map_pose")
    poses = []
    for key in ("world_pose", "map_pose"):
        xyz = np.asarray(document[key], dtype=float)
        if xyz.shape != (3,) or not np.isfinite(xyz).all():
            raise ValueError("Map reference poses must contain finite x, y, yaw")
        poses.append(pose_matrix([xyz[0], xyz[1], 0., 0., 0., xyz[2]]))
    world_base, map_base = poses
    return map_base @ np.linalg.inv(world_base), world_base, map_base


def planar(matrix):
    return {"x": float(matrix[0, 3]), "y": float(matrix[1, 3]),
            "yaw": float(Rotation.from_matrix(matrix[:3, :3]).as_euler("xyz")[2])}


def dock_reference(config, alignment, gazebo_share):
    from wheel_arm_manipulation.living_room import layout
    dock = layout(config, gazebo_share)[-1]
    with Path(alignment).expanduser().open(encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    nominal = reference(alignment)[0] @ dock
    if "dock_map_pose" in document:
        value = np.asarray(document["dock_map_pose"], dtype=float)
        if value.shape != (3,) or not np.isfinite(value).all():
            raise ValueError("dock_map_pose must be finite map x, y, yaw")
        nominal = pose_matrix([value[0], value[1], 0., 0., 0., value[2]])
    return nominal, nominal @ np.linalg.inv(dock)


def validate_goal(config, alignment, gazebo_share, goal):
    expected = planar(dock_reference(config, alignment, gazebo_share)[0])
    yaw_error = np.arctan2(np.sin(goal["yaw"]-expected["yaw"]), np.cos(goal["yaw"]-expected["yaw"]))
    if np.hypot(goal["x"]-expected["x"], goal["y"]-expected["y"]) > .06 or abs(yaw_error) > .08:
        raise ValueError("客厅目标不是当前杯子布局的抓取停靠点，请更新目标点和地图参考后再执行")
    return expected


def parameters(source, prior):
    with Path(source).open(encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    amcl = document["amcl"]["ros__parameters"]
    amcl.update(set_initial_pose=True, initial_pose={**planar(prior), "z": 0.})
    amcl.update(update_min_d=.02, update_min_a=.02, max_beams=120)
    controller = document["controller_server"]["ros__parameters"]
    controller["general_goal_checker"].update(xy_goal_tolerance=.04, yaw_goal_tolerance=.06, stateful=False)
    controller["progress_checker"].update(required_movement_radius=.05, movement_time_allowance=20.)
    controller["FollowPath"].update(xy_goal_tolerance=.04, max_vel_x=.18,
        max_speed_xy=.18, max_vel_theta=.45, trans_stopped_velocity=.015)
    # Convex hull of the .62 x .42 chassis and front-offset wheels (.53 outer width).
    footprint = "[[-0.31,-0.21],[0.07,-0.265],[0.25,-0.265],[0.31,-0.21],[0.31,0.21],[0.25,0.265],[0.07,0.265],[-0.31,0.21]]"
    for name in ("local_costmap", "global_costmap"):
        costmap = document[name][name]["ros__parameters"]
        costmap.pop("robot_radius", None)
        costmap.update(footprint=footprint, footprint_padding=.005)
    follow = controller["FollowPath"]
    follow.update(plugin="nav2_rotation_shim_controller::RotationShimController",
                  primary_controller="dwb_core::DWBLocalPlanner", angular_dist_threshold=.5,
                  forward_sampling_distance=.5, rotate_to_heading_angular_vel=.4,
                  max_angular_accel=1.2, simulate_ahead_time=1.)
    follow["critics"] = ["ObstacleFootprint" if item == "BaseObstacle" else item for item in follow["critics"]]
    follow["ObstacleFootprint.scale"] = follow.pop("BaseObstacle.scale")
    document["planner_server"]["ros__parameters"]["GridBased"]["tolerance"] = 0.
    return document
