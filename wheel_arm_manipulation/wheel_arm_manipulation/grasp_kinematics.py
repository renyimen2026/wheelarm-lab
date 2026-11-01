"""URDF-based numerical pose IK, cross-checked by MoveIt before motion."""

from pathlib import Path
import xml.etree.ElementTree as ET

from ament_index_python.packages import get_package_share_directory
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
import xacro


class GraspKinematics:
    def __init__(self, model):
        path = Path(get_package_share_directory("wheel_arm_description")) / "urdf/wheel_arm.urdf.xacro"
        robot = ET.fromstring(xacro.process_file(str(path), mappings={"use_gazebo_control": "false"}).toxml())
        self.parents = {joint.find("child").get("link"): joint for joint in robot.findall("joint")}
        self.model = model
        self.chain = []
        link = "gripper_tcp"
        while link != model.frame:
            joint = self.parents[link]
            origin = joint.find("origin")
            transform = np.eye(4)
            if origin is not None:
                transform[:3, 3] = [float(v) for v in origin.get("xyz", "0 0 0").split()]
                transform[:3, :3] = Rotation.from_euler("xyz", [float(v) for v in origin.get("rpy", "0 0 0").split()]).as_matrix()
            axis = joint.find("axis")
            self.chain.append((joint.get("name"), joint.get("type"), transform,
                               np.array([float(v) for v in axis.get("xyz").split()]) if axis is not None else None))
            link = joint.find("parent").get("link")
        self.chain.reverse()
        transform = np.eye(4)
        for _, kind, origin, _ in self.chain:
            transform = transform @ origin
            if kind == "revolute":
                break
        self.axis_origin = transform[:3, 3].copy()

    def forward(self, values):
        transform = np.eye(4)
        for name, kind, origin, axis in self.chain:
            transform = transform @ origin
            if kind == "revolute":
                rotation = np.eye(4)
                rotation[:3, :3] = Rotation.from_rotvec(axis*values[name]).as_matrix()
                transform = transform @ rotation
            elif kind != "fixed":
                raise ValueError("当前抓取链只支持固定和转动关节")
        return transform

    def solve(self, target, start):
        target = np.asarray(target, dtype=float)
        if target.shape != (3,) or not np.isfinite(target).all():
            raise ValueError("抓取目标必须是有限三维位置")
        yaw = np.arctan2(target[1]-self.axis_origin[1], target[0]-self.axis_origin[0])
        desired = Rotation.from_euler("z", yaw)
        lower, upper = np.array([self.model.bounds[n] for n in self.model.joints]).T

        def residual(q):
            tf = self.forward(dict(zip(self.model.joints, q)))
            return np.r_[5*(tf[:3, 3]-target), (desired.inv()*Rotation.from_matrix(tf[:3, :3])).as_rotvec()]

        seeds = [[start[n] for n in self.model.joints], [self.model.named["ready"][n] for n in self.model.joints],
                 (lower+upper)/2]
        for seed in seeds:
            result = least_squares(residual, np.clip(seed, lower+1e-6, upper-1e-6),
                                   bounds=(lower, upper), max_nfev=200, xtol=1e-10, ftol=1e-10, gtol=1e-10)
            error = residual(result.x)
            if np.linalg.norm(error[:3])/5 < .001 and np.linalg.norm(error[3:]) < .015:
                goal = dict(zip(self.model.joints, map(float, result.x)))
                self.model.check(goal)
                return goal
        raise ValueError("测量目标无法在关节限位内保持夹爪水平到达；未执行")
