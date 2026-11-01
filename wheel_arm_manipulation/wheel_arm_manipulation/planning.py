"""Read the shared robot model and validate position-only arm trajectories."""

from dataclasses import dataclass
import math
from pathlib import Path
import xml.etree.ElementTree as ET


@dataclass
class ArmModel:
    joints: tuple
    bounds: dict
    velocities: dict
    accelerations: dict
    named: dict
    frame: str = "base_footprint"
    tip: str = "tool0"
    group: str = "arm"
    start_tolerance: float = 0.03
    goal_tolerance: float = 0.01
    reached_tolerance: float = 0.05

    def check(self, values):
        if set(values) != set(self.joints):
            raise ValueError("机械臂关节不完整或包含额外关节")
        for name, value in values.items():
            lower, upper = self.bounds[name]
            if not math.isfinite(value) or not lower <= value <= upper:
                raise ValueError(f"关节 {name} 超出限位或不是有限数值：{value}")


def load_model(group="arm"):
    from ament_index_python.packages import get_package_share_directory
    import xacro
    import yaml

    description = Path(get_package_share_directory("wheel_arm_description"))
    config = Path(get_package_share_directory("wheel_arm_moveit_config")) / "config"
    urdf = ET.fromstring(xacro.process_file(str(description / "urdf/wheel_arm.urdf.xacro"),
                                            mappings={"use_gazebo_control": "false"}).toxml())
    srdf = ET.parse(config / "wheel_arm.srdf").getroot()
    definition = srdf.find(f"group[@name='{group}']")
    if group not in ("arm", "gripper") or definition is None:
        raise ValueError("不支持的规划组")
    chain = definition.find("chain")
    parents = {j.find("child").get("link"): j for j in urdf.findall("joint")}
    if chain is not None:
        current, names = chain.get("tip_link"), []
        while current != chain.get("base_link"):
            joint = parents[current]
            if joint.get("type") != "fixed":
                if joint.get("type") != "revolute":
                    raise ValueError("当前入口仅支持有位置限位的转动机械臂关节")
                names.append(joint.get("name"))
            current = joint.find("parent").get("link")
        names.reverse()
    else:
        names = [joint.get("name") for joint in definition.findall("joint")]
    joints = {j.get("name"): j for j in urdf.findall("joint")}
    limits = yaml.safe_load((config / "joint_limits.yaml").read_text())["joint_limits"]
    model = ArmModel(tuple(names), {}, {}, {}, {}, group=group,
                     tip=chain.get("tip_link") if chain is not None else "gripper_tcp")
    if group == "gripper":
        model.start_tolerance, model.goal_tolerance, model.reached_tolerance = 0.002, 0.001, 0.002
    for name in names:
        limit = joints[name].find("limit")
        model.bounds[name] = (float(limit.get("lower")), float(limit.get("upper")))
        model.velocities[name] = min(float(limit.get("velocity")), limits[name]["max_velocity"])
        model.accelerations[name] = limits[name]["max_acceleration"]
        if not all(math.isfinite(v) and v > 0 for v in (model.velocities[name], model.accelerations[name])):
            raise ValueError(f"关节 {name} 的规划速度或加速度配置不合法")
    for state in srdf.findall(f"group_state[@group='{group}']"):
        values = {j.get("name"): float(j.get("value")) for j in state}
        model.check(values)
        model.named[state.get("name")] = values
    return model


def values_from_state(message, model):
    if len(message.name) != len(message.position) or len(set(message.name)) != len(message.name):
        raise ValueError("关节反馈的名称和位置不匹配，或存在重复关节")
    values = dict(zip(message.name, message.position))
    if not all(name in values for name in model.joints):
        raise ValueError("关节反馈缺少机械臂关节")
    result = {name: values[name] for name in model.joints}
    model.check(result)
    return result


def validate_trajectory(trajectory, model, start, goal, velocity, acceleration):
    if trajectory.multi_dof_joint_trajectory.points or trajectory.multi_dof_joint_trajectory.joint_names:
        raise ValueError("拒绝包含移动底盘或其他多自由度关节的轨迹")
    joint = trajectory.joint_trajectory
    if joint.header.stamp.sec != 0 or joint.header.stamp.nanosec != 0:
        raise ValueError("只接受从执行时刻开始的相对时间轨迹")
    if len(joint.joint_names) != len(model.joints) or set(joint.joint_names) != set(model.joints):
        raise ValueError("轨迹关节必须且只能包含当前机械臂的全部关节")
    if not 2 <= len(joint.points) <= 1000:
        raise ValueError("轨迹点数量必须在 2 至 1000 之间")
    previous, previous_positions = -1.0, None
    for point in joint.points:
        if any(len(v) != len(model.joints) for v in (point.positions, point.velocities, point.accelerations)):
            raise ValueError("轨迹必须包含完整的位置、速度和加速度")
        if point.effort:
            raise ValueError("当前入口不接受力矩轨迹")
        values = dict(zip(joint.joint_names, point.positions))
        model.check(values)
        seconds = point.time_from_start.sec + point.time_from_start.nanosec / 1e9
        if (point.time_from_start.sec < 0 or not 0 <= point.time_from_start.nanosec < 10**9
                or not math.isfinite(seconds) or seconds < 0 or seconds <= previous):
            raise ValueError("轨迹时间必须非负且严格递增")
        if previous_positions is not None:
            for name, value in values.items():
                if abs(value - previous_positions[name]) > model.velocities[name] * velocity * (seconds - previous) + 1e-3:
                    raise ValueError("相邻轨迹点的位置变化超过速度限制")
        previous, previous_positions = seconds, values
        for name, speed, rate in zip(joint.joint_names, point.velocities, point.accelerations):
            if (not math.isfinite(speed) or not math.isfinite(rate)
                    or abs(speed) > model.velocities[name] * velocity + 1e-3
                    or abs(rate) > model.accelerations[name] * acceleration + 1e-3):
                raise ValueError(f"轨迹关节 {name} 的速度或加速度不合法")
    for point, reference, tolerance in ((joint.points[0], start, model.start_tolerance),
                                        (joint.points[-1], goal, model.goal_tolerance)):
        if any(abs(value - reference[name]) > tolerance for name, value in zip(joint.joint_names, point.positions)):
            raise ValueError("轨迹起点或终点与请求不一致")
    return previous
