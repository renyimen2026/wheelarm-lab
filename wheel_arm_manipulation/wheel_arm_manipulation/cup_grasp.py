"""Confirmed RGB-D grasp/lift/place demo for one upright handleless ceramic cup."""

import argparse
import copy
from dataclasses import replace
import json
import math
import signal
import sys
import time

import numpy as np
import rclpy
from moveit_msgs.msg import DisplayTrajectory, RobotTrajectory
from builtin_interfaces.msg import Duration
from trajectory_msgs.msg import JointTrajectoryPoint
from rclpy.signals import SignalHandlerOptions
from rclpy.utilities import remove_ros_args
from scipy.spatial.transform import Rotation
from scipy.interpolate import CubicSpline

from wheel_arm_manipulation.arm_goal import ArmGoalClient, positive, scaling
from wheel_arm_manipulation.cup_perception import CupObserver
from wheel_arm_manipulation.grasp_kinematics import GraspKinematics
from wheel_arm_manipulation.grasp_scene import GraspScene
from wheel_arm_manipulation.planning import load_model, validate_trajectory, values_from_state


def parser():
    result = argparse.ArgumentParser(description="RGB-D 桌面直立无柄杯抓取；默认仅预览，执行需人工确认")
    mode = result.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--detect-only", action="store_true")
    result.add_argument("--wait", type=positive, default=15.)
    result.add_argument("--timeout", type=positive, default=120.)
    result.add_argument("--velocity", type=scaling, default=.2)
    result.add_argument("--acceleration", type=scaling, default=.2)
    result.add_argument("--lift-height", type=positive, default=.08)
    result.add_argument("--rgb-topic", default="/camera/image")
    result.add_argument("--depth-topic", default="/camera/depth_image")
    result.add_argument("--info-topic", default="/camera/camera_info")
    result.add_argument("--camera-frame", default="camera_optical_frame")
    result.add_argument("--output-dir", help="可选：保存每次采集的首帧 RGB-D、内参和 TF，用于复核")
    result.add_argument("--joint-states-topic", default="/wheel_arm/joint_states")
    result.add_argument("--odom-topic", default="/odometry/filtered")
    return result


def unchanged(reference, current):
    if (np.linalg.norm(reference.center-current.center) > .004
            or abs(reference.radius-current.radius) > .0015
            or abs(reference.top-current.top) > .005
            or np.max(np.abs(reference.table_bounds[:, 2]-current.table_bounds[:, 2])) > .005):
        raise RuntimeError("杯子位置、尺寸或桌面已变化，原抓取计划作废；请重新识别规划")


def verify_lift(reference, lifted, height):
    delta = lifted.top-reference.top
    if (np.linalg.norm(reference.center[:2]-lifted.center[:2]) > .025
            or abs(reference.radius-lifted.radius) > .002
            or delta < .04 or abs(delta-height) > .018):
        raise RuntimeError("未通过 RGB-D 杯口升高验证；不能报告抓取成功，不会自动松爪")
    return float(delta)


def straight_trajectory(solver, model, start, target, velocity, acceleration):
    """Sample horizontal Cartesian IK, then conservatively time a clamped spline."""
    origin = solver.forward(start)[:3, 3]
    count = max(3, int(np.ceil(np.linalg.norm(target-origin)/.004))+1)
    steps = np.linspace(0, 1, count)
    values, previous = [[start[n] for n in model.joints]], start
    for fraction in steps[1:]:
        previous = solver.solve(origin+fraction*(target-origin), previous)
        values.append([previous[n] for n in model.joints])
    spline = CubicSpline(steps, values, bc_type="clamped")
    dense = np.linspace(0, 1, 1001)
    vmax = np.array([model.velocities[n]*velocity for n in model.joints])
    amax = np.array([model.accelerations[n]*acceleration for n in model.joints])
    duration = max(1., float(np.max(np.abs(spline(dense, 1))/vmax)),
                   float(np.sqrt(np.max(np.abs(spline(dense, 2))/amax))))*1.2
    trajectory = RobotTrajectory()
    trajectory.joint_trajectory.joint_names = list(model.joints)
    for fraction in np.linspace(0, 1, int(np.ceil(duration/.05))+1):
        ns = round(fraction*duration*1e9)
        trajectory.joint_trajectory.points.append(JointTrajectoryPoint(
            positions=spline(fraction).tolist(), velocities=(spline(fraction, 1)/duration).tolist(),
            accelerations=(spline(fraction, 2)/duration**2).tolist(),
            time_from_start=Duration(sec=ns//10**9, nanosec=ns%10**9)))
    validate_trajectory(trajectory, model, start, previous, velocity, acceleration)
    return trajectory, start, previous


class CupGraspClient(ArmGoalClient):
    def __init__(self, options):
        options.command = "vision"
        self.arm = replace(load_model(), tip="gripper_tcp")
        self.gripper = load_model("gripper")
        super().__init__(options, self.arm)
        self.stationary_models = (self.arm, self.gripper)
        self.attachment = None
        self.closed = False
        self.base_anchor = None
        self.kinematics = GraspKinematics(self.arm)
        self.observer = CupObserver(self, options)
        self.motion_guard = self.guard_base

    def state(self, message, values=None):
        state = super().state(message, values)
        if self.attachment is not None:
            state.attached_collision_objects = [copy.deepcopy(self.attachment)]
        return state

    def guard_base(self):
        if not self.fresh(self.odom, self.odom_received, 0, 0):
            raise RuntimeError("运动中里程计过期，请检查仿真；请求取消")
        for vector, bound in ((self.odom.twist.twist.linear, .02), (self.odom.twist.twist.angular, .04)):
            values = (vector.x, vector.y, vector.z)
            if not all(math.isfinite(v) for v in values) or math.hypot(*values) > bound:
                raise RuntimeError("运动中检测到底盘移动；请求取消")
        if self.base_anchor is not None:
            actual, anchor = self.odom.pose.pose, self.base_anchor
            a, b = actual.position, anchor.position
            q, r = actual.orientation, anchor.orientation
            rotation = Rotation.from_quat([r.x, r.y, r.z, r.w]).inv()*Rotation.from_quat([q.x, q.y, q.z, q.w])
            if math.dist((a.x, a.y, a.z), (b.x, b.y, b.z)) > .003 or rotation.magnitude() > .01:
                raise RuntimeError("底盘相对本次识别位姿已变化，抓取计划作废；请停稳后重新识别")

    def check_fk(self, message, target):
        pose = self.fk(self.state(message)).pose
        point, q = pose.position, pose.orientation
        actual = np.array([point.x, point.y, point.z])
        expected = self.kinematics.forward(values_from_state(message, self.arm))
        if (np.linalg.norm(actual-target) > .012 or np.linalg.norm(expected[:3, 3]-actual) > .002
                or np.linalg.norm(Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()-expected[:3, :3]) > .025):
            raise RuntimeError("MoveIt FK、URDF 运动学或实际 TCP 位置不一致；停止")

    def check_path(self, trajectory, start, target):
        direction = target-start
        length = np.linalg.norm(direction)
        for point in trajectory.joint_trajectory.points:
            tf = self.kinematics.forward(dict(zip(trajectory.joint_trajectory.joint_names, point.positions)))
            p = tf[:3, 3]
            t = 0. if length < 1e-8 else np.clip((p-start) @ direction/(length*length), 0, 1)
            if (np.linalg.norm(p-(start+t*direction)) > .012 or p[2] < min(start[2], target[2])-.002
                    or tf[2, 0]**2 + tf[2, 1]**2 > .0012 or tf[2, 2] < .999):
                raise RuntimeError("接近/抬升轨迹偏离直线或夹爪偏离水平；拒绝执行，请重新规划")

    def make_plan(self, cup, scene):
        if not .05 <= self.options.lift_height <= .12 or cup.height < .065:
            raise ValueError("当前安全范围要求杯高至少 65 mm，抬升高度为 50 至 120 mm")
        center = cup.center.copy()
        direction = center[:2]-self.kinematics.axis_origin[:2]
        direction /= np.linalg.norm(direction)
        pre = center.copy()
        pre[:2] -= .08*direction
        lifted = center + np.array([0, 0, self.options.lift_height])
        width = 2*cup.radius-.001
        if not .02 < width < .074:
            raise ValueError("视觉估计开口不适合当前夹爪")
        self.model = self.arm
        message = self.snapshot()
        initial = copy.deepcopy(message)
        plans = []
        stages = [("open", self.gripper, None, .08), ("pregrasp", self.arm, pre, None),
                  ("approach", self.arm, center, None), ("close", self.gripper, None, width),
                  ("lift", self.arm, lifted, None), ("lower", self.arm, center, None),
                  ("release", self.gripper, None, .08), ("retreat", self.arm, pre, None)]
        try:
            for label, model, target, opening in stages:
                self.model = model
                if label == "lift":
                    scene.attach(self.kinematics.forward(values_from_state(message, self.arm)))
                elif label == "release":
                    scene.detach()
                elif label == "approach":
                    scene.finger_contacts(True)
                if model.group == "arm":
                    goal = self.kinematics.solve(target, values_from_state(message, self.arm))
                else:
                    goal = {n: opening/2 for n in model.joints}
                print(f"[阶段预览] {label}", flush=True)
                full_start = copy.deepcopy(message)
                current_values = values_from_state(message, model)
                if all(abs(current_values[n]-goal[n]) <= model.goal_tolerance for n in model.joints):
                    self.valid_state(self.state(message))
                    print(f"[跳过] {label} 已到达目标，无需重复运动。", flush=True)
                    plans.append((label, model, None, current_values, current_values, full_start, target))
                    continue
                if target is not None and label != "pregrasp":
                    trajectory, start, goal = straight_trajectory(self.kinematics, model,
                        values_from_state(message, model), target,
                        self.options.velocity, self.options.acceleration)
                    for point in trajectory.joint_trajectory.points:
                        self.valid_state(self.state(message, dict(zip(model.joints, point.positions))))
                else:
                    trajectory, start, goal = self.plan(message, goal_values=goal)
                message = self.state(message, goal).joint_state
                if target is not None:
                    self.check_fk(message, target)
                    if label != "pregrasp":
                        self.check_path(trajectory, self.kinematics.forward(values_from_state(full_start, self.arm))[:3, 3], target)
                plans.append((label, model, trajectory, start, goal, full_start, target))
        finally:
            self.model = self.arm
            if self.attachment is not None:
                scene.detach()
            if not self.stopping:
                scene.finger_contacts(False)
        self.display.publish(DisplayTrajectory(model_id="wheel_arm", trajectory_start=self.state(initial),
                                                trajectory=[plan[2] for plan in plans if plan[2] is not None]))
        return plans

    def perform(self, plans, cup, scene):
        for label, model, trajectory, start, goal, expected, target in plans:
            self.model = model
            current = self.snapshot()
            self.guard_base()
            for group in (self.arm, self.gripper):
                a, b = values_from_state(current, group), values_from_state(expected, group)
                if any(abs(a[n]-b[n]) > group.start_tolerance for n in group.joints):
                    raise RuntimeError("完整关节状态偏离预览，原任务作废")
            if label in ("approach", "close"):
                print("[复查] 重新采集 RGB-D，确认杯子未移动...", flush=True)
                unchanged(cup, self.observer.capture())
            if label == "lift":
                self.model = self.arm
                scene.attach(self.kinematics.forward(values_from_state(current, self.arm)))
            elif label == "release":
                scene.detach()
            elif label == "approach":
                scene.finger_contacts(True)
            print(f"[阶段执行] {label}", flush=True)
            if label == "close":
                self.closed = True  # A failed close can still leave an object between the fingers.
            if trajectory is not None:
                self.execute(trajectory, start, goal)
            if label == "release":
                self.closed = False
            if target is not None:
                self.check_fk(self.joints, target)
            if label == "lift":
                print("[验证] 使用 RGB-D 检查杯子是否随夹爪升高...", flush=True)
                raised = self.observer.capture(require_support=False)
                delta = verify_lift(cup, raised, self.options.lift_height)
                print(f"[视觉确认] 杯口升高 {delta*1000:.1f} mm；非力传感器确认", flush=True)
                end = time.monotonic()+1
                while time.monotonic() < end:
                    rclpy.spin_once(self, timeout_sec=.02)
            if label == "retreat":
                placed = self.observer.capture()
                if np.linalg.norm(placed.center-cup.center) > .015:
                    raise RuntimeError("放下后位置偏离原目标超过 15 mm，请检查杯子")
        print("[完成] RGB-D 定位、抓取、视觉抬升验证、放回和退离均完成。", flush=True)


def run(node, options):
    if options.execute and not sys.stdin.isatty():
        raise ValueError("执行必须在交互终端确认，不接受管道输入")
    print("[检查] 请停止导航、遥控和其他机械臂任务；当前仅针对桌面直立无柄陶瓷杯。", flush=True)
    node.snapshot()
    node.base_anchor = copy.deepcopy(node.odom.pose.pose)
    print("[识别] 同步 RGB-D，拟合桌面和杯身，等待三帧稳定结果...", flush=True)
    cup = node.observer.capture()
    print("[测量] " + json.dumps({"frame": node.model.frame, "center": cup.center.tolist(),
          "diameter": 2*cup.radius, "height": cup.height, "fit_rmse": cup.residual}), flush=True)
    print("[限制] 几何识别不是通用语义识别；相似圆柱容器可能混淆。仅检查 MoveIt 已知碰撞物。", flush=True)
    if options.detect_only:
        for _ in range(25):
            rclpy.spin_once(node, timeout_sec=.02)
        return 0
    scene = GraspScene(node, cup)
    try:
        scene.install()
        plans = node.make_plan(cup, scene)
        if not options.execute:
            print("[预览] 已规划完整流程，未发送运动目标。增加 --execute 后仍需确认。", flush=True)
            return 0
        if input("确认测量、RViz 轨迹和环境安全，执行抓取并放回？[y/N] ").strip().lower() != "y":
            print("[取消] 未执行。", flush=True)
            return 0
        node.snapshot()
        unchanged(cup, node.observer.capture())
        node.perform(plans, cup, scene)
        return 0
    finally:
        if node.closed:
            print("[安全] 夹持状态未解除，未自动松爪；保留规划碰撞对象。请人工检查，重启测试场景后再试。", flush=True)
        else:
            node.model = node.arm
            scene.cleanup()


def main(args=None):
    argv = sys.argv if args is None else ["cup_grasp", *args]
    options = parser().parse_args(remove_ros_args(argv)[1:])
    if not .05 <= options.lift_height <= .12:
        parser().error("--lift-height 必须在 0.05 至 0.12 米之间")
    node, handlers = None, {}
    try:
        rclpy.init(args=[argv[0], "--ros-args", "-p", "use_sim_time:=true", "--", *argv[1:]],
                   signal_handler_options=SignalHandlerOptions.NO)
        node = CupGraspClient(options)

        def interrupted(*_):
            node.stopping = True
            if not node.execution_pending:
                raise KeyboardInterrupt()

        for signum in (signal.SIGINT, signal.SIGTERM):
            handlers[signum] = signal.signal(signum, interrupted)
        return run(node, options)
    except (KeyboardInterrupt, EOFError):
        print("[取消] 当前任务已中断。", flush=True)
        return 130
    except (RuntimeError, ValueError, OSError) as error:
        print(f"[失败] {error}", file=sys.stderr, flush=True)
        return 1
    finally:
        for signum, handler in handlers.items():
            signal.signal(signum, handler)
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
