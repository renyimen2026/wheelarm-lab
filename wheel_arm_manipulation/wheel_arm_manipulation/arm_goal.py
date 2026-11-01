"""Position-only MoveIt commands; preview by default, confirm every execution."""

import argparse
import copy
import math
import signal
import sys
import time
from uuid import uuid4

import rclpy
from action_msgs.msg import GoalStatus
from action_msgs.srv import CancelGoal
from moveit_msgs.action import ExecuteTrajectory
from moveit_msgs.msg import Constraints, DisplayTrajectory, JointConstraint, MoveItErrorCodes, RobotState
from moveit_msgs.srv import GetMotionPlan, GetPositionFK, GetPositionIK, GetStateValidity
from nav_msgs.msg import Odometry
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
from rclpy.signals import SignalHandlerOptions
from rclpy.utilities import remove_ros_args
from sensor_msgs.msg import JointState
from unique_identifier_msgs.msg import UUID

from wheel_arm_manipulation.planning import load_model, validate_trajectory, values_from_state


def finite(value):
    result = float(value)
    if not math.isfinite(result):
        raise argparse.ArgumentTypeError("必须是有限数值")
    return result


def positive(value):
    result = finite(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("必须大于零")
    return result


def scaling(value):
    result = positive(value)
    if result > 1:
        raise argparse.ArgumentTypeError("缩放必须在 0 至 1 之间")
    return result


def opening(value):
    result = finite(value)
    if not 0 <= result <= 0.08:
        raise argparse.ArgumentTypeError("开口宽度必须在 0 至 0.08 米之间")
    return result


def parser():
    result = argparse.ArgumentParser(description="机械臂末端位置与命名姿态；默认只规划，--execute 后仍需确认")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--wait", type=positive, default=10.0)
    common.add_argument("--joint-states-topic", default="/wheel_arm/joint_states")
    common.add_argument("--odom-topic", default="/odometry/filtered")
    motion = argparse.ArgumentParser(add_help=False, parents=[common])
    motion.add_argument("--execute", action="store_true")
    motion.add_argument("--velocity", type=scaling, default=0.2)
    motion.add_argument("--acceleration", type=scaling, default=0.2)
    motion.add_argument("--timeout", type=positive, default=120.0)
    commands = result.add_subparsers(dest="command", required=True)
    commands.add_parser("state", parents=[common], help="只读取当前关节和末端位置")
    named = commands.add_parser("named", parents=[motion], help="规划 SRDF 中的命名姿态")
    named.add_argument("name")
    gripper = commands.add_parser("gripper", parents=[motion], help="双指夹爪位置控制，非力控；开口最大 8 cm")
    gripper.add_argument("name", choices=("open", "closed", "width"))
    gripper.add_argument("--width", type=opening, help="width 模式的内侧总开口，单位米")
    position = commands.add_parser("position", parents=[motion], help="规划 tool0 在 base_footprint 中的位置，单位米")
    for axis in ("x", "y", "z"):
        position.add_argument("--" + axis, type=finite, required=True)
    return result


class ArmGoalClient(Node):
    def __init__(self, options, model):
        super().__init__("arm_goal")
        self.options, self.model = options, model
        self.stopping = self.execution_pending = False
        self.joints = self.odom = None
        self.stationary_models = (load_model(), model) if model.group == "gripper" else (model,)
        self.joint_received = self.odom_received = 0.0
        self.create_subscription(JointState, options.joint_states_topic, self.read_joints, qos_profile_sensor_data)
        self.create_subscription(Odometry, options.odom_topic, self.read_odom, qos_profile_sensor_data)
        self.display = self.create_publisher(DisplayTrajectory, "/display_planned_path",
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))

    def read_joints(self, message):
        self.joints, self.joint_received = message, time.monotonic()

    def read_odom(self, message):
        self.odom, self.odom_received = message, time.monotonic()

    def wait(self, future, timeout, ignore_stop=False):
        deadline = time.monotonic() + timeout
        while rclpy.ok() and not future.done() and time.monotonic() < deadline:
            if self.stopping and not ignore_stop:
                raise KeyboardInterrupt()
            rclpy.spin_once(self, timeout_sec=0.02)
        return future.done()

    def call(self, kind, name, request, timeout=None, ignore_stop=False):
        timeout = self.options.wait if timeout is None else timeout
        deadline = time.monotonic() + timeout
        client = self.create_client(kind, name)
        try:
            while not client.service_is_ready() and time.monotonic() < deadline:
                if self.stopping and not ignore_stop:
                    raise KeyboardInterrupt()
                rclpy.spin_once(self, timeout_sec=0.02)
            if not client.service_is_ready():
                raise RuntimeError(f"服务未就绪：{name}；请启动 MoveIt")
            future = client.call_async(request)
            if not self.wait(future, max(0.0, deadline - time.monotonic()), ignore_stop):
                raise RuntimeError(f"服务请求超时：{name}")
            return future.result()
        finally:
            self.destroy_client(client)

    def fresh(self, message, received, after, ros_after):
        if message is None or received < after or time.monotonic() - received > 0.5:
            return False
        stamp = message.header.stamp.sec * 10**9 + message.header.stamp.nanosec
        now = self.get_clock().now().nanoseconds
        return now > 0 and stamp >= ros_after and 0 <= (now - stamp) / 1e9 <= 0.5

    def snapshot(self, settled=True, ignore_stop=False):
        after, ros_after = time.monotonic(), self.get_clock().now().nanoseconds
        deadline, quiet_wall, quiet_ros = after + self.options.wait, None, None
        while time.monotonic() < deadline and rclpy.ok():
            if self.stopping and not ignore_stop:
                raise KeyboardInterrupt()
            rclpy.spin_once(self, timeout_sec=0.02)
            valid = self.fresh(self.joints, self.joint_received, after, ros_after)
            if valid:
                values_from_state(self.joints, self.model)
            if valid and not settled:
                return copy.deepcopy(self.joints)
            if valid:
                for model in self.stationary_models:
                    values_from_state(self.joints, model)
                    indices = [self.joints.name.index(name) for name in model.joints]
                    maximum = 0.005 if model.group == "gripper" else 0.02
                    valid = valid and len(self.joints.velocity) == len(self.joints.name) and all(
                        math.isfinite(self.joints.velocity[i]) and abs(self.joints.velocity[i]) <= maximum for i in indices)
            valid = valid and self.fresh(self.odom, self.odom_received, after, ros_after)
            if valid:
                twist = self.odom.twist.twist
                valid = self.odom.child_frame_id == self.model.frame
                for vector, maximum in ((twist.linear, 0.02), (twist.angular, 0.04)):
                    components = (vector.x, vector.y, vector.z)
                    valid = valid and all(math.isfinite(v) for v in components) and math.hypot(*components) <= maximum
            now = self.get_clock().now().nanoseconds
            if not valid:
                quiet_wall = quiet_ros = None
            elif quiet_wall is None:
                quiet_wall, quiet_ros = time.monotonic(), now
            elif time.monotonic() - quiet_wall >= 0.3 and (now - quiet_ros) / 1e9 >= 0.3:
                return copy.deepcopy(self.joints)
        raise RuntimeError("未获得新鲜关节反馈或连续停稳状态；检查 /clock、关节速度、里程计话题和底盘")

    def state(self, message, values=None):
        result = RobotState(joint_state=copy.deepcopy(message))
        if values is not None:
            for name, value in values.items():
                result.joint_state.position[result.joint_state.name.index(name)] = value
        return result

    def fk(self, state):
        request = GetPositionFK.Request(robot_state=state, fk_link_names=[self.model.tip])
        request.header.frame_id = self.model.frame
        response = self.call(GetPositionFK, "/compute_fk", request)
        if response.error_code.val != MoveItErrorCodes.SUCCESS or len(response.pose_stamped) != 1:
            raise RuntimeError("末端正运动学计算失败")
        pose = response.pose_stamped[0]
        xyz = pose.pose.position
        if pose.header.frame_id != self.model.frame or not all(math.isfinite(v) for v in (xyz.x, xyz.y, xyz.z)):
            raise RuntimeError("末端正运动学返回的坐标系或位置不合法")
        return pose

    def valid_state(self, state, timeout=None):
        result = self.call(GetStateValidity, "/check_state_validity",
                           GetStateValidity.Request(robot_state=state, group_name=self.model.group), timeout)
        if not result.valid:
            contacts = ", ".join(f"{p.contact_body_1}/{p.contact_body_2}" for p in result.contacts[:5])
            raise RuntimeError("MoveIt 状态无效或碰撞：" + contacts)

    def plan(self, message, goal_values=None):
        print("[规划] 检查当前状态、求解目标并生成轨迹...", flush=True)
        start = values_from_state(message, self.model)
        current = self.state(message)
        self.valid_state(current)
        if goal_values is not None:
            goal = dict(goal_values)
        elif self.options.command in ("named", "gripper"):
            if self.options.command == "gripper" and self.options.name == "width":
                goal = {name: self.options.width / 2 for name in self.model.joints}
            elif self.options.name not in self.model.named:
                raise ValueError("未知姿态；可用：" + ", ".join(self.model.named))
            else:
                goal = self.model.named[self.options.name]
        else:
            pose = self.fk(current)
            pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = (
                self.options.x, self.options.y, self.options.z)
            request = GetPositionIK.Request()
            ik = request.ik_request
            ik.group_name, ik.ik_link_name = self.model.group, self.model.tip
            ik.robot_state, ik.pose_stamped, ik.avoid_collisions = current, pose, True
            ik.timeout.sec = 1
            result = self.call(GetPositionIK, "/compute_ik", request)
            if result.error_code.val != MoveItErrorCodes.SUCCESS:
                raise RuntimeError(f"目标没有找到无碰撞逆解，MoveIt 错误码 {result.error_code.val}；未执行")
            goal = values_from_state(result.solution.joint_state, self.model)
        self.model.check(goal)
        target = self.state(message, goal)
        self.valid_state(target)
        pose = self.fk(target)
        xyz = pose.pose.position
        orientation_note = "已指定关节目标" if goal_values is not None else "不约束朝向"
        print(f"[目标] {self.model.tip} @ {self.model.frame}: x={xyz.x:.3f}, y={xyz.y:.3f}, z={xyz.z:.3f} m；{orientation_note}", flush=True)
        if self.options.command == "position" and math.dist(
                (pose.pose.position.x, pose.pose.position.y, pose.pose.position.z),
                (self.options.x, self.options.y, self.options.z)) > 0.005:
            raise RuntimeError("逆解的实际末端位置偏离目标超过 5 mm；未执行")
        request = GetMotionPlan.Request()
        plan = request.motion_plan_request
        plan.group_name, plan.pipeline_id, plan.planner_id = self.model.group, "ompl", "RRTConnectkConfigDefault"
        plan.start_state = current
        plan.allowed_planning_time, plan.num_planning_attempts = 5.0, 3
        plan.max_velocity_scaling_factor = self.options.velocity
        plan.max_acceleration_scaling_factor = self.options.acceleration
        tolerance = 0.0001 if self.model.group == "gripper" else 0.005
        plan.goal_constraints = [Constraints(joint_constraints=[JointConstraint(joint_name=name, position=value,
            tolerance_above=tolerance, tolerance_below=tolerance, weight=1.0) for name, value in goal.items()])]
        response = self.call(GetMotionPlan, "/plan_kinematic_path", request).motion_plan_response
        if response.error_code.val != MoveItErrorCodes.SUCCESS:
            raise RuntimeError(f"规划失败，MoveIt 错误码 {response.error_code.val}；未执行")
        duration = validate_trajectory(response.trajectory, self.model, start, goal,
                                       self.options.velocity, self.options.acceleration)
        self.display.publish(DisplayTrajectory(model_id="wheel_arm", trajectory_start=current,
                                                trajectory=[response.trajectory]))
        print("[规划] 关节目标：" + ", ".join(f"{n}={v:.3f}" for n, v in goal.items()), flush=True)
        if self.model.group == "gripper":
            print(f"[夹爪] 目标内侧开口 {sum(goal.values()) * 1000:.1f} mm；仅位置控制，不代表抓取成功。", flush=True)
        print(f"[规划] {len(response.trajectory.joint_trajectory.points)} 个轨迹点；预计 {duration:.2f} 秒（ROS 时间）", flush=True)
        print("[注意] 仅检查 MoveIt 已知碰撞物；Gazebo 家具和地面不会自动导入。请停止导航和遥控。", flush=True)
        return response.trajectory, start, goal

    def cancel_and_confirm(self, client, handle, result_future, identity):
        try:
            if handle is not None:
                self.wait(handle.cancel_goal_async(), 3.0, ignore_stop=True)
            else:
                request = CancelGoal.Request()
                request.goal_info.goal_id = identity
                self.call(CancelGoal, "/execute_trajectory/_action/cancel_goal", request, 3.0, True)
            if result_future is not None:
                result = result_future.result() if result_future.done() else (
                    result_future.result() if self.wait(result_future, 5.0, True) else None)
            else:
                kind = ExecuteTrajectory.Impl.GetResultService
                result = self.call(kind, "/execute_trajectory/_action/get_result",
                                   kind.Request(goal_id=identity), 5.0, True)
            if result is not None and result.status in (
                    GoalStatus.STATUS_SUCCEEDED, GoalStatus.STATUS_CANCELED, GoalStatus.STATUS_ABORTED):
                self.snapshot(ignore_stop=True)
                self.execution_pending = False
                print("[取消] 已确认目标结束，反馈显示机械臂与底盘停稳。", flush=True)
                return True
        except Exception as error:
            self.get_logger().error(f"取消或停稳确认失败：{error}")
        print("[安全] 无法确认目标结束和停稳！请检查 MoveIt、控制器及仿真，不要继续发送任务。", flush=True)
        return False

    def execute(self, trajectory, start, goal):
        # Revalidate the same previewed path after confirmation, then refresh the start.
        message = self.snapshot()
        deadline = time.monotonic() + max(20.0, self.options.wait)
        for point in trajectory.joint_trajectory.points:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError("执行前轨迹碰撞复查超时；未执行")
            values = dict(zip(trajectory.joint_trajectory.joint_names, point.positions))
            self.valid_state(self.state(message, values), min(self.options.wait, remaining))
        message = self.snapshot()
        current = values_from_state(message, self.model)
        if any(abs(current[name] - start[name]) > self.model.start_tolerance for name in self.model.joints):
            raise RuntimeError("确认期间机械臂状态已变化，原轨迹作废；请重新规划")
        client = ActionClient(self, ExecuteTrajectory, "/execute_trajectory")
        handle = result_future = None
        identity = UUID(uuid=list(uuid4().bytes))
        try:
            if not client.wait_for_server(timeout_sec=self.options.wait):
                raise RuntimeError("MoveIt 执行 Action 未就绪")
            if self.stopping:
                raise KeyboardInterrupt()
            self.execution_pending = True
            future = client.send_goal_async(ExecuteTrajectory.Goal(trajectory=trajectory), goal_uuid=identity)
            if not self.wait(future, self.options.wait, True):
                raise RuntimeError("执行目标接受状态超时")
            handle = future.result()
            if not handle.accepted:
                self.execution_pending = False
                raise RuntimeError("MoveIt 拒绝执行目标")
            result_future = handle.get_result_async()
            print("[执行] 机械臂运动中，Ctrl+C 请求取消。", flush=True)
            deadline = time.monotonic() + self.options.timeout
            while not result_future.done() and time.monotonic() < deadline and not self.stopping:
                rclpy.spin_once(self, timeout_sec=0.02)
                guard = getattr(self, "motion_guard", None)
                if guard is not None:
                    guard()
            if self.stopping:
                raise KeyboardInterrupt()
            if not result_future.done():
                raise RuntimeError("机械臂执行超时")
            result = result_future.result()
            if result.status not in (GoalStatus.STATUS_SUCCEEDED, GoalStatus.STATUS_CANCELED, GoalStatus.STATUS_ABORTED):
                raise RuntimeError("MoveIt 未返回明确执行终态")
            message = self.snapshot(ignore_stop=True)
            self.execution_pending = False
            if self.stopping:
                raise KeyboardInterrupt()
            if result.status != GoalStatus.STATUS_SUCCEEDED or result.result.error_code.val != MoveItErrorCodes.SUCCESS:
                raise RuntimeError(f"执行失败，MoveIt 错误码 {result.result.error_code.val}")
        except (Exception, KeyboardInterrupt):
            if self.execution_pending and not self.cancel_and_confirm(client, handle, result_future, identity):
                raise RuntimeError("执行状态未知，必须人工检查机器人停止") from None
            raise
        finally:
            client.destroy()
        actual = values_from_state(message, self.model)
        if any(abs(actual[name] - goal[name]) > self.model.reached_tolerance for name in self.model.joints):
            raise RuntimeError("Action 已结束，但关节反馈尚未达到目标；不报告成功")
        pose = self.fk(self.state(message)).pose.position
        if self.options.command == "position" and math.dist(
                (pose.x, pose.y, pose.z), (self.options.x, self.options.y, self.options.z)) > 0.03:
            raise RuntimeError("Action 已结束，但末端反馈偏离目标超过 3 cm；不报告成功")
        print(f"[完成] {self.model.tip} @ {self.model.frame}: x={pose.x:.3f}, y={pose.y:.3f}, z={pose.z:.3f} m", flush=True)
        if self.model.group == "gripper":
            print(f"[夹爪] 反馈内侧开口 {sum(actual.values()) * 1000:.1f} mm；未判断物体接触或夹持力。", flush=True)


def run(node, options):
    if options.command == "state":
        message = node.snapshot(settled=False)
        pose = node.fk(node.state(message)).pose
        print("关节（rad）：" + str(values_from_state(message, node.model)), flush=True)
        print(f"{node.model.tip} @ {node.model.frame}: {pose}", flush=True)
        return 0
    if options.execute and not sys.stdin.isatty():
        raise ValueError("执行需要交互终端和人工确认，不接受管道输入")
    print("[检查] 等待新鲜反馈和机械臂、底盘停稳...", flush=True)
    trajectory, start, goal = node.plan(node.snapshot())
    if not options.execute:
        for _ in range(10):
            rclpy.spin_once(node, timeout_sec=0.02)
        print("[预览] 未执行。需要运动时增加 --execute，仍需输入 y 确认。", flush=True)
        return 0
    if input("确认 RViz 轨迹、周围空间和底盘停稳后执行？[y/N] ").strip().lower() != "y":
        print("[取消] 未发送执行目标。", flush=True)
        return 0
    node.execute(trajectory, start, goal)
    return 0


def main(args=None):
    argv = sys.argv if args is None else ["arm_goal", *args]
    options = parser().parse_args(remove_ros_args(argv)[1:])
    if options.command == "gripper" and ((options.name == "width") != (options.width is not None)):
        parser().error("gripper width 必须配合 --width；open/closed 不接受 --width")
    node, handlers = None, {}
    try:
        model = load_model("gripper" if options.command == "gripper" else "arm")
        if options.command == "named" and options.name not in model.named:
            raise ValueError("未知姿态；可用：" + ", ".join(model.named))
        rclpy.init(args=[argv[0], "--ros-args", "-p", "use_sim_time:=true", "--", *argv[1:]],
                   signal_handler_options=SignalHandlerOptions.NO)
        node = ArmGoalClient(options, model)

        def interrupted(*_args):
            node.stopping = True
            if not node.execution_pending:
                raise KeyboardInterrupt()

        for signum in (signal.SIGINT, signal.SIGTERM):
            handlers[signum] = signal.signal(signum, interrupted)
        return run(node, options)
    except (KeyboardInterrupt, EOFError):
        print("[取消] 当前命令已中断。", flush=True)
        return 130
    except (ValueError, RuntimeError, OSError) as error:
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
