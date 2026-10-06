"""Record and navigate to named map poses using the Nav2 action interface."""

import argparse
from dataclasses import dataclass
import math
import signal
import sys
import time

import rclpy
from action_msgs.msg import GoalStatus
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from rclpy.time import Time
from rclpy.utilities import remove_ros_args
from tf2_ros import Buffer, TransformException, TransformListener

from wheel_arm_navigation.goals import load, save_pose


def positive(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("必须是大于零的有限数值")
    return number


def parser():
    result = argparse.ArgumentParser(description="记录地图语义目标点并调用 Nav2 导航")
    result.add_argument("--file", default="maps/semantic_goals.yaml")
    commands = result.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="列出目标点，不需要启动 ROS")
    save = commands.add_parser("save", help="记录 map -> base_footprint 当前位姿")
    save.add_argument("name")
    save.add_argument("--overwrite", action="store_true")
    save.add_argument("--wait", type=positive, default=10.0)
    set_pose = commands.add_parser("set", help="手动添加地图坐标目标点")
    set_pose.add_argument("name")
    for key in ("x", "y", "yaw"):
        set_pose.add_argument("--" + key, type=float, required=True)
    set_pose.add_argument("--overwrite", action="store_true")
    go = commands.add_parser("go", help="按名称导航，Ctrl+C 取消")
    go.add_argument("name")
    go.add_argument("--wait", type=positive, default=10.0)
    go.add_argument("--timeout", type=positive, default=180.0)
    return result


@dataclass
class NavigationResult:
    status: str
    reason: str = ""
    action_status: int = None
    safe_to_continue: bool = True


class GoalClient(Node):
    def __init__(self, node_name="semantic_goal"):
        super().__init__(node_name)
        self.stopping = False
        self.last_feedback = 0.0

    def wait(self, future, seconds, ignore_stop=False):
        deadline = time.monotonic() + seconds
        while not future.done() and (ignore_stop or not self.stopping) and rclpy.ok():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            rclpy.spin_once(self, timeout_sec=min(0.1, remaining))
        return future.done()

    def record(self, args):
        buffer = Buffer()
        listener = TransformListener(buffer, self)
        deadline = time.monotonic() + args.wait
        while not self.stopping and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            try:
                transform = buffer.lookup_transform("map", "base_footprint", Time())
            except TransformException:
                continue
            # Reject stale TF after a localization publisher has stopped.
            now = self.get_clock().now().nanoseconds
            stamp = Time.from_msg(transform.header.stamp).nanoseconds
            if now == 0 or not 0 <= (now - stamp) / 1e9 <= 1.0:
                continue
            q = transform.transform.rotation
            yaw = math.atan2(2 * (q.w * q.z + q.x * q.y),
                             1 - 2 * (q.y * q.y + q.z * q.z))
            position = transform.transform.translation
            save_pose(args.file, args.name, position.x, position.y, yaw, args.overwrite)
            self.get_logger().info(
                f"已记录 {args.name}: x={position.x:.3f}, y={position.y:.3f}, yaw={yaw:.3f}")
            return 0
        raise RuntimeError("未收到有效 map -> base_footprint TF；请启动定位并设置初始位姿")

    def feedback(self, message):
        if time.monotonic() - self.last_feedback >= 2.0:
            self.get_logger().info(f"剩余距离 {message.feedback.distance_remaining:.2f} m")
            self.last_feedback = time.monotonic()

    def cancel(self, handle, result_future=None):
        # A cancel acknowledgement alone does not mean the goal has terminated.
        result_future = result_future or handle.get_result_async()
        future = handle.cancel_goal_async()
        self.wait(future, 3.0, ignore_stop=True)
        if future.done() and future.result().goals_canceling:
            self.get_logger().warn("Nav2 已接受取消请求")
        if self.wait(result_future, 3.0, ignore_stop=True):
            if result_future.result().status in (
                    GoalStatus.STATUS_SUCCEEDED, GoalStatus.STATUS_ABORTED,
                    GoalStatus.STATUS_CANCELED):
                self.get_logger().info("导航目标已结束")
                return True
        self.get_logger().error("目标结束未确认，请检查 Nav2 状态并确保机器人停止")
        return False

    def navigate(self, args):
        document = load(args.file)
        if args.name not in document["goals"]:
            raise ValueError(f"目标点 {args.name} 不存在；先使用 list 查看名称")
        result = self.navigate_pose(args.name, document["goals"][args.name],
                                    document["frame_id"], args.wait, args.timeout)
        if result.status == "succeeded":
            return 0
        self.get_logger().error(f"导航未完成：{result.status}；{result.reason}")
        return 130 if self.stopping else 1

    def navigate_pose(self, name, pose, frame_id, wait, timeout):
        client = ActionClient(self, NavigateToPose, "navigate_to_pose")
        self.last_feedback = 0.0
        try:
            return self._navigate(client, name, pose, frame_id, wait, timeout)
        except Exception as error:
            # Communication failures cannot establish whether an accepted goal is still active.
            return NavigationResult("error", str(error), safe_to_continue=False)
        finally:
            client.destroy()

    def _navigate(self, client, name, pose, frame_id, wait, timeout):
        deadline = time.monotonic() + wait
        while not client.server_is_ready():
            if self.stopping:
                return NavigationResult("canceled", "任务已中断，尚未发送目标")
            if time.monotonic() >= deadline:
                return NavigationResult("unavailable", "Nav2 navigate_to_pose 未就绪",
                                        safe_to_continue=False)
            rclpy.spin_once(self, timeout_sec=0.1)
        deadline = time.monotonic() + wait
        while self.get_clock().now().nanoseconds == 0:
            if self.stopping:
                return NavigationResult("canceled", "任务已中断，尚未发送目标")
            if time.monotonic() >= deadline:
                return NavigationResult("unavailable", "未收到 /clock",
                                        safe_to_continue=False)
            rclpy.spin_once(self, timeout_sec=0.1)
        if self.stopping:
            return NavigationResult("canceled", "任务已中断，尚未发送目标")
        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id = frame_id
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x = float(pose["x"])
        goal.pose.pose.position.y = float(pose["y"])
        goal.pose.pose.orientation.z = math.sin(pose["yaw"] / 2)
        goal.pose.pose.orientation.w = math.cos(pose["yaw"] / 2)
        future = client.send_goal_async(goal, feedback_callback=self.feedback)
        if not self.wait(future, wait):
            self.wait(future, 3.0, ignore_stop=True)
            confirmed = future.done()
            if confirmed and future.result().accepted:
                confirmed = self.cancel(future.result())
            return NavigationResult(
                "canceled" if self.stopping else "acceptance_timeout",
                "目标应答中断或超时" if confirmed else "目标状态未知，请确认机器人停止",
                safe_to_continue=confirmed)
        handle = future.result()
        if not handle.accepted:
            return NavigationResult("rejected", "Nav2 拒绝目标点")
        self.get_logger().info(f"开始导航到 {name}")
        result_future = handle.get_result_async()
        if not self.wait(result_future, timeout) or self.stopping:
            confirmed = self.cancel(handle, result_future)
            return NavigationResult(
                "canceled" if self.stopping else "timeout",
                "目标已结束" if confirmed else "目标结束未确认，请确认机器人停止",
                safe_to_continue=confirmed)
        status = result_future.result().status
        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info(f"已到达 {name}")
            return NavigationResult("succeeded", action_status=status)
        if status == GoalStatus.STATUS_CANCELED:
            return NavigationResult("canceled", "Nav2 返回取消状态", status)
        if status == GoalStatus.STATUS_ABORTED:
            return NavigationResult("aborted", "Nav2 导航失败", status)
        return NavigationResult("error", "Nav2 返回非终态结果", status, False)

    def pause(self, seconds):
        deadline = time.monotonic() + seconds
        while not self.stopping and rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=max(0.0, min(0.1, deadline - time.monotonic())))
        return not self.stopping and rclpy.ok()


def main(args=None):
    argv = sys.argv if args is None else ["semantic_goal", *args]
    options = parser().parse_args(remove_ros_args(argv)[1:])
    node = None
    previous_handlers = {}
    try:
        if options.command == "list":
            for name, pose in load(options.file)["goals"].items():
                print(f"{name}: x={pose['x']:.3f}, y={pose['y']:.3f}, yaw={pose['yaw']:.3f}")
            return 0
        if options.command == "set":
            save_pose(options.file, options.name, options.x, options.y,
                      options.yaw, options.overwrite)
            print(f"已保存 {options.name}")
            return 0
        rclpy.init(args=argv, signal_handler_options=SignalHandlerOptions.NO)
        node = GoalClient()
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[signum] = signal.signal(
                signum, lambda *_: setattr(node, "stopping", True))
        return node.record(options) if options.command == "save" else node.navigate(options)
    except (OSError, ValueError, RuntimeError) as error:
        print(f"错误：{error}", file=sys.stderr)
        return 1
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
