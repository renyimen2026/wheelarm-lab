"""CLI for sequential named-goal navigation missions."""

import argparse
import signal
import sys

import rclpy
from rclpy.signals import SignalHandlerOptions
from rclpy.utilities import remove_ros_args

from wheel_arm_navigation.goals import load
from wheel_arm_navigation.mission import check_goals, execute, load_task, validate_task
from wheel_arm_navigation.semantic_goal import GoalClient


def parser():
    result = argparse.ArgumentParser(description="顺序执行地图命名目标点任务")
    result.add_argument("goals", nargs="*", help="目标名称，按输入顺序执行，可重复")
    result.add_argument("--task", help="任务 YAML 文件，不能与位置参数混用")
    result.add_argument("--file", default="maps/semantic_goals.yaml")
    result.add_argument("--log-dir", default="mission_logs")
    result.add_argument("--name")
    result.add_argument("--stay", type=float)
    result.add_argument("--timeout", type=float)
    result.add_argument("--wait", type=float)
    result.add_argument("--repeat", type=int)
    result.add_argument("--on-failure", choices=("stop", "skip"))
    return result


def main(args=None):
    argv = sys.argv if args is None else ["mission_runner", *args]
    options = parser().parse_args(remove_ros_args(argv)[1:])
    node = None
    handlers = {}
    try:
        if options.task and options.goals:
            raise ValueError("--task 与命令行目标列表不能混用")
        task = load_task(options.task) if options.task else {"goals": options.goals}
        for key in ("name", "stay", "timeout", "wait", "repeat", "on_failure"):
            value = getattr(options, key)
            if value is not None:
                task[key] = value
        plan = validate_task(task)
        document = load(options.file)
        check_goals(plan, document)
        rclpy.init(args=argv, signal_handler_options=SignalHandlerOptions.NO)
        node = GoalClient("mission_runner")
        for signum in (signal.SIGINT, signal.SIGTERM):
            handlers[signum] = signal.signal(
                signum, lambda *_: setattr(node, "stopping", True))
        code, _ = execute(plan, document, options.file, options.log_dir,
                          node.navigate_pose, lambda: node.stopping, node.pause,
                          node.get_logger().info)
        return code
    except (OSError, ValueError, RuntimeError) as error:
        print(f"错误：{error}", file=sys.stderr)
        return 1
    finally:
        for signum, handler in handlers.items():
            signal.signal(signum, handler)
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
