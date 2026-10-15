"""Confirmed local language task: go to one named place, then observe once."""

import argparse
import json
import signal
import sys

from wheel_arm_navigation.goals import load
from wheel_arm_navigation import observe_llm, vision
from wheel_arm_navigation.scene_snapshot import positive


def parser():
    result = argparse.ArgumentParser(description="到一个已记录地点，停稳后拍照回答；执行前确认")
    result.add_argument("instruction")
    result.add_argument("--file", default="maps/semantic_goals.yaml")
    result.add_argument("--model", default="qwen3:4b")
    result.add_argument("--vision-model", default="qwen3-vl:4b-instruct")
    result.add_argument("--endpoint", default="http://127.0.0.1:11434")
    result.add_argument("--llm-timeout", type=float, default=180)
    result.add_argument("--vision-timeout", type=float, default=600)
    result.add_argument("--nav-wait", type=positive, default=10)
    result.add_argument("--nav-timeout", type=positive, default=180)
    result.add_argument("--capture-wait", type=positive, default=15)
    result.add_argument("--settle-duration", type=positive, default=0.5)
    result.add_argument("--max-age", type=positive, default=0.5)
    result.add_argument("--rgb-topic", default="/camera/image")
    result.add_argument("--odom-topic", default="/odometry/filtered")
    result.add_argument("--log-dir", default="mission_logs")
    result.add_argument("--observations-dir", default="observations")
    result.add_argument("--preview", action="store_true")
    result.add_argument("--no-vision-preload", action="store_true",
                        help="Disable optional background vision loading during navigation")
    return result


def run_execution(task, document, options, argv, source):
    import rclpy
    from rclpy.signals import SignalHandlerOptions
    from wheel_arm_navigation.arrival_capture import ObserveClient
    from wheel_arm_navigation.observe_task import execute
    from wheel_arm_navigation.vision_preload import VisionPreload

    node, preload, handlers = None, None, {}
    try:
        rclpy.init(args=[argv[0], "--ros-args", "-p", "use_sim_time:=true", "--", *argv[1:]],
                   signal_handler_options=SignalHandlerOptions.NO)
        node = ObserveClient(options)
        preload = VisionPreload(options.vision_model, options.endpoint, min(options.vision_timeout, 120),
                                node.get_logger().info, lambda: node.stopping)
        node.vision_preload = preload

        def interrupted(*_args):
            node.stopping = True
            if node.phase in ("vision", "preflight"):
                raise KeyboardInterrupt()

        for signum in (signal.SIGINT, signal.SIGTERM):
            handlers[signum] = signal.signal(signum, interrupted)

        def infer(image, question):
            node.phase = "vision"
            import time
            started = time.monotonic()
            preload.wait(options.vision_timeout)
            remaining = options.vision_timeout - (time.monotonic() - started)
            if remaining <= 0:
                raise RuntimeError("等待预加载耗尽视觉推理时间预算；已保留拍摄图片")
            return vision.describe(image, question, options.vision_model, options.endpoint, remaining)

        def preflight():
            node.phase = "preflight"
            try:
                info = observe_llm.check_vision_model(options.vision_model, options.endpoint, min(options.vision_timeout, 30))
                preload.enable(info.get("preload_supported", False) and not options.no_vision_preload)
                return info
            finally:
                node.phase = "navigation"

        code, _ = execute(task, document, options.file, options, node.navigate_pose,
                          node.capture_at_goal, infer, preflight, lambda: node.stopping,
                          node.get_logger().info, source, preload_snapshot=preload.snapshot)
        return code
    finally:
        if preload is not None:
            preload.close()
        for signum, handler in handlers.items():
            signal.signal(signum, handler)
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def run(options, argv):
    if not 0 < options.vision_timeout <= 600:
        raise ValueError("视觉超时必须在 0 至 600 秒之间")
    if options.settle_duration >= options.capture_wait:
        raise ValueError("停稳时长必须小于采集等待超时")
    document = load(options.file)
    response, task, elapsed = observe_llm.parse_instruction(
        options.instruction, document, options.model, options.endpoint, options.llm_timeout)
    print(f"解析状态：{response['status']}；耗时：{elapsed:g} 秒；{response['message']}")
    if task is None:
        print("未启动导航，请明确地点和观察问题。")
        return 1
    print(json.dumps(task, ensure_ascii=False, indent=2))
    print("目标位姿：" + json.dumps(document["goals"][task["goal"]], ensure_ascii=False))
    if options.preview:
        print("仅预览，不导航、不拍摄、不调用视觉模型。")
        return 0
    if not sys.stdin.isatty():
        raise ValueError("执行需要交互终端确认；非交互环境请使用 --preview")
    print("请确认定位正常、目标朝向合适，并退出键盘遥控和其他导航任务。")
    if input("执行导航并单帧观察？[y/N] ").strip().lower() != "y":
        print("已取消，未启动任务。")
        return 0
    source = {"kind": "local_llm_observe", "instruction": options.instruction,
              "model": options.model, "vision_model": options.vision_model, "endpoint": options.endpoint,
              "parse_sec": elapsed, "response": response, "confirmed": True}
    return run_execution(task, document, options, argv, source)


def main(args=None):
    from rclpy.utilities import remove_ros_args
    argv = sys.argv if args is None else ["language_observe", *args]
    options = parser().parse_args(remove_ros_args(argv)[1:])
    try:
        return run(options, argv)
    except (ValueError, RuntimeError, OSError) as error:
        print(f"导航观察失败：{error}", file=sys.stderr)
        return 1
    except (KeyboardInterrupt, EOFError):
        print("任务已中断，请确认机器人停止。", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
