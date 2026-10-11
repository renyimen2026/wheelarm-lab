"""Preview and explicitly confirm local LLM-generated navigation missions."""

import argparse
import json
import sys

from wheel_arm_navigation.goals import load
from wheel_arm_navigation.llm import parse_instruction


def parser():
    result = argparse.ArgumentParser(description="本地 LLM 自然语言导航，执行前需确认")
    result.add_argument("instruction", help="自然语言指令，请用引号包围")
    result.add_argument("--file", default="maps/semantic_goals.yaml")
    result.add_argument("--model", default="qwen3:4b")
    result.add_argument("--endpoint", default="http://127.0.0.1:11434")
    result.add_argument("--llm-timeout", type=float, default=180)
    result.add_argument("--log-dir", default="mission_logs")
    result.add_argument("--preview", action="store_true", help="仅解析预览，不启动 ROS 导航")
    return result


def run(options, argv):
    document = load(options.file)
    print(f"已记录目标：{'、'.join(document['goals'])}")
    print(f"正在请求本地模型 {options.model}，此时不会发送导航目标……", flush=True)
    result, plan, elapsed = parse_instruction(
        options.instruction, document, options.model, options.endpoint, options.llm_timeout)
    print(f"解析耗时：{elapsed:g} 秒；状态：{result['status']}")
    print(result["message"])
    if plan is None:
        print("未启动导航，请修改指令后重试。")
        return 1
    print(json.dumps(plan, ensure_ascii=False, indent=2))
    for index, name in enumerate(plan["goals"], 1):
        pose = document["goals"][name]
        print(f"{index}. {name}: x={pose['x']:.3f}, y={pose['y']:.3f}, yaw={pose['yaw']:.3f}")
    if options.preview:
        print("仅预览，未启动导航。")
        return 0
    if not sys.stdin.isatty():
        raise ValueError("执行需要交互终端确认；非交互环境请使用 --preview")
    print("请核对路线、坐标、停留和循环次数，并确认定位正常且已退出键盘遥控。")
    if input("执行这项任务？[y/N] ").strip().lower() != "y":
        print("已取消，未启动导航。")
        return 0
    # Import the ROS executor only after validation and human confirmation.
    from wheel_arm_navigation.mission_runner import run_plan
    source = {"kind": "local_llm", "instruction": options.instruction,
              "model": options.model, "endpoint": options.endpoint,
              "parse_sec": elapsed, "response": result, "confirmed": True}
    return run_plan(plan, document, options.file, options.log_dir, argv, source=source)


def main(args=None):
    from rclpy.utilities import remove_ros_args

    argv = sys.argv if args is None else ["language_goal", *args]
    options = parser().parse_args(remove_ros_args(argv)[1:])
    try:
        return run(options, argv)
    except (OSError, ValueError, RuntimeError) as error:
        print(f"错误：{error}", file=sys.stderr)
        return 1
    except (KeyboardInterrupt, EOFError):
        print("已中断，未继续执行任务。", file=sys.stderr)
        return 130
