"""Sequential terminal tasks with real stage feedback and per-task confirmation."""

import copy
import sys
from uuid import uuid4

from wheel_arm_navigation import language_observe


def execute_task(options, argv, **kwargs):
    from wheel_arm_navigation.chat_llm import parse_instruction
    return language_observe.run(options, argv, parse_instruction=parse_instruction, **kwargs)


class TaskFeedback:
    def __init__(self, output):
        self.output = output
        self.navigation_pending = False
        self.unsafe = False
        self.status = None

    def __call__(self, stage, details):
        goal = details.get("goal", "")
        if stage == "parsing":
            self.output("[任务] 任务解析中...")
        elif stage == "parsed":
            self.output(f"[任务] 指令解析完成。目标：{goal}；问题：{details['question']}")
            if details.get("action") == "pick_cup":
                self.output("[任务] 动作：视觉确认后抓起杯子并保持，不包含放回或递送。")
        elif stage == "navigation_requested":
            self.navigation_pending = True
        elif stage == "navigating":
            self.navigation_pending = True
            self.output(f"[任务] 前往目标地点：{goal}...")
        elif stage == "arrived":
            self.output(f"[任务] 已到达{goal}，正在检查停稳并采集画面...")
        elif stage == "recognizing":
            self.output("[任务] 画面已采集，场景识别中...")
        elif stage == "cup_checked":
            status = {"found": "发现杯子", "not_found": "未发现杯子", "uncertain": "无法确认杯子"}[details["status"]]
            self.output(f"[视觉] {status}；数量：{details['count']}；{details['evidence']}")
        elif stage == "grasping":
            self.output("[任务] 视觉已确认杯子，正在用 RGB-D 定位并规划抓取...")
        elif stage == "completed":
            self.output("[任务] 识别完成。")
            result = details["result"]
            self.output("[结果] " + result["summary"])
            for uncertainty in result["uncertainties"]:
                self.output("[不确定性] " + uncertainty)
        elif stage == "finished":
            self.status = details["status"]
            self.unsafe = self.status in ("unsafe_stop", "holding")
            self.navigation_pending = self.unsafe
            if self.status == "holding":
                self.output("[任务] 杯子已抓起，保持夹持。处理杯子前不再接收新任务。")
            elif self.status == "canceled":
                self.output("[任务] 当前任务已取消。")
            elif self.status != "completed":
                self.output(f"[任务] 未完成：{self.status}；{details.get('reason', '')}")


def parser():
    from ament_index_python.packages import get_package_share_directory
    from pathlib import Path
    result = language_observe.parser(include_instruction=False)
    result.description = "持续输入导航观察任务；每条任务独立解析并人工确认"
    result.set_defaults(endpoint="http://127.0.0.1:11435")
    share = Path(get_package_share_directory("wheel_arm_manipulation"))
    result.add_argument("--house-reference", default=str(share / "config/house_map_reference.yaml"))
    result.add_argument("--house-layout", default=str(share / "config/living_room.yaml"))
    return result


def run_session(options, argv, read=None, output=None, execute=None):
    read = input if read is None else read
    output = (lambda message: print(message, flush=True)) if output is None else output
    execute = execute_task if execute is None else execute
    if not sys.stdin.isatty():
        raise ValueError("robot_chat 需要交互终端，不接受管道或批量输入")
    session_id = uuid4().hex
    output("[系统] 机器人任务窗口已就绪。输入地点和观察问题，/help 查看帮助，/quit 退出。")
    output("[系统] 每条任务独立解析；支持单地点观察，或客厅固定抓取点的无柄杯抓取，不接收语音。")
    output(f"[系统] 模型接口：{options.endpoint}；执行前请退出遥控和其他导航任务。")
    while True:
        try:
            instruction = read("你：").strip()
        except EOFError:
            output("[系统] 会话结束。")
            return 0
        except KeyboardInterrupt:
            output("[系统] 会话结束。")
            return 130
        if not instruction:
            continue
        if instruction in ("/quit", "/exit"):
            output("[系统] 会话结束。")
            return 0
        if instruction == "/help":
            output("[系统] 示例：去厨房看看有没有杯子。解析后核对目标、坐标和问题，输入 y 执行。")
            output("[系统] 抓取示例：去客厅看看帮我拿下杯子。视觉未确认唯一杯子时不会抓取。")
            output("[系统] 执行期间 Ctrl+C 取消当前任务；确认安全结束后可输入下一条。空闲时 Ctrl+C 退出。")
            continue
        feedback = TaskFeedback(output)
        task_options = copy.copy(options)
        task_options.instruction = instruction
        try:
            code = execute(task_options, argv, on_stage=feedback, session_id=session_id)
            if code == 130 and feedback.status is None:
                output("[任务] 当前任务已中断。")
        except (KeyboardInterrupt, EOFError):
            output("[任务] 当前输入或任务已中断。")
        except (ValueError, RuntimeError, OSError) as error:
            output(f"[任务] 失败：{error}")
        if feedback.unsafe or feedback.navigation_pending:
            if feedback.status == "holding":
                return 0
            output("[安全] 任务终态或夹持状态未确认，会话停止接收新任务。请检查导航、机械臂和夹持物。")
            return 1


def main(args=None):
    from rclpy.utilities import remove_ros_args

    # Readline supplies in-terminal editing; no persistent history file is written.
    try:
        import readline  # noqa: F401
    except ImportError:
        pass
    argv = sys.argv if args is None else ["robot_chat", *args]
    options = parser().parse_args(remove_ros_args(argv)[1:])
    try:
        return run_session(options, argv)
    except (ValueError, RuntimeError, OSError) as error:
        print(f"任务窗口启动失败：{error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
