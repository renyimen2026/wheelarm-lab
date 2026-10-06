"""Mission validation and execution independent of ROS transports."""

from datetime import datetime, timezone
import json
import math
from pathlib import Path
import tempfile
import time
from uuid import uuid4

import yaml


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def validate_task(task):
    if not isinstance(task, dict):
        raise ValueError("任务必须是 YAML 映射")
    allowed = {"name", "goals", "stay", "timeout", "wait", "repeat", "on_failure"}
    unknown = set(task) - allowed
    if unknown:
        raise ValueError(f"未知任务配置项：{unknown}")
    plan = {"name": "navigation_task", "stay": 0.0, "timeout": 180.0,
            "wait": 10.0, "repeat": 1, "on_failure": "stop", **task}
    if not isinstance(plan["name"], str) or not plan["name"].strip():
        raise ValueError("任务 name 必须是非空字符串")
    goals = plan.get("goals")
    if (not isinstance(goals, list) or not goals
            or any(not isinstance(name, str) or not name.strip() for name in goals)):
        raise ValueError("goals 必须是非空目标名称列表")
    for key in ("stay", "timeout", "wait"):
        value = plan[key]
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value < 0
                or (key != "stay" and value == 0)):
            raise ValueError(f"{key} 必须是有限数值；stay >= 0，timeout 和 wait > 0")
    if type(plan["repeat"]) is not int or plan["repeat"] < 1:
        raise ValueError("repeat 必须是正整数")
    if plan["on_failure"] not in ("stop", "skip"):
        raise ValueError("on_failure 只能是 stop 或 skip")
    return plan


def load_task(path):
    with Path(path).expanduser().open(encoding="utf-8") as stream:
        try:
            return validate_task(yaml.safe_load(stream))
        except yaml.YAMLError as error:
            raise ValueError(f"任务 YAML 格式错误：{error}") from error


def check_goals(plan, document):
    missing = sorted(set(plan["goals"]) - set(document["goals"]))
    if missing:
        raise ValueError(f"以下目标尚未记录：{', '.join(missing)}；整项任务未启动")


class Journal:
    def __init__(self, directory, plan, document, goals_file):
        directory = Path(directory).expanduser()
        directory.mkdir(parents=True, exist_ok=True)
        identifier = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex
        self.path = directory / (identifier + ".json")
        self.started = time.monotonic()
        self.data = {
            "schema_version": 1, "mission_id": identifier, "task": plan,
            "goals_file": str(Path(goals_file).expanduser().resolve()),
            "frame_id": document["frame_id"], "started_at": utc_now(),
            "finished_at": None, "elapsed_sec": 0.0, "status": "running",
            "steps": [
                {"index": index + 1, "round": round_number + 1, "name": name,
                 "pose": dict(document["goals"][name]), "status": "not_run",
                 "started_at": None, "finished_at": None, "navigation_sec": 0.0,
                 "stay_sec": 0.0, "reason": "", "action_status": None,
                 "safe_to_continue": True}
                for index, (round_number, name) in enumerate(
                    (round_number, name) for round_number in range(plan["repeat"])
                    for name in plan["goals"])
            ],
        }
        self.flush()

    def flush(self):
        self.data["elapsed_sec"] = round(time.monotonic() - self.started, 3)
        steps = self.data["steps"]
        self.data["summary"] = {
            "total": len(steps),
            "succeeded": sum(step["status"] == "succeeded" for step in steps),
            "failed": sum(step["status"] not in (
                "not_run", "running", "succeeded", "canceled") for step in steps),
            "canceled": sum(step["status"] == "canceled" for step in steps),
            "not_run": sum(step["status"] == "not_run" for step in steps),
            "running": sum(step["status"] == "running" for step in steps),
        }
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                    mode="w", encoding="utf-8", dir=self.path.parent,
                    prefix=".mission-", suffix=".json", delete=False) as stream:
                temporary = Path(stream.name)
                json.dump(self.data, stream, ensure_ascii=False, indent=2, allow_nan=False)
                stream.write("\n")
            temporary.replace(self.path)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()


def execute(plan, document, goals_file, log_dir, navigate, stopping, pause, progress):
    plan = validate_task(plan)
    check_goals(plan, document)
    journal = Journal(log_dir, plan, document, goals_file)
    progress(f"任务日志：{journal.path}")
    status = "completed"
    had_failure = False
    try:
        for step in journal.data["steps"]:
            if stopping():
                status = "canceled"
                break
            progress(f"[{step['index']}/{len(journal.data['steps'])}] 导航到 {step['name']}")
            step.update(status="running", started_at=utc_now())
            journal.flush()
            started = time.monotonic()
            try:
                result = navigate(step["name"], step["pose"], document["frame_id"],
                                  plan["wait"], plan["timeout"])
                step.update(status=result.status, reason=result.reason,
                            action_status=result.action_status,
                            safe_to_continue=result.safe_to_continue)
            except Exception as error:
                step.update(status="error", reason=str(error), safe_to_continue=False)
            step["navigation_sec"] = round(time.monotonic() - started, 3)
            step["finished_at"] = utc_now()
            journal.flush()
            if not step["safe_to_continue"]:
                status = "unsafe_stop"
                progress("目标状态未确认，停止任务；请确认 Nav2 和机器人状态")
                break
            if stopping() or step["status"] == "canceled":
                status = "canceled"
                break
            if step["status"] != "succeeded":
                had_failure = True
                progress(f"目标失败：{step['status']}；{step['reason']}")
                if plan["on_failure"] == "stop":
                    status = "failed"
                    break
                continue
            if plan["stay"]:
                progress(f"停留 {plan['stay']:g} 秒")
                started = time.monotonic()
                rested = pause(plan["stay"])
                step["stay_sec"] = round(time.monotonic() - started, 3)
                journal.flush()
                if not rested or stopping():
                    status = "canceled"
                    break
        if status == "completed" and had_failure:
            status = "completed_with_failures"
    except BaseException:
        status = "error"
        raise
    finally:
        journal.data.update(status=status, finished_at=utc_now())
        journal.flush()
    progress(f"任务结束：{status}；日志：{journal.path}")
    return (0 if status == "completed" else 130 if status == "canceled" else 1), journal.path
