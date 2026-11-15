"""Explicit, single-task sidecars for an existing observation episode."""

from datetime import datetime, timezone
import json
from pathlib import Path
from uuid import uuid4


def document(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2)+"\n"


class TaskMetadataRecorder:
    """A missing terminal result always leaves task_success unknown.

    Event stamps use the same ROS clock as observations, not file wall time.
    One episode accepts one task; exclusive creation prevents accidental reuse.
    """

    def __init__(self, episode_dir, finish, now_ns):
        self.path = Path(episode_dir).expanduser().resolve(strict=True)
        frames = self.path/"frames.jsonl"
        if not frames.is_file() or (self.path/"summary.json").exists():
            raise ValueError("--episode-dir must be an active recording directory with frames.jsonl")
        with frames.open(encoding="utf-8") as stream:
            first = json.loads(stream.readline())
            latest = first
            for line in stream:
                if line.endswith("\n"):
                    latest = json.loads(line)
        if first.get("clock") != "ros_sim_time":
            raise ValueError("cup_grasp metadata requires observations using ros_sim_time")
        stamp = now_ns()
        if stamp <= 0 or not 0 <= stamp-latest["joints_stamp_ns"] <= 2_000_000_000:
            raise ValueError("Episode must have a recent frame (within 2 s) on the same ROS clock")
        if finish not in ("place", "hold"):
            raise ValueError("Unknown task finish mode")
        self.now_ns = now_ns
        self.last_stamp = stamp
        self.metadata = {
            "schema_version": 1, "task_id": uuid4().hex, "source": "cup_grasp",
            "episode_dir": str(self.path), "frames_file": "frames.jsonl",
            "events_file": "task_events.jsonl", "clock": "ros_sim_time",
            "finish": finish, "execution_status": "running", "task_success": None,
            "execution_started": False, "procedure_completed": False,
            "lift_verified": False, "placement_verified": False,
            "active_phase": None, "result_reason": "No terminal result recorded",
        }
        # Claim before opening events: existing metadata is never overwritten.
        with (self.path/"task_metadata.json").open("x", encoding="utf-8") as stream:
            stream.write(document(self.metadata))
        self.stream = (self.path/"task_events.jsonl").open("x", encoding="utf-8", buffering=1)
        self.event("task_linked")

    def save(self):
        temporary = self.path/(".task_metadata-"+self.metadata["task_id"]+".tmp")
        temporary.write_text(document(self.metadata), encoding="utf-8")
        temporary.replace(self.path/"task_metadata.json")

    def event(self, kind, **details):
        stamp = self.now_ns()
        if stamp <= 0 or stamp < self.last_stamp:
            raise ValueError("Task ROS clock moved backwards; start a new episode")
        event = {"schema_version": 1, "task_id": self.metadata["task_id"],
                 "event_index": getattr(self, "count", 0), "stamp_ns": stamp,
                 "wall_time_utc": datetime.now(timezone.utc).isoformat(),
                 "event": kind, **details}
        self.stream.write(json.dumps(event, ensure_ascii=False, allow_nan=False)+"\n")
        self.count = event["event_index"]+1
        self.last_stamp = stamp
        if kind == "execution_started":
            self.metadata["execution_started"] = True
        elif kind == "phase_started":
            self.metadata["active_phase"] = details["phase"]
        elif kind == "phase_finished":
            self.metadata["active_phase"] = None
        elif kind in ("lift_verified", "placement_verified", "procedure_completed"):
            self.metadata[kind] = True
        self.save()

    def finish(self, status, reason):
        try:
            success = None
            if status == "finished":
                success = (self.metadata["procedure_completed"] and self.metadata["lift_verified"]
                           and (self.metadata["finish"] == "hold" or self.metadata["placement_verified"]))
                if not success:
                    status = "failed"
                    reason = "Procedure returned without required verification"
            elif status == "failed" and self.metadata["execution_started"]:
                success = False
            # Persist the terminal event before claiming a known outcome.
            self.event("task_finished", execution_status=status, task_success=success, reason=reason)
            self.metadata.update(execution_status=status, task_success=success, result_reason=reason)
            self.save()
        finally:
            self.stream.close()


def task_event(node, kind, **details):
    recorder = getattr(node, "task_record", None)
    if recorder is not None:
        recorder.event(kind, **details)


def run_with_metadata(node, options, run, confirm=None):
    """Keep success pending until run() and its scene cleanup both return."""
    if not options.episode_dir:
        return run(node, options, confirm)
    # A new use_sim_time node starts at zero until /clock has been processed.
    node.snapshot()
    recorder = TaskMetadataRecorder(options.episode_dir, options.finish,
                                    lambda: node.get_clock().now().nanoseconds)
    node.task_record = recorder
    print(f"[示范关联] {recorder.path}；任务 {recorder.metadata['task_id']}", flush=True)
    try:
        result = run(node, options, confirm)
    except (KeyboardInterrupt, EOFError):
        recorder.finish("interrupted", "Task interrupted; inspect the scene")
        raise
    except Exception as error:
        recorder.finish("failed", str(error))
        raise
    else:
        status = "finished" if recorder.metadata["execution_started"] else "cancelled"
        recorder.finish(status, "Procedure returned" if status == "finished" else "Execution not confirmed")
        print(f"[任务结果] {recorder.metadata['execution_status']}；"
              f"task_success={recorder.metadata['task_success']}", flush=True)
        return 1 if recorder.metadata["execution_status"] == "failed" else result
