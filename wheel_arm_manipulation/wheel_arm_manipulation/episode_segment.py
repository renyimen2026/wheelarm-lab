"""Select a verified grasp-and-hold prefix without modifying source recordings."""

import json


SEGMENT_FILE = "grasp_hold_segment.json"
SEGMENT_CANDIDATES_FILE = "grasp_hold_action_candidates.jsonl"
SEGMENT_REPORT_FILE = "grasp_hold_action_alignment_report.json"
HOLD_INSTRUCTION = "拿起桌上的杯子并保持夹持"
HOLD_PHASES = ("open", "pregrasp", "approach", "close", "lift")


def select_grasp_hold(episode):
    root, frames = episode["root"], episode["frames"]
    task = json.loads((root/"task_metadata.json").read_text(encoding="utf-8"))
    events = [json.loads(line) for line in (root/"task_events.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip()]
    if (task.get("source") != "cup_grasp" or task.get("execution_status") != "finished"
            or task.get("task_success") is not True or task.get("lift_verified") is not True):
        raise ValueError("Grasp-hold selection requires a successfully finished, lift-verified cup_grasp task")
    if not events or not task.get("task_id"):
        raise ValueError("Task events and task_id are required")
    if task.get("clock") != frames[0]["source"]["clock"]:
        raise ValueError("Task and observations must use the same clock")
    previous = 0
    for index, event in enumerate(events):
        stamp = event.get("stamp_ns")
        if (event.get("task_id") != task["task_id"] or event.get("event_index") != index
                or not isinstance(stamp, int) or stamp <= 0 or stamp < previous):
            raise ValueError("Task events must have matching IDs and ordered indices/timestamps")
        previous = stamp
    terminal = events[-1]
    if (terminal.get("event") != "task_finished" or terminal.get("task_success") is not True
            or terminal.get("execution_status") != "finished"):
        raise ValueError("Missing successful terminal task event")

    def unique(kind, phase=None):
        matches = [e for e in events if e["event"] == kind
                   and (phase is None or e.get("phase") == phase)]
        if len(matches) != 1:
            raise ValueError(f"Expected one {kind} event for {phase}")
        return matches[0]

    boundaries = []
    for phase in HOLD_PHASES:
        start, end = unique("phase_started", phase), unique("phase_finished", phase)
        if start["event_index"] >= end["event_index"]:
            raise ValueError("Phase must start before it finishes")
        boundaries.extend((start, end))
    if [e["event_index"] for e in boundaries] != sorted(e["event_index"] for e in boundaries):
        raise ValueError("Grasp-hold phase order is inconsistent")
    verified = unique("lift_verified")
    lift_start, lift_end = boundaries[-2:]
    if not lift_start["event_index"] < verified["event_index"] < lift_end["event_index"]:
        raise ValueError("Lift verification must occur inside the lift phase")
    for event in events:
        if event.get("phase") in ("lower", "release", "retreat") and event["event"] == "phase_started":
            if event["event_index"] < lift_end["event_index"]:
                raise ValueError("Place motions must not precede the lift boundary")

    first_stamp = min(frames[0]["stamp_ns"], frames[0]["image_stamp_ns"])
    last_stamp = max(frames[-1]["stamp_ns"], frames[-1]["image_stamp_ns"])
    if first_stamp > boundaries[0]["stamp_ns"] or last_stamp < lift_end["stamp_ns"]:
        raise ValueError("Observations must cover the complete grasp-hold phases")
    start_ns = max(first_stamp, boundaries[0]["stamp_ns"]-1_000_000_000)
    end_ns = lift_end["stamp_ns"]
    selected = [f for f in frames if start_ns <= min(f["stamp_ns"], f["image_stamp_ns"])
                and max(f["stamp_ns"], f["image_stamp_ns"]) < end_ns]
    if not selected:
        raise ValueError("No observations within the grasp-hold segment")
    segment = {
        "schema_version": 1, "kind": "episode_segment", "task": "grasp_hold",
        "source_episode": str(root), "source_task_id": task["task_id"],
        "source_task_finish": task.get("finish"), "clock": task["clock"],
        "instruction": HOLD_INSTRUCTION, "phases": list(HOLD_PHASES),
        "timestamp": "joints_stamp_ns_and_image_stamp_ns",
        "start_stamp_ns_inclusive": start_ns, "end_stamp_ns_exclusive": end_ns,
        "requested_preroll_seconds": 1., "postroll_seconds": 0.,
        "terminal_event": "lift.phase_finished", "lift_verified_stamp_ns": verified["stamp_ns"],
        "verification_to_phase_end_seconds": (end_ns-verified["stamp_ns"])/1e9,
        "source_frames": len(frames), "selected_frames": len(selected),
        "first_source_frame_index": selected[0]["source"]["frame_index"],
        "last_source_frame_index": selected[-1]["source"]["frame_index"],
        "action_boundary_policy": "skip_incomplete_chunks_no_padding; interpolation_support_before_end",
        "success_evidence": "source_task_rgbd_lift_verification",
    }
    return selected, segment
