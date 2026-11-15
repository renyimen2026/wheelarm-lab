"""Record interpolated controller references for later observation/action alignment."""

import json
import math


ARM_JOINTS = ("shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_pitch_joint")
FINGER_JOINTS = ("left_finger_joint", "right_finger_joint")


def controller_reference(message, group, now_ns, clock):
    names = list(message.joint_names)
    required = ARM_JOINTS if group == "arm" else FINGER_JOINTS
    if group not in ("arm", "gripper") or len(set(names)) != len(names) or set(names) != set(required):
        raise ValueError("Controller joint names do not match the expected group")
    # Humble installations may publish the legacy desired field or the newer reference field.
    point = getattr(message, "reference", None)
    field = "reference"
    if point is None or not point.positions:
        point, field = message.desired, "desired"
    if len(point.positions) != len(names) or not all(math.isfinite(v) for v in point.positions):
        raise ValueError("Controller reference positions are missing or non-finite")
    stamp = message.header.stamp.sec*1_000_000_000+message.header.stamp.nanosec
    if stamp <= 0 or not 0 <= now_ns-stamp <= 500_000_000:
        raise ValueError("Controller reference timestamp is zero, stale, or in the future")
    positions = dict(zip(names, point.positions))
    ordered = [positions[name] for name in required]
    if group == "gripper" and any(v < -1e-6 or v > .040001 for v in ordered):
        raise ValueError("Finger reference is outside the model's 0..0.04 m range")
    return {"schema_version": 1, "kind": "controller_reference", "controller": group,
            "stamp_ns": stamp, "clock": clock, "source_field": field,
            "semantics": "current_interpolated_position_reference",
            "names": list(required), "units": ["rad"]*4 if group == "arm" else ["m", "m"],
            "positions": ordered,
            **({"gripper_width_m": sum(ordered)} if group == "gripper" else {})}


class ControllerReferenceRecorder:
    def __init__(self, directory, topics):
        self.stream = (directory/"controller_references.jsonl").open("w", encoding="utf-8", buffering=1)
        self.topics = topics
        self.saved = {"arm": 0, "gripper": 0}
        self.rejected = {"arm": 0, "gripper": 0}
        self.last_error = {"arm": None, "gripper": None}
        self.last_stamp = {"arm": None, "gripper": None}

    def append(self, message, group, now_ns, clock):
        try:
            row = controller_reference(message, group, now_ns, clock)
        except ValueError as error:
            self.rejected[group] += 1
            self.last_error[group] = str(error)
            return
        last = self.last_stamp[group]
        if last is not None and row["stamp_ns"] < last:
            raise ValueError("Controller reference clock moved backwards; start a new episode")
        if row["stamp_ns"] == last:
            return
        row["topic"] = self.topics[group]
        self.stream.write(json.dumps(row, allow_nan=False)+"\n")
        self.last_stamp[group] = row["stamp_ns"]
        self.saved[group] += 1

    def close(self):
        self.stream.close()
        return {"file": "controller_references.jsonl", "topics": self.topics,
                "saved": dict(self.saved), "rejected": dict(self.rejected),
                "last_error": dict(self.last_error),
                "both_controllers_recorded": all(self.saved.values()),
                "aligned_action_labels": False}
