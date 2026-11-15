"""Diagnostic 10 Hz resampling; retain rejected slots instead of compressing time."""

from bisect import bisect_left
from collections import Counter
import json
import math

from wheel_arm_manipulation.action_alignment import STATE_NAMES, STATE_UNITS, load_episode
from wheel_arm_manipulation.episode_segment import select_grasp_hold


GRID_FILE = "grasp_hold_time_grid.jsonl"
GRID_REPORT_FILE = "grasp_hold_time_grid_report.json"
AVAILABLE_GRID_FILE = "grasp_hold_available_time_grid.jsonl"
AVAILABLE_GRID_REPORT_FILE = "grasp_hold_available_time_grid_report.json"
PERIOD_NS = 100_000_000
MAX_IMAGE_ERROR_NS = 50_000_000
MAX_AVAILABLE_IMAGE_AGE_NS = 100_000_000
MAX_STATE_GAP_NS = 250_000_000
MAX_REFERENCE_GAP_NS = 100_000_000


def bounded_interpolation(rows, stamp, max_gap_ns):
    """Return interpolated values and their real temporal support, never extrapolate."""
    stamps = [row["stamp_ns"] for row in rows]
    index = bisect_left(stamps, stamp)
    if index < len(rows) and stamps[index] == stamp:
        lower = upper = rows[index]
    elif index == 0 or index == len(rows):
        raise LookupError("coverage")
    else:
        lower, upper = rows[index-1:index+1]
    span = upper["stamp_ns"]-lower["stamp_ns"]
    if span > max_gap_ns:
        raise LookupError("gap")
    fraction = (stamp-lower["stamp_ns"])/span if span else 0.
    values = [a+fraction*(b-a) for a, b in zip(lower["positions"], upper["positions"])]
    support = {"lower_stamp_ns": lower["stamp_ns"], "upper_stamp_ns": upper["stamp_ns"],
               "gap_ms": span/1e6}
    if "source_frame_index" in lower:
        support["source_frame_indices"] = [lower["source_frame_index"], upper["source_frame_index"]]
    return values, support


def nearest_image(frames, stamp):
    stamps = [f["image_stamp_ns"] for f in frames]
    index = bisect_left(stamps, stamp)
    choices = frames[max(0, index-1):min(len(frames), index+1)]
    # Prefer the earlier real frame on an exact tie.
    return min(choices, key=lambda f: (abs(f["image_stamp_ns"]-stamp), f["image_stamp_ns"]))


def pair_validation_stamp_ns(frame):
    """Recover the node ROS time recorded when this pair passed freshness checks.

    This is a conservative raw-image availability marker, not transport arrival
    time or completion of image conversion/file writing. Old files without ages
    cannot establish availability and must not silently fall back to source time.
    """
    age = frame["source"].get("image_age_ms")
    if type(age) not in (int, float) or not math.isfinite(age) or not 0 <= age <= 500:
        raise ValueError("available-past requires recorded finite image_age_ms in [0, 500] for every source frame")
    stamp = frame["image_stamp_ns"]+round(age*1e6)
    if stamp < frame["stamp_ns"]:
        raise ValueError("Pair validation time must not precede its joint timestamp")
    joint_age = frame["source"].get("joints_age_ms")
    if joint_age is not None:
        if (type(joint_age) not in (int, float) or not math.isfinite(joint_age) or joint_age < 0
                or abs(frame["stamp_ns"]+round(joint_age*1e6)-stamp) > 1):
            raise ValueError("Image and joint ages must describe the same pair validation time")
    return stamp


def latest_available_image(images_with_availability, stamp):
    eligible = [(frame, available) for frame, available in images_with_availability if available <= stamp]
    return max(eligible, key=lambda pair: pair[0]["image_stamp_ns"]) if eligible else (None, None)


def statistics_ms(values):
    if not values:
        return {"count": 0, "mean": None, "p95": None, "max": None}
    ordered = sorted(values)
    return {"count": len(values), "mean": sum(values)/len(values)/1e6,
            "p95": ordered[math.ceil(.95*len(values))-1]/1e6, "max": ordered[-1]/1e6}


def valid_runs(rows):
    runs = []
    for row in rows:
        if row["status"] != "accepted":
            continue
        index = row["grid_index"]
        if not runs or runs[-1]["last_grid_index"] != index-1:
            runs.append({"first_grid_index": index, "last_grid_index": index, "samples": 1})
        else:
            runs[-1]["last_grid_index"] = index
            runs[-1]["samples"] += 1
    return runs


def build_time_grid(frames, references, segment, chunk_size=4, image_policy="nearest"):
    if type(chunk_size) is not int or chunk_size <= 0:
        raise ValueError("Chunk size must be a positive integer")
    if image_policy not in ("nearest", "available-past"):
        raise ValueError("Image policy must be nearest or available-past")
    start, end = segment["start_stamp_ns_inclusive"], segment["end_stamp_ns_exclusive"]
    if start >= end:
        raise ValueError("Segment must have positive duration")
    # All interpolation support and source images remain strictly inside the segment.
    frames = [f for f in frames if start <= min(f["stamp_ns"], f["image_stamp_ns"])
              and max(f["stamp_ns"], f["image_stamp_ns"]) < end]
    if not frames:
        raise ValueError("Segment has no source observations")
    available_images = [(f, pair_validation_stamp_ns(f)) for f in frames] if image_policy == "available-past" else []
    max_image_error = MAX_AVAILABLE_IMAGE_AGE_NS if image_policy == "available-past" else MAX_IMAGE_ERROR_NS
    states = [{"stamp_ns": f["stamp_ns"], "positions": f["actual"],
               "source_frame_index": f["source"]["frame_index"]} for f in frames]
    references = {group: [r for r in rows if start <= r["stamp_ns"] < end]
                  for group, rows in references.items()}
    rows, errors_all, errors_accepted = [], [], []
    reasons = Counter()
    for index, stamp in enumerate(range(start, end, PERIOD_NS)):
        row = {"schema_version": 1, "kind": "time_grid_diagnostic",
               "grid_index": index, "stamp_ns": stamp, "relative_stamp_ns": stamp-start,
               "clock": segment["clock"], "instruction": segment["instruction"],
               "names": list(STATE_NAMES), "units": list(STATE_UNITS), "rejection_reasons": []}
        available = None
        if image_policy == "available-past":
            image, available = latest_available_image(available_images, stamp)
        else:
            image = nearest_image(frames, stamp)
        row["image"] = None
        if image is None:
            row["rejection_reasons"].append("image_unavailable")
        else:
            offset = image["image_stamp_ns"]-stamp
            errors_all.append(abs(offset))
            row["image"] = {"file": image["image_file"], "stamp_ns": image["image_stamp_ns"],
                            "source_frame_index": image["source"]["frame_index"],
                            "offset_from_grid_ms": offset/1e6}
            if available is not None:
                row["image"].update(pair_validation_stamp_ns=available, age_at_grid_ms=-offset/1e6)
            if abs(offset) > max_image_error:
                row["rejection_reasons"].append("image_error")
        try:
            row["state"], row["state_support"] = bounded_interpolation(states, stamp, MAX_STATE_GAP_NS)
        except LookupError as error:
            row["rejection_reasons"].append("state_"+str(error))
        if stamp+chunk_size*PERIOD_NS >= end:
            row["rejection_reasons"].append("action_segment_boundary")
        else:
            future, support = [], []
            for step in range(1, chunk_size+1):
                target = stamp+step*PERIOD_NS
                values, brackets = {}, {}
                for group in ("arm", "gripper"):
                    try:
                        values[group], brackets[group] = bounded_interpolation(
                            references[group], target, MAX_REFERENCE_GAP_NS)
                    except LookupError as error:
                        reason = group+"_reference_"+str(error)
                        if reason not in row["rejection_reasons"]:
                            row["rejection_reasons"].append(reason)
                if len(values) == 2:
                    future.append(values["arm"]+[sum(values["gripper"])])
                    support.append(brackets)
            if len(future) == chunk_size:
                row["action"] = future[0]
                row["action_stamp_ns"] = stamp+PERIOD_NS
                row["future_stamp_ns"] = [stamp+i*PERIOD_NS for i in range(1, chunk_size+1)]
                row["future_absolute_references"] = future
                row["reference_support"] = support
        row["status"] = "rejected" if row["rejection_reasons"] else "accepted"
        reasons.update(row["rejection_reasons"])
        if row["status"] == "accepted":
            errors_accepted.append(abs(offset))
        rows.append(row)
    accepted = [r for r in rows if r["status"] == "accepted"]
    image_uses = Counter(r["image"]["source_frame_index"] for r in accepted)
    runs = valid_runs(rows)
    report = {
        "schema_version": 1, "kind": "time_grid_report", "semantics": "diagnostic_not_lerobot_dataset",
        "grid_file": GRID_FILE if image_policy == "nearest" else AVAILABLE_GRID_FILE,
        "report_file": GRID_REPORT_FILE if image_policy == "nearest" else AVAILABLE_GRID_REPORT_FILE,
        "segment": segment,
        "grid_hz": 10, "period_ns": PERIOD_NS, "grid_anchor": "segment_start",
        "image_mode": image_policy,
        "image_policy": ("nearest_real_image; ties_choose_earlier; may_use_future_image_within_tolerance"
                         if image_policy == "nearest" else
                         "latest_source_image_with_pair_validation_ros_time_at_or_before_grid"),
        "state_policy": "linear_interpolation_no_extrapolation",
        "action_policy": "absolute_reference_at_grid_plus_0.1s",
        "chunk_size": chunk_size, "future_offsets_seconds": [i/10 for i in range(1, chunk_size+1)],
        "thresholds_ms": {"max_image_error": max_image_error/1e6,
                          "max_state_interpolation_gap": MAX_STATE_GAP_NS/1e6,
                          "max_reference_interpolation_gap": MAX_REFERENCE_GAP_NS/1e6},
        "grid_slots": len(rows), "accepted_slots": len(accepted),
        "rejected_slots": len(rows)-len(accepted), "rejection_reason_counts": dict(reasons),
        "rejection_counts_may_overlap": True,
        "absolute_image_error_all_slots_ms": statistics_ms(errors_all),
        "absolute_image_error_accepted_slots_ms": statistics_ms(errors_accepted),
        "accepted_future_image_slots": sum(r["image"]["offset_from_grid_ms"] > 0 for r in accepted),
        "accepted_unique_images": len(image_uses),
        "accepted_reused_image_slots": sum(n-1 for n in image_uses.values()),
        "max_accepted_state_support_gap_ms": max((r["state_support"]["gap_ms"] for r in accepted), default=None),
        "valid_contiguous_runs": runs,
        "contiguous_chunk_start_slots": sum(max(0, r["samples"]-chunk_size+1) for r in runs),
        "time_compression_allowed": False,
    }
    if image_policy == "available-past":
        report["availability_semantics"] = "node_ros_time_at_pair_validation; not_transport_latency_or_decode_completion"
        report["accepted_future_availability_slots"] = sum(
            r["image"]["pair_validation_stamp_ns"] > r["stamp_ns"] for r in accepted)
        report["accepted_image_age_ms"] = statistics_ms(errors_accepted)
    return rows, report


def resample_episode(directory, chunk_size=4, image_policy="nearest"):
    episode = load_episode(directory)
    frames, segment = select_grasp_hold(episode)
    rows, report = build_time_grid(frames, episode["references"], segment, chunk_size, image_policy)
    root = episode["root"]
    (root/report["grid_file"]).write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False)+"\n"
                                       for row in rows), encoding="utf-8")
    (root/report["report_file"]).write_text(json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2)+"\n",
                                      encoding="utf-8")
    return report
