"""Align recorded controller references and build experimental action labels."""

import argparse
from bisect import bisect_left
import json
import math
from pathlib import Path

from wheel_arm_manipulation.controller_reference import ARM_JOINTS, FINGER_JOINTS
from wheel_arm_manipulation.episode_segment import (
    SEGMENT_FILE, SEGMENT_CANDIDATES_FILE, SEGMENT_REPORT_FILE, select_grasp_hold,
)

STATE_NAMES = (*ARM_JOINTS, "gripper_width")
STATE_UNITS = ("rad", "rad", "rad", "rad", "m")
CANDIDATES_FILE = "action_candidates.jsonl"
REPORT_FILE = "action_alignment_report.json"


def read_jsonl(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def ordered_vector(values, names, units, required_names, required_units):
    if len(values) != len(names) or len(names) != len(units) or len(set(names)) != len(names):
        raise ValueError("Vector names, values, and units must match and names must be unique")
    if set(names) != set(required_names):
        raise ValueError("Vector joint names do not match the expected state or controller group")
    by_name = dict(zip(names, zip(values, units)))
    if any(by_name[name][1] != unit for name, unit in zip(required_names, required_units)):
        raise ValueError("Vector units do not match the expected joint order")
    ordered = [by_name[name][0] for name in required_names]
    if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in ordered):
        raise ValueError("State and reference values must be finite numbers")
    return ordered


def normalize_reference(row, group, clock):
    if row.get("kind") != "controller_reference" or row.get("controller") != group:
        raise ValueError("Unexpected controller reference row")
    if row.get("clock") != clock:
        raise ValueError("Observations and controller references must use the same clock")
    if row.get("semantics") != "current_interpolated_position_reference":
        raise ValueError("Unsupported controller reference semantics")
    names = ARM_JOINTS if group == "arm" else FINGER_JOINTS
    units = ("rad",)*4 if group == "arm" else ("m", "m")
    positions = ordered_vector(row["positions"], row["names"], row["units"], names, units)
    if group == "gripper":
        if any(value < -1e-6 or value > .040001 for value in positions):
            raise ValueError("Finger reference is outside the model's 0..0.04 m range")
        if "gripper_width_m" in row and not math.isclose(
                row["gripper_width_m"], sum(positions), rel_tol=0., abs_tol=1e-9):
            raise ValueError("Gripper reference width does not equal the sum of finger positions")
    stamp = row["stamp_ns"]
    if not isinstance(stamp, int) or stamp <= 0:
        raise ValueError("Controller reference timestamps must be positive integers")
    return {"stamp_ns": stamp, "positions": positions}


def interpolate(rows, stamp_ns):
    stamps = [row["stamp_ns"] for row in rows]
    index = bisect_left(stamps, stamp_ns)
    if index < len(rows) and stamps[index] == stamp_ns:
        return list(rows[index]["positions"])
    if index == 0 or index == len(rows):
        raise LookupError("Reference interpolation would require extrapolation")
    lower, upper = rows[index-1], rows[index]
    fraction = (stamp_ns-lower["stamp_ns"])/(upper["stamp_ns"]-lower["stamp_ns"])
    return [a+fraction*(b-a) for a, b in zip(lower["positions"], upper["positions"])]


def load_episode(directory):
    root = Path(directory).expanduser().resolve()
    summary = json.loads((root/"summary.json").read_text(encoding="utf-8"))
    if summary.get("collection_status") != "finished":
        raise ValueError("Only successfully finished episodes can be aligned")
    reference_summary = summary.get("controller_references")
    if not reference_summary or not reference_summary.get("both_controllers_recorded"):
        raise ValueError("Episode must contain both arm and gripper controller references")
    raw_frames = read_jsonl(root/summary.get("frames_file", "frames.jsonl"))
    if not raw_frames or len(raw_frames) != summary.get("frames"):
        raise ValueError("Frame count is empty or does not match summary.json")
    raw_references = read_jsonl(root/reference_summary["file"])
    for group in ("arm", "gripper"):
        count = sum(row.get("controller") == group for row in raw_references)
        if count != reference_summary["saved"][group]:
            raise ValueError(f"{group} reference count does not match summary.json")

    clock = raw_frames[0].get("clock")
    if clock not in ("ros_sim_time", "ros_time"):
        raise ValueError("Episode must declare its ROS clock")
    frames = []
    previous_image_stamp = previous_joint_stamp = None
    for index, frame in enumerate(raw_frames):
        if frame.get("kind") != "observation_only" or frame.get("clock") != clock:
            raise ValueError("Unexpected observation row or inconsistent clock")
        if frame.get("frame_index") != index:
            raise ValueError("Frame indices must be contiguous and ordered")
        actual = ordered_vector(frame["state"]["values"], frame["state"]["names"],
                                frame["state"]["units"], STATE_NAMES, STATE_UNITS)
        image_stamp, joint_stamp = frame["image"]["stamp_ns"], frame["joints_stamp_ns"]
        if not isinstance(image_stamp, int) or not isinstance(joint_stamp, int) or min(
                image_stamp, joint_stamp) <= 0:
            raise ValueError("Observation timestamps must be positive integers")
        if previous_image_stamp is not None and (image_stamp <= previous_image_stamp
                                                or joint_stamp <= previous_joint_stamp):
            raise ValueError("Observation timestamps must increase")
        previous_image_stamp, previous_joint_stamp = image_stamp, joint_stamp
        image_file = frame["image"]["file"]
        image_path = root/image_file
        if not image_path.is_file() or image_path.stat().st_size == 0:
            raise ValueError(f"Missing or empty observation image: {image_file}")
        frames.append({"source": frame, "stamp_ns": joint_stamp, "image_stamp_ns": image_stamp,
                       "actual": actual, "image_file": image_file})

    references = {}
    for group in ("arm", "gripper"):
        rows = [normalize_reference(row, group, clock) for row in raw_references
                if row.get("controller") == group]
        if not rows:
            raise ValueError(f"No {group} controller references")
        if any(b["stamp_ns"] <= a["stamp_ns"] for a, b in zip(rows, rows[1:])):
            raise ValueError(f"{group} controller reference timestamps must increase")
        references[group] = rows
    return {"root": root, "summary": summary, "frames": frames, "references": references}


def build_candidates(frames, references, chunk_size=4, chunk_step_seconds=.1, end_stamp_ns=None):
    if not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("Action chunk size must be a positive integer")
    if not math.isfinite(chunk_step_seconds) or chunk_step_seconds <= 0:
        raise ValueError("Action chunk step must be positive and finite")
    offsets_ns = [round((index+1)*chunk_step_seconds*1e9) for index in range(chunk_size)]
    if offsets_ns[0] <= 0 or len(set(offsets_ns)) != len(offsets_ns):
        raise ValueError("Action chunk step is too small for integer nanosecond timestamps")
    if end_stamp_ns is not None:
        # Even an in-range query must not interpolate against a later lower-stage reference.
        references = {group: [r for r in rows if r["stamp_ns"] < end_stamp_ns]
                      for group, rows in references.items()}
    candidates = []
    skipped = {"current_reference_coverage": 0, "future_reference_coverage": 0}
    if end_stamp_ns is not None:
        skipped["segment_boundary"] = 0
    for frame in frames:
        stamp = frame["stamp_ns"]
        if end_stamp_ns is not None and stamp+offsets_ns[-1] >= end_stamp_ns:
            skipped["segment_boundary"] += 1
            continue
        try:
            arm = interpolate(references["arm"], stamp)
            fingers = interpolate(references["gripper"], stamp)
        except LookupError:
            skipped["current_reference_coverage"] += 1
            continue
        current = arm+[sum(fingers)]
        try:
            future = [interpolate(references["arm"], stamp+offset)+
                      [sum(interpolate(references["gripper"], stamp+offset))]
                      for offset in offsets_ns]
        except LookupError:
            skipped["future_reference_coverage"] += 1
            continue
        candidates.append({
            "schema_version": 1,
            "kind": "action_label_candidates",
            "source_frame_index": frame["source"]["frame_index"],
            "image_file": frame["image_file"],
            "state_stamp_ns": stamp,
            "clock": frame["source"]["clock"],
            "instruction": frame["source"].get("instruction"),
            "names": list(STATE_NAMES),
            "units": list(STATE_UNITS),
            "actual_state": frame["actual"],
            "absolute_reference": current,
            "delta_from_actual": [reference-value for reference, value in zip(current, frame["actual"])],
            "future_offsets_seconds": [offset/1e9 for offset in offsets_ns],
            "future_absolute_references": future,
        })
    if not candidates:
        raise ValueError("No frames have complete current and future controller reference coverage")
    return candidates, skipped


def vector_by_name(vectors):
    return {name: values for name, values in zip(STATE_NAMES, zip(*vectors))}


def peak_to_peak(vectors):
    return {name: max(values)-min(values) for name, values in vector_by_name(vectors).items()}


def maximum_absolute(vectors):
    return {name: max(map(abs, values)) for name, values in vector_by_name(vectors).items()}


def make_report(episode, candidates, skipped, chunk_size, chunk_step_seconds):
    frames, summary = episode["frames"], episode["summary"]
    last_stamp = candidates[-1]["state_stamp_ns"]
    tail = [row for row in candidates if last_stamp-row["state_stamp_ns"] <= 2e9]
    future_change = [[row["future_absolute_references"][-1][index]-row["absolute_reference"][index]
                      for index in range(len(STATE_NAMES))] for row in candidates]
    return {
        "schema_version": 1,
        "kind": "action_alignment_report",
        "source_episode": str(episode["root"]),
        "candidate_file": CANDIDATES_FILE,
        "candidate_semantics": "experimental_not_final_training_labels",
        "alignment": {
            "timestamp": "joints_stamp_ns",
            "method": "linear_interpolation_no_extrapolation",
            "names": list(STATE_NAMES),
            "units": list(STATE_UNITS),
            "future_chunk_size": chunk_size,
            "future_chunk_step_seconds": chunk_step_seconds,
        },
        "input_frames": len(frames),
        "candidate_frames": len(candidates),
        "skipped_frames": skipped,
        "images_checked": len(frames),
        "all_state_and_action_values_finite": True,
        "source_actual_hz": summary["actual_hz"],
        "max_observation_skew_ms": max(frame["source"]["timestamp_difference_ms"] for frame in frames),
        "reference_coverage_seconds": {
            group: (rows[-1]["stamp_ns"]-rows[0]["stamp_ns"])/1e9
            for group, rows in episode["references"].items()
        },
        "actual_motion_range": peak_to_peak([row["actual_state"] for row in candidates]),
        "max_abs_reference_minus_actual": maximum_absolute(
            [row["delta_from_actual"] for row in candidates]),
        "max_abs_future_last_minus_current": maximum_absolute(future_change),
        "last_two_seconds": {
            "frames": len(tail),
            "actual_peak_to_peak": peak_to_peak([row["actual_state"] for row in tail]),
            "max_abs_reference_minus_actual": maximum_absolute(
                [row["delta_from_actual"] for row in tail]),
        },
    }


def align_episode(directory, chunk_size=4, chunk_step_seconds=.1, task_segment=None):
    episode = load_episode(directory)
    segment = None
    source_frame_count = len(episode["frames"])
    candidate_file, report_file = CANDIDATES_FILE, REPORT_FILE
    if task_segment is not None:
        if task_segment != "grasp-hold":
            raise ValueError("Unsupported task segment")
        frames, segment = select_grasp_hold(episode)
        episode = {**episode, "frames": frames}
        candidate_file, report_file = SEGMENT_CANDIDATES_FILE, SEGMENT_REPORT_FILE
    candidates, skipped = build_candidates(episode["frames"], episode["references"],
                                           chunk_size, chunk_step_seconds,
                                           None if segment is None else segment["end_stamp_ns_exclusive"])
    if segment is not None:
        for row in candidates:
            row["source_instruction"] = row["instruction"]
            row["instruction"] = segment["instruction"]
    report = make_report(episode, candidates, skipped, chunk_size, chunk_step_seconds)
    report["candidate_file"] = candidate_file
    report["report_file"] = report_file
    if segment is not None:
        report["segment"] = segment
        report["source_input_frames"] = source_frame_count
        report["images_checked"] = source_frame_count
    root = episode["root"]
    documents = "".join(json.dumps(row, ensure_ascii=False, allow_nan=False)+"\n" for row in candidates)
    if segment is not None:
        (root/SEGMENT_FILE).write_text(json.dumps(segment, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    (root/candidate_file).write_text(documents, encoding="utf-8")
    (root/report_file).write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+"\n",
                                  encoding="utf-8")
    return report


def main(args=None):
    parser = argparse.ArgumentParser(description="离线对齐控制器参考并生成实验性动作标签候选")
    parser.add_argument("episode", help="包含 summary.json 的记录目录")
    parser.add_argument("--chunk-size", type=int, default=4, help="未来绝对位置参考的数量，默认 4")
    parser.add_argument("--chunk-step", type=float, default=.1, help="未来参考的时间间隔，秒，默认 0.1")
    parser.add_argument("--task-segment", choices=("grasp-hold",),
                        help="仅选取 open 前 1 秒至 lift 结束；动作块禁止跨入 lower；输出使用独立文件名")
    parser.add_argument("--time-grid", action="store_true",
                        help="配合 grasp-hold 验证统一 10 Hz 时间网格；保留无效槽位，单独保存诊断文件")
    parser.add_argument("--image-policy", choices=("nearest", "available-past"), default="nearest",
                        help="用于 --time-grid；available-past 只选配对处理时刻不晚于网格的图像，最大图龄 100 ms")
    options = parser.parse_args(args)
    if options.image_policy != "nearest" and not options.time_grid:
        parser.error("--image-policy available-past requires --time-grid")
    if options.time_grid:
        if options.task_segment != "grasp-hold" or options.chunk_step != .1:
            parser.error("--time-grid requires --task-segment grasp-hold and --chunk-step 0.1")
        from wheel_arm_manipulation.time_grid import resample_episode
        try:
            report = resample_episode(options.episode, options.chunk_size, options.image_policy)
        except (KeyError, OSError, ValueError) as error:
            print(f"[失败] {error}", flush=True)
            return 1
        print(f"[时间网格] 10 Hz，共 {report['grid_slots']} 个槽位；"
              f"有效 {report['accepted_slots']}，拒绝 {report['rejected_slots']}。", flush=True)
        print(f"[拒绝原因] {report['rejection_reason_counts']}；同一槽位可有多个原因。", flush=True)
        print(f"[图像策略] {report['image_mode']}；"
              f"时间误差/年龄上限 {report['thresholds_ms']['max_image_error']} ms。", flush=True)
        print(f"[图像误差/全部槽位 ms] {report['absolute_image_error_all_slots_ms']}", flush=True)
        print(f"[图像误差/有效槽位 ms] {report['absolute_image_error_accepted_slots_ms']}", flush=True)
        print(f"[图像复用] 有效槽位重复使用 {report['accepted_reused_image_slots']} 次；"
              f"选到网格时刻之后图像 {report['accepted_future_image_slots']} 次。", flush=True)
        if report['image_mode'] == 'available-past':
            print(f"[有效图像年龄 ms] {report['accepted_image_age_ms']}；"
                  f"配对处理时刻越过网格 {report['accepted_future_availability_slots']} 次。", flush=True)
        print(f"[状态插值] 有效槽位最大支撑间隔 {report['max_accepted_state_support_gap_ms']} ms。", flush=True)
        runs = report['valid_contiguous_runs']
        print(f"[连续性] {len(runs)} 段，最长 {max((r['samples'] for r in runs), default=0)} 个有效槽位；"
              f"可组成连续 {options.chunk_size} 步的起点 {report['contiguous_chunk_start_slots']} 个。", flush=True)
        root = Path(options.episode).expanduser().resolve()
        print(f"[保存] {root/report['grid_file']}", flush=True)
        print(f"[报告] {root/report['report_file']}", flush=True)
        print("[说明] 仅离线诊断；无效槽位不压缩时间，尚未生成 LeRobot 数据集。", flush=True)
        return 0
    try:
        report = align_episode(options.episode, options.chunk_size, options.chunk_step, options.task_segment)
    except (KeyError, OSError, ValueError) as error:
        print(f"[失败] {error}", flush=True)
        return 1
    print(f"[检查] 输入 {report['input_frames']} 帧，生成 {report['candidate_frames']} 帧候选；"
          f"当前参考覆盖不足跳过 {report['skipped_frames']['current_reference_coverage']}，"
          f"未来参考覆盖不足跳过 {report['skipped_frames']['future_reference_coverage']}。", flush=True)
    print(f"[格式] {report['alignment']['names']}；单位 {report['alignment']['units']}；"
          "图像、关节顺序、有限值和时间戳检查通过。", flush=True)
    print(f"[运动] 实际变化范围 {report['actual_motion_range']}", flush=True)
    print(f"[对照] 当前参考减实际状态的最大绝对差 {report['max_abs_reference_minus_actual']}", flush=True)
    print(f"[未来候选] 末尾参考减当前参考的最大绝对变化 "
          f"{report['max_abs_future_last_minus_current']}", flush=True)
    print(f"[末尾2秒] {report['last_two_seconds']}", flush=True)
    if "segment" in report:
        segment = report["segment"]
        print(f"[片段] {segment['task']}；原始 {segment['source_frames']} 帧，"
              f"选取 {segment['selected_frames']} 帧；区间 "
              f"[{segment['start_stamp_ns_inclusive']/1e9:.3f}, "
              f"{segment['end_stamp_ns_exclusive']/1e9:.3f}) s；"
              f"动作块越界跳过 {report['skipped_frames']['segment_boundary']} 帧。", flush=True)
        print(f"[片段元数据] {Path(options.episode).expanduser().resolve()/SEGMENT_FILE}", flush=True)
    print(f"[保存] {Path(options.episode).expanduser().resolve()/report['candidate_file']}", flush=True)
    print(f"[报告] {Path(options.episode).expanduser().resolve()/report['report_file']}", flush=True)
    print("[说明] 这些是绝对位置、增量和未来轨迹候选，尚不是最终 SmolVLA 训练标签。", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
