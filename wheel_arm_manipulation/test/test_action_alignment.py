import json

import pytest

from wheel_arm_manipulation.action_alignment import (
    CANDIDATES_FILE, REPORT_FILE, STATE_NAMES, STATE_UNITS, align_episode, build_candidates,
    interpolate, load_episode, ordered_vector,
)
from wheel_arm_manipulation.controller_reference import ARM_JOINTS, FINGER_JOINTS


def reference(group, stamp, positions, reverse=False):
    names = list(ARM_JOINTS if group == "arm" else FINGER_JOINTS)
    units = ["rad"]*4 if group == "arm" else ["m"]*2
    if reverse:
        names, positions, units = names[::-1], positions[::-1], units[::-1]
    return {"kind": "controller_reference", "controller": group, "stamp_ns": stamp,
            "clock": "ros_sim_time", "semantics": "current_interpolated_position_reference",
            "names": names, "units": units, "positions": positions,
            **({"gripper_width_m": sum(positions)} if group == "gripper" else {})}


def observation(stamp, actual, image="images/000000.png", index=0):
    return {"kind": "observation_only", "frame_index": index, "clock": "ros_sim_time",
            "image": {"file": image, "stamp_ns": stamp}, "joints_stamp_ns": stamp,
            "timestamp_difference_ms": 0., "instruction": "移动",
            "state": {"names": list(STATE_NAMES), "units": list(STATE_UNITS), "values": actual}}


def normalized_frame(stamp, actual, index=0):
    source = observation(stamp, actual, index=index)
    return {"source": source, "stamp_ns": stamp, "image_stamp_ns": stamp,
            "actual": actual, "image_file": source["image"]["file"]}


def normalized_references(stamps, arm_values, finger_values):
    return {
        "arm": [{"stamp_ns": stamp, "positions": list(values)} for stamp, values in zip(stamps, arm_values)],
        "gripper": [{"stamp_ns": stamp, "positions": list(values)}
                    for stamp, values in zip(stamps, finger_values)],
    }


def test_vector_order_and_units_are_checked_by_joint_name():
    values = ordered_vector([.9, -.6, -.3, 0.], list(ARM_JOINTS)[::-1], ["rad"]*4,
                            ARM_JOINTS, ("rad",)*4)
    assert values == [0., -.3, -.6, .9]
    with pytest.raises(ValueError, match="units"):
        ordered_vector([0., 0., 0., 0.], ARM_JOINTS, ["m"]*4, ARM_JOINTS, ("rad",)*4)


def test_interpolation_is_linear_and_never_extrapolates():
    rows = [{"stamp_ns": 1_000_000_000, "positions": [0., 2.]},
            {"stamp_ns": 1_100_000_000, "positions": [1., 4.]}]
    assert interpolate(rows, 1_040_000_000) == pytest.approx([.4, 2.8])
    assert interpolate(rows, 1_000_000_000) == [0., 2.]
    with pytest.raises(LookupError):
        interpolate(rows, 999_000_000)
    with pytest.raises(LookupError):
        interpolate(rows, 1_101_000_000)


def test_candidates_include_absolute_delta_and_future_chunks():
    stamps = [900_000_000, 1_000_000_000, 1_100_000_000, 1_200_000_000, 1_300_000_000]
    arm_values = [[value, 0., 0., 0.] for value in (0., .1, .2, .3, .4)]
    references = normalized_references(stamps, arm_values, [[.02, .02]]*5)
    frames = [normalized_frame(1_000_000_000, [0., 0., 0., 0., .04]),
              normalized_frame(1_100_000_000, [.15, 0., 0., 0., .04], 1)]
    candidates, skipped = build_candidates(frames, references, chunk_size=2, chunk_step_seconds=.1)
    assert len(candidates) == 2
    assert skipped == {"current_reference_coverage": 0, "future_reference_coverage": 0}
    first = candidates[0]
    assert first["kind"] == "action_label_candidates"
    assert first["names"] == list(STATE_NAMES)
    assert first["absolute_reference"] == pytest.approx([.1, 0., 0., 0., .04])
    assert first["delta_from_actual"] == pytest.approx([.1, 0., 0., 0., 0.])
    assert first["future_offsets_seconds"] == pytest.approx([.1, .2])
    for actual, expected in zip(first["future_absolute_references"],
                                [[.2, 0., 0., 0., .04], [.3, 0., 0., 0., .04]]):
        assert actual == pytest.approx(expected)
    assert candidates[1]["delta_from_actual"][0] == pytest.approx(.05)


def test_frames_without_current_or_future_coverage_are_not_fabricated():
    stamps = [1_000_000_000, 1_100_000_000]
    references = normalized_references(stamps, [[0., 0., 0., 0.]]*2, [[.02, .02]]*2)
    frames = [normalized_frame(900_000_000, [0.]*4+[.04]),
              normalized_frame(1_000_000_000, [0.]*4+[.04], 1),
              normalized_frame(1_100_000_000, [0.]*4+[.04], 2)]
    candidates, skipped = build_candidates(frames, references, chunk_size=1, chunk_step_seconds=.1)
    assert len(candidates) == 1
    assert candidates[0]["source_frame_index"] == 1
    assert skipped == {"current_reference_coverage": 1, "future_reference_coverage": 1}


def write_episode(root, reverse=False):
    (root/"images").mkdir(parents=True)
    (root/"images/000000.png").write_bytes(b"png")
    actual = [0., -.6, .9, -.3, .06]
    frame = observation(1_000_000_000, actual)
    if reverse:
        frame["state"] = {"names": list(STATE_NAMES)[::-1], "units": list(STATE_UNITS)[::-1],
                          "values": actual[::-1]}
    (root/"frames.jsonl").write_text(json.dumps(frame)+"\n", encoding="utf-8")
    rows = []
    for stamp in (900_000_000, 1_000_000_000, 1_100_000_000, 1_200_000_000):
        rows.append(reference("arm", stamp, actual[:4], reverse))
        rows.append(reference("gripper", stamp, [.03, .03], reverse))
    (root/"controller_references.jsonl").write_text(
        "".join(json.dumps(row)+"\n" for row in rows), encoding="utf-8")
    summary = {"collection_status": "finished", "frames": 1, "actual_hz": None,
               "controller_references": {"file": "controller_references.jsonl",
                                         "both_controllers_recorded": True,
                                         "saved": {"arm": 4, "gripper": 4}}}
    (root/"summary.json").write_text(json.dumps(summary), encoding="utf-8")


def test_episode_alignment_writes_candidates_and_full_report(tmp_path):
    write_episode(tmp_path, reverse=True)
    report = align_episode(tmp_path, chunk_size=2, chunk_step_seconds=.1)
    assert report["input_frames"] == report["candidate_frames"] == 1
    assert report["images_checked"] == 1
    assert report["all_state_and_action_values_finite"]
    assert report["candidate_semantics"] == "experimental_not_final_training_labels"
    assert report["actual_motion_range"] == {name: 0. for name in STATE_NAMES}
    assert report["last_two_seconds"]["frames"] == 1
    candidates = [json.loads(line) for line in (tmp_path/CANDIDATES_FILE).read_text().splitlines()]
    assert candidates[0]["absolute_reference"] == pytest.approx([0., -.6, .9, -.3, .06])
    assert len(candidates[0]["future_absolute_references"]) == 2
    assert json.loads((tmp_path/REPORT_FILE).read_text()) == report


def test_episode_integrity_rejects_missing_images_and_bad_counts(tmp_path):
    write_episode(tmp_path)
    (tmp_path/"images/000000.png").unlink()
    with pytest.raises(ValueError, match="Missing or empty"):
        load_episode(tmp_path)
    (tmp_path/"images/000000.png").write_bytes(b"png")
    summary = json.loads((tmp_path/"summary.json").read_text())
    summary["frames"] = 2
    (tmp_path/"summary.json").write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="Frame count"):
        load_episode(tmp_path)


@pytest.mark.parametrize("chunk_size,step", [(0, .1), (1, 0.), (1, float("nan"))])
def test_invalid_future_chunk_parameters_are_rejected(chunk_size, step):
    with pytest.raises(ValueError):
        build_candidates([], {}, chunk_size, step)


def write_segment_episode(root):
    write_episode(root)
    stamps = list(range(800_000_000, 2_000_000_001, 100_000_000))
    frames = [observation(stamp, [0.]*4+[.04], index=index) for index, stamp in enumerate(stamps)]
    (root/"frames.jsonl").write_text("".join(json.dumps(f)+"\n" for f in frames))
    rows = []
    for stamp in range(700_000_000, 2_100_000_001, 100_000_000):
        # Distinct lower-stage values must never appear in hold candidates.
        rows.extend([reference("arm", stamp, [0. if stamp < 1_600_000_000 else 1.]*4),
                     reference("gripper", stamp, [.02, .02])])
    (root/"controller_references.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))
    summary = json.loads((root/"summary.json").read_text())
    summary["frames"] = len(frames)
    summary["controller_references"]["saved"] = {"arm": len(rows)//2, "gripper": len(rows)//2}
    (root/"summary.json").write_text(json.dumps(summary))
    task = {"task_id": "test-task", "source": "cup_grasp", "execution_status": "finished",
            "task_success": True, "lift_verified": True, "clock": "ros_sim_time", "finish": "place"}
    events = []

    def event(stamp, kind, **data):
        events.append({"task_id": "test-task", "event_index": len(events),
                       "stamp_ns": stamp, "event": kind, **data})

    for index, phase in enumerate(("open", "pregrasp", "approach", "close", "lift")):
        start = 900_000_000+index*100_000_000
        event(start, "phase_started", phase=phase)
        if phase == "lift":
            event(1_400_000_000, "lift_verified")
        event(start+100_000_000 if phase != "lift" else 1_600_000_000, "phase_finished", phase=phase)
    event(1_600_000_000, "phase_started", phase="lower")
    event(1_900_000_000, "task_finished", execution_status="finished", task_success=True)
    (root/"task_metadata.json").write_text(json.dumps(task))
    (root/"task_events.jsonl").write_text("".join(json.dumps(e)+"\n" for e in events))


def test_hold_segment_keeps_source_indices_and_separate_files_without_lower_actions(tmp_path):
    from wheel_arm_manipulation.episode_segment import SEGMENT_FILE, SEGMENT_CANDIDATES_FILE
    write_segment_episode(tmp_path)
    (tmp_path/CANDIDATES_FILE).write_text("existing full-episode candidates\n")
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir() if p.is_file()}
    report = align_episode(tmp_path, task_segment="grasp-hold")
    assert report["segment"]["end_stamp_ns_exclusive"] == 1_600_000_000
    assert report["segment"]["selected_frames"] == 8
    assert report["candidate_frames"] == 4
    assert report["skipped_frames"]["segment_boundary"] == 4
    rows = [json.loads(l) for l in (tmp_path/SEGMENT_CANDIDATES_FILE).read_text().splitlines()]
    assert [r["source_frame_index"] for r in rows] == [0, 1, 2, 3]
    for row in rows:
        assert row["state_stamp_ns"]+400_000_000 < 1_600_000_000
        assert row["instruction"] == "拿起桌上的杯子并保持夹持"
        assert row["source_instruction"] == "移动"
        assert all(vector[:4] == [0.]*4 for vector in row["future_absolute_references"])
    assert json.loads((tmp_path/SEGMENT_FILE).read_text()) == report["segment"]
    assert all((tmp_path/name).read_bytes() == content for name, content in before.items())


def test_hold_interpolation_cannot_use_reference_at_or_after_lower_boundary():
    references = normalized_references([1_000_000_000, 1_500_000_000, 1_700_000_000],
                                      [[0.]*4, [0.]*4, [10.]*4], [[.02, .02]]*3)
    frames = [normalized_frame(1_000_000_000, [0.]*4+[.04]),
              normalized_frame(1_450_000_000, [0.]*4+[.04], 1)]
    rows, skipped = build_candidates(frames, references, chunk_size=1, chunk_step_seconds=.1,
                                    end_stamp_ns=1_600_000_000)
    assert len(rows) == 1
    # Query 1.55 s is inside the segment, but its upper interpolation support is outside.
    assert skipped["future_reference_coverage"] == 1


def test_time_grid_outputs_are_separate_and_source_files_are_unchanged(tmp_path):
    from wheel_arm_manipulation.time_grid import GRID_FILE, GRID_REPORT_FILE, resample_episode
    write_segment_episode(tmp_path)
    (tmp_path/CANDIDATES_FILE).write_text("existing candidates\n")
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir() if p.is_file()}
    report = resample_episode(tmp_path)
    assert report["grid_hz"] == 10
    assert report["semantics"] == "diagnostic_not_lerobot_dataset"
    assert json.loads((tmp_path/GRID_REPORT_FILE).read_text()) == report
    rows = [json.loads(l) for l in (tmp_path/GRID_FILE).read_text().splitlines()]
    assert len(rows) == report["grid_slots"]
    assert report["accepted_slots"]+report["rejected_slots"] == len(rows)
    assert all((tmp_path/name).read_bytes() == content for name, content in before.items())


def test_available_grid_preserves_nearest_outputs_and_raw_files(tmp_path):
    from wheel_arm_manipulation.time_grid import resample_episode, AVAILABLE_GRID_FILE, AVAILABLE_GRID_REPORT_FILE
    write_segment_episode(tmp_path)
    frames = [json.loads(line) for line in (tmp_path/"frames.jsonl").read_text().splitlines()]
    for frame in frames:
        frame["image_age_ms"] = frame["joints_age_ms"] = 10.
    (tmp_path/"frames.jsonl").write_text("".join(json.dumps(f)+"\n" for f in frames))
    resample_episode(tmp_path)
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir() if p.is_file()}
    report = resample_episode(tmp_path, image_policy="available-past")
    assert report["grid_file"] == AVAILABLE_GRID_FILE
    assert report["report_file"] == AVAILABLE_GRID_REPORT_FILE
    assert report["image_mode"] == "available-past"
    assert json.loads((tmp_path/AVAILABLE_GRID_REPORT_FILE).read_text()) == report
    assert (tmp_path/AVAILABLE_GRID_FILE).is_file()
    assert all((tmp_path/name).read_bytes() == content for name, content in before.items())


@pytest.mark.parametrize("invalid", ["failed", "unknown", "wrong_task", "missing_end", "outside_verification",
                                      "clock", "uncovered", "bad_terminal", "early_lower"])
def test_invalid_hold_task_or_events_are_rejected_without_outputs(tmp_path, invalid):
    from wheel_arm_manipulation.episode_segment import SEGMENT_FILE
    write_segment_episode(tmp_path)
    task = json.loads((tmp_path/"task_metadata.json").read_text())
    events = [json.loads(l) for l in (tmp_path/"task_events.jsonl").read_text().splitlines()]
    if invalid == "failed":
        task["execution_status"] = "failed"
    elif invalid == "unknown":
        task["task_success"] = None
    elif invalid == "wrong_task":
        events[0]["task_id"] = "other"
    elif invalid == "missing_end":
        events[10]["event"] = "unrelated"
    elif invalid == "outside_verification":
        events[7]["event"], events[9]["event"] = "lift_verified", "unrelated"
    elif invalid == "clock":
        task["clock"] = "ros_time"
    elif invalid == "uncovered":
        events[10]["stamp_ns"] = events[11]["stamp_ns"] = events[12]["stamp_ns"] = 3_000_000_000
    elif invalid == "bad_terminal":
        events[-1]["task_success"] = False
    else:
        events[7]["event"], events[7]["phase"] = "phase_started", "lower"
    (tmp_path/"task_metadata.json").write_text(json.dumps(task))
    (tmp_path/"task_events.jsonl").write_text("".join(json.dumps(e)+"\n" for e in events))
    with pytest.raises(ValueError):
        align_episode(tmp_path, task_segment="grasp-hold")
    assert not (tmp_path/SEGMENT_FILE).exists()
