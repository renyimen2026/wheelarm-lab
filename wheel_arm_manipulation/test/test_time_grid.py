import pytest

from wheel_arm_manipulation.time_grid import (
    PERIOD_NS, bounded_interpolation, build_time_grid, nearest_image,
    pair_validation_stamp_ns, latest_available_image,
)


def inputs():
    frames, arm, gripper = [], [], []
    for index in range(20):
        stamp = 10_000_000_000+index*PERIOD_NS
        frames.append({"stamp_ns": stamp, "image_stamp_ns": stamp,
                       "actual": [index*.1, 0., 0., 0., .04],
                       "source": {"frame_index": 100+index}, "image_file": f"images/{100+index}.png"})
        arm.append({"stamp_ns": stamp, "positions": [index*.2, 0., 0., 0.]})
        gripper.append({"stamp_ns": stamp, "positions": [.02, .02]})
    segment = {"start_stamp_ns_inclusive": 10_000_000_000,
               "end_stamp_ns_exclusive": 12_000_000_000, "clock": "ros_sim_time",
               "instruction": "拿起桌上的杯子并保持夹持"}
    return frames, {"arm": arm, "gripper": gripper}, segment


def test_uniform_grid_action_shift_and_strict_end_boundary():
    frames, references, segment = inputs()
    rows, report = build_time_grid(frames, references, segment)
    assert len(rows) == 20
    assert [r["grid_index"] for r in rows] == list(range(20))
    assert all(b["stamp_ns"]-a["stamp_ns"] == PERIOD_NS for a, b in zip(rows, rows[1:]))
    assert report["accepted_slots"] == 16
    assert report["rejected_slots"] == 4
    assert report["valid_contiguous_runs"] == [{"first_grid_index": 0, "last_grid_index": 15, "samples": 16}]
    assert report["contiguous_chunk_start_slots"] == 13
    assert rows[0]["state"] == [0., 0., 0., 0., .04]
    assert rows[0]["action"] == pytest.approx([.2, 0., 0., 0., .04])
    assert rows[0]["future_absolute_references"][-1] == pytest.approx([.8, 0., 0., 0., .04])
    assert rows[0]["image"]["source_frame_index"] == 100
    assert rows[16]["rejection_reasons"] == ["action_segment_boundary"]
    assert all(r["future_stamp_ns"][-1] < 12_000_000_000 for r in rows if r["status"] == "accepted")


def test_irregular_state_is_interpolated_from_real_support():
    frames, references, segment = inputs()
    # Move one state sample to 10.12 s; interpolation at 10.10 s should still give q=.1.
    frames[1]["stamp_ns"] += 20_000_000
    frames[1]["actual"][0] = .12
    rows, _ = build_time_grid(frames, references, segment)
    assert rows[1]["state"][0] == pytest.approx(.1)
    assert rows[1]["state_support"]["gap_ms"] == 120.
    assert rows[1]["state_support"]["source_frame_indices"] == [100, 101]


def test_missing_observations_keep_grid_slots_and_break_contiguous_runs():
    frames, references, segment = inputs()
    frames = [f for index, f in enumerate(frames) if index not in (5, 6)]
    rows, report = build_time_grid(frames, references, segment)
    assert len(rows) == 20
    for index in (5, 6):
        assert rows[index]["status"] == "rejected"
        assert set(rows[index]["rejection_reasons"]) == {"image_error", "state_gap"}
    assert report["rejection_reason_counts"]["state_gap"] == 2
    assert report["valid_contiguous_runs"] == [
        {"first_grid_index": 0, "last_grid_index": 4, "samples": 5},
        {"first_grid_index": 7, "last_grid_index": 15, "samples": 9},
    ]
    assert report["contiguous_chunk_start_slots"] == 8


def test_nearest_image_ties_choose_earlier_and_report_future_selection():
    frames, references, segment = inputs()
    for frame in frames:
        frame["image_stamp_ns"] += 50_000_000
    assert nearest_image(frames, 10_100_000_000)["source"]["frame_index"] == 100
    rows, report = build_time_grid(frames, references, segment)
    assert rows[0]["status"] == "accepted"  # Exactly 50 ms is allowed.
    assert rows[0]["image"]["offset_from_grid_ms"] == 50.
    assert rows[1]["image"]["offset_from_grid_ms"] == -50.
    assert report["accepted_future_image_slots"] == 1
    assert report["accepted_reused_image_slots"] == 1


def test_reference_gaps_are_rejected_even_when_images_and_state_are_available():
    frames, references, segment = inputs()
    references["arm"] = [r for r in references["arm"] if r["stamp_ns"] != 10_500_000_000]
    rows, report = build_time_grid(frames, references, segment)
    assert "arm_reference_gap" in rows[1]["rejection_reasons"]
    assert report["rejection_reason_counts"]["arm_reference_gap"] == 4


def test_reference_support_after_segment_cannot_supply_an_in_range_action():
    frames, references, segment = inputs()
    references["arm"][-1]["stamp_ns"] = 12_000_000_000
    references["arm"][-1]["positions"] = [999.]*4
    rows, _ = build_time_grid(frames, references, segment)
    assert "arm_reference_coverage" in rows[15]["rejection_reasons"]
    assert rows[14]["status"] == "accepted"


def test_state_extrapolation_is_rejected_at_leading_grid_slot():
    frames, references, segment = inputs()
    frames[0]["stamp_ns"] += 10_000_000
    rows, _ = build_time_grid(frames, references, segment)
    assert "state_coverage" in rows[0]["rejection_reasons"]
    assert rows[1]["status"] == "accepted"


def test_interpolation_gap_limit_and_exact_sample_semantics():
    rows = [{"stamp_ns": 100, "positions": [1.]}, {"stamp_ns": 400, "positions": [4.]}]
    with pytest.raises(LookupError, match="gap"):
        bounded_interpolation(rows, 200, 250)
    assert bounded_interpolation(rows, 100, 250)[0] == [1.]
    with pytest.raises(LookupError, match="coverage"):
        bounded_interpolation(rows, 500, 250)


@pytest.mark.parametrize("chunk_size", [0, -1, 1.5, True])
def test_invalid_chunk_size_is_rejected(chunk_size):
    with pytest.raises(ValueError):
        build_time_grid(*inputs(), chunk_size=chunk_size)


def available_inputs(age_ms=30.):
    frames, references, segment = inputs()
    for frame in frames:
        frame["source"]["image_age_ms"] = age_ms
        frame["source"]["joints_age_ms"] = age_ms
    return frames, references, segment


def test_available_past_never_uses_image_before_its_recorded_pair_validation():
    rows, report = build_time_grid(*available_inputs(), image_policy="available-past")
    assert rows[0]["image"] is None
    assert rows[0]["rejection_reasons"] == ["image_unavailable"]
    # The image at 10.1 s is already captured but is not processed until 10.13 s.
    assert rows[1]["image"]["source_frame_index"] == 100
    assert rows[1]["image"]["pair_validation_stamp_ns"] == 10_030_000_000
    assert rows[1]["image"]["age_at_grid_ms"] == 100.
    assert rows[1]["state"][0] == pytest.approx(.1)
    assert rows[1]["action"][0] == pytest.approx(.4)
    assert report["accepted_slots"] == 15
    assert report["valid_contiguous_runs"] == [{"first_grid_index": 1, "last_grid_index": 15, "samples": 15}]
    assert report["accepted_future_image_slots"] == report["accepted_future_availability_slots"] == 0
    for row in rows:
        if row["status"] == "accepted":
            assert row["image"]["stamp_ns"] <= row["image"]["pair_validation_stamp_ns"] <= row["stamp_ns"]
            assert row["future_stamp_ns"][-1] < 12_000_000_000


def test_available_past_does_not_wait_for_a_fresh_image_or_compress_a_stale_slot():
    frames, references, segment = available_inputs()
    frames[1]["source"]["image_age_ms"] = frames[1]["source"]["joints_age_ms"] = 150.
    rows, _ = build_time_grid(frames, references, segment, image_policy="available-past")
    # At 10.2 s, the latest processed image is still the 10.0 s image: 200 ms old.
    assert rows[2]["grid_index"] == 2
    assert rows[2]["image"]["source_frame_index"] == 100
    assert rows[2]["image"]["age_at_grid_ms"] == 200.
    assert rows[2]["rejection_reasons"] == ["image_error"]
    assert rows[3]["status"] == "accepted"


def test_exact_pair_validation_time_is_available_and_source_recency_wins_over_arrival_recency():
    frames, _, _ = available_inputs()
    a, b = frames[:2]
    selected, stamp = latest_available_image([(a, 10_200_000_000), (b, 10_150_000_000)], 10_200_000_000)
    assert selected is b
    assert stamp == 10_150_000_000
    selected, stamp = latest_available_image([(b, 10_150_000_000)], 10_150_000_000)
    assert selected is b


@pytest.mark.parametrize("age", [None, -1., float("nan"), float("inf"), 501., True])
def test_missing_or_invalid_pair_age_is_rejected_without_timestamp_fallback(age):
    frames, references, segment = available_inputs()
    frames[0]["source"]["image_age_ms"] = age
    with pytest.raises(ValueError, match="image_age_ms"):
        build_time_grid(frames, references, segment, image_policy="available-past")


def test_image_and_joint_ages_must_describe_same_node_time():
    frame = available_inputs()[0][0]
    frame["source"]["joints_age_ms"] = 40.
    with pytest.raises(ValueError, match="same pair validation"):
        pair_validation_stamp_ns(frame)


def test_pair_validation_time_cannot_precede_joint_sample():
    frame = available_inputs()[0][0]
    frame["stamp_ns"] += 40_000_000
    with pytest.raises(ValueError, match="precede"):
        pair_validation_stamp_ns(frame)


def test_unknown_image_policy_is_rejected():
    with pytest.raises(ValueError, match="Image policy"):
        build_time_grid(*inputs(), image_policy="unknown")
