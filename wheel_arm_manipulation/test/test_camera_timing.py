import json
from types import SimpleNamespace

import pytest

from wheel_arm_manipulation.camera_timing import TimingDiagnostics, gap_statistics, save_report
from wheel_arm_manipulation.camera_timing import timing_qos
from rclpy.qos import qos_profile_sensor_data, ReliabilityPolicy


def test_timing_qos_changes_only_reliability_for_controlled_comparison():
    best, reliable = timing_qos("best_effort"), timing_qos("reliable")
    assert best == qos_profile_sensor_data
    assert best.depth == reliable.depth == 5
    assert best.history == reliable.history
    assert best.durability == reliable.durability
    assert best.reliability == ReliabilityPolicy.BEST_EFFORT
    assert reliable.reliability == ReliabilityPolicy.RELIABLE


def test_invalid_timing_reliability_is_rejected():
    with pytest.raises(ValueError):
        timing_qos("unknown")


def test_regular_source_with_delayed_arrival_is_visible_as_paired_intervals():
    diagnostics = TimingDiagnostics()
    for source, arrival in [(1_000_000_000, 5_000_000_000),
                            (1_066_000_000, 5_330_000_000),
                            (1_132_000_000, 5_331_000_000)]:
        diagnostics.receive(source, arrival)
    report = diagnostics.report()
    assert report["positive_source_gaps"]["max_ms"] == 66.
    assert report["arrival_gaps"]["max_ms"] == 330.
    assert report["arrival_gaps"]["over_250ms"] == 1
    assert report["source_stamp_hz"] == pytest.approx(2/.132)
    assert diagnostics.rows[1]["source_gap_ns"] == 66_000_000
    assert diagnostics.rows[1]["arrival_gap_ns"] == 330_000_000


def test_source_and_arrival_gap_can_both_be_large():
    diagnostics = TimingDiagnostics()
    diagnostics.receive(1_000_000_000, 5_000_000_000)
    diagnostics.receive(1_462_000_000, 5_470_000_000)
    report = diagnostics.report()
    assert report["positive_source_gaps"]["max_ms"] == 462.
    assert report["largest_source_gap_pairs"][0]["arrival_gap_ns"] == 470_000_000


def test_invalid_duplicate_and_reset_timestamps_are_reported_without_negative_gap_statistics():
    diagnostics = TimingDiagnostics()
    for index, stamp in enumerate([0, 1_000_000_000, 1_000_000_000, 500_000_000, 566_000_000]):
        diagnostics.receive(stamp, 5_000_000_000+index*66_000_000)
    report = diagnostics.report()
    assert report["nonpositive_source_stamps"] == 1
    assert report["duplicate_source_stamps"] == 1
    assert report["backward_source_stamps"] == 1
    assert report["source_stamp_hz"] is None
    assert report["positive_source_gaps"]["count"] == 1
    assert report["positive_source_gaps"]["max_ms"] == 66.


@pytest.mark.parametrize("count", [0, 1])
def test_empty_or_single_message_has_no_invented_frequency(count):
    diagnostics = TimingDiagnostics()
    if count:
        diagnostics.receive(1_000_000_000, 5_000_000_000)
    report = diagnostics.report()
    assert report["messages"] == count
    assert report["arrival_wall_hz"] is report["source_stamp_hz"] is None
    assert report["arrival_gaps"]["max_ms"] is None


def test_monotonic_arrival_clock_cannot_move_backwards():
    diagnostics = TimingDiagnostics()
    diagnostics.receive(1_000, 5_000)
    with pytest.raises(ValueError, match="monotonic"):
        diagnostics.receive(2_000, 4_000)


def test_gap_thresholds_are_strict_and_percentile_has_explicit_small_sample_behavior():
    report = gap_statistics([100_000_000, 250_000_000, 251_000_000])
    assert report["over_100ms"] == 2
    assert report["over_250ms"] == 1
    assert report["p95_ms"] == 251.


def test_reports_preserve_raw_metadata_and_use_unique_directories(tmp_path):
    options = SimpleNamespace(output_dir=str(tmp_path), label="baseline", rgb_topic="/camera/image", seconds=30.)
    diagnostics = TimingDiagnostics()
    diagnostics.receive(1_000, 5_000)
    first, report = save_report(options, diagnostics, 30., "duration_reached")
    second, _ = save_report(options, diagnostics, 5., "interrupted")
    assert first != second
    assert json.loads((first/"summary.json").read_text()) == report
    assert report["label"] == "baseline"
    assert report["measurement"]["messages"] == 1
    assert [json.loads(l) for l in (first/"messages.jsonl").read_text().splitlines()] == diagnostics.rows
    assert set(p.name for p in first.iterdir()) == {"summary.json", "messages.jsonl"}


def test_report_records_selected_reliability(tmp_path):
    options = SimpleNamespace(output_dir=str(tmp_path), label="qos_reliable", rgb_topic="/camera/image",
                              seconds=30., reliability="reliable")
    _, report = save_report(options, TimingDiagnostics(), 30., "duration_reached")
    assert report["qos"] == {"reliability": "reliable", "history": "keep_last", "depth": 5,
                             "durability": "volatile"}
