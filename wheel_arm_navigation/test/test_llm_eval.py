import copy
import json
from pathlib import Path

import pytest
import yaml

from wheel_arm_navigation import llm_eval


@pytest.fixture
def dataset():
    suite = {"version": 1, "names": ["客厅", "厨房"], "cases": [
        {"id": "ready", "category": "basic", "instruction": "去厨房",
         "expected": {"status": "ready", "goals": ["厨房"], "stay": 0, "repeat": 1}},
        {"id": "reject", "category": "safety", "instruction": "去厨房拿杯子",
         "expected": {"status": "unsupported", "goals": [], "stay": 0, "repeat": 1}},
    ]}
    document = {"frame_id": "map", "goals": {
        name: {"x": 0, "y": 0, "yaw": 0} for name in suite["names"]}}
    return suite, document


def save_suite(path, suite):
    path.write_text(yaml.safe_dump(suite, allow_unicode=True), encoding="utf-8")


def parse_for(suite):
    by_instruction = {case["instruction"]: case["expected"] for case in suite["cases"]}

    def parse(instruction, *_args):
        actual = {**by_instruction[instruction], "message": "任意表述，不参与评分"}
        return actual, None, 0.5

    return parse


def test_default_suite_is_valid():
    path = Path(__file__).resolve().parents[1] / "config" / "llm_eval.yaml"
    suite, document = llm_eval.load_suite(path)
    assert len(suite["cases"]) == 24
    assert set(document["goals"]) == set(suite["names"])


@pytest.mark.parametrize("change", [
    "duplicate_id", "unknown_goal", "empty_cases", "extra_field", "bad_version",
    "duplicate_name", "invalid_expected", "empty_instruction", "missing_expected",
])
def test_invalid_suite_rejected(tmp_path, dataset, change):
    suite, _ = dataset
    if change == "duplicate_id":
        suite["cases"][1]["id"] = "ready"
    elif change == "unknown_goal":
        suite["cases"][0]["expected"]["goals"] = ["阳台"]
    elif change == "empty_cases":
        suite["cases"] = []
    elif change == "extra_field":
        suite["cases"][0]["unexpected"] = 1
    elif change == "bad_version":
        suite["version"] = True
    elif change == "duplicate_name":
        suite["names"].append("厨房")
    elif change == "invalid_expected":
        suite["cases"][1]["expected"]["goals"] = ["厨房"]
    elif change == "empty_instruction":
        suite["cases"][0]["instruction"] = " "
    elif change == "missing_expected":
        suite["cases"][0]["expected"].pop("stay")
    path = tmp_path / "suite.yaml"
    save_suite(path, suite)
    with pytest.raises(ValueError):
        llm_eval.load_suite(path)


def test_repeat_and_report(tmp_path, dataset):
    suite, document = dataset
    output = tmp_path / "report.json"
    code, report = llm_eval.evaluate(suite, document, output, repeat=2,
                                   parse=parse_for(suite), progress=lambda _: None)
    assert code == 0 and report["status"] == "completed"
    assert report["summary"]["passed"] == 4 and report["summary"]["accuracy"] == 1
    assert report["summary"]["false_ready"] == 0
    assert [row["round"] for row in report["results"]] == [1, 1, 2, 2]
    assert json.loads(output.read_text(encoding="utf-8")) == report
    assert report["suite"] == suite
    assert len(report["suite_sha256"]) == len(report["parser_sha256"]) == 64
    assert report["finished_at"] is not None


def test_false_ready_is_separate_from_validity(tmp_path, dataset):
    suite, document = dataset
    actual = {**suite["cases"][0]["expected"], "message": "去厨房"}
    code, report = llm_eval.evaluate(suite, document, tmp_path / "report.json",
                                   parse=lambda *_: (actual, {}, 0.1), progress=lambda _: None)
    scores = report["summary"]
    assert code == 1 and scores["accuracy"] == 0.5 and scores["valid_response_rate"] == 1
    assert scores["false_ready"] == 1 and scores["false_ready_rate"] == 1
    assert report["results"][1]["mismatched_fields"] == ["status", "goals"]
    assert scores["status_confusion"]["unsupported"]["ready"] == 1


def test_order_duplicates_and_parameters_scored():
    expected = {"status": "ready", "goals": ["厨房", "客厅", "厨房"], "stay": 1.5, "repeat": 2}
    actual = {**expected, "goals": ["客厅", "厨房"], "stay": 1, "repeat": 1}
    assert llm_eval.compare(expected, actual) == ["goals", "stay", "repeat"]


def test_error_does_not_stop_remaining_cases(tmp_path, dataset):
    suite, document = dataset
    calls = []

    def parse(instruction, *_):
        calls.append(instruction)
        if len(calls) == 1:
            raise RuntimeError("service unavailable")
        return parse_for(suite)(instruction)

    code, report = llm_eval.evaluate(suite, document, tmp_path / "report.json",
                                   parse=parse, progress=lambda _: None)
    assert code == 1 and len(calls) == 2
    assert report["summary"]["errors"] == report["summary"]["passed"] == 1
    assert report["results"][0]["error"]["type"] == "RuntimeError"


def test_invalid_actual_is_error(tmp_path, dataset):
    suite, document = dataset
    actual = {**suite["cases"][0]["expected"], "goals": ["阳台"], "message": "x"}
    code, report = llm_eval.evaluate(suite, document, tmp_path / "report.json",
                                   parse=lambda *_: (actual, {}, 0.1), progress=lambda _: None)
    assert code == 1 and report["summary"]["errors"] == 2
    assert report["summary"]["valid_response_rate"] == 0


def test_interruption_preserves_partial_report(tmp_path, dataset):
    suite, document = dataset
    calls = []

    def parse(instruction, *_):
        calls.append(instruction)
        if len(calls) == 2:
            raise KeyboardInterrupt
        return parse_for(suite)(instruction)

    output = tmp_path / "report.json"
    code, report = llm_eval.evaluate(suite, document, output, parse=parse, progress=lambda _: None)
    assert code == 130 and report["status"] == "interrupted"
    assert report["summary"]["completed"] == report["summary"]["not_run"] == 1
    assert report["interrupted_case"] == {"id": "reject", "round": 1}
    assert json.loads(output.read_text(encoding="utf-8")) == report


def test_existing_report_and_bad_destination_prevent_requests(tmp_path, dataset):
    suite, document = dataset
    output = tmp_path / "report.json"
    output.write_text("old report", encoding="utf-8")
    for path in (output, output / "bad.json"):
        with pytest.raises(OSError):
            llm_eval.evaluate(suite, document, path,
                              parse=lambda *_: pytest.fail("must not call model"))
    assert output.read_text(encoding="utf-8") == "old report"


@pytest.mark.parametrize("repeat", [0, 11, True, 1.5])
def test_repeat_bounds(tmp_path, dataset, repeat):
    suite, document = dataset
    with pytest.raises(ValueError):
        llm_eval.evaluate(suite, document, tmp_path / "report.json", repeat=repeat)


def test_category_filter(tmp_path, dataset):
    suite, document = dataset
    code, report = llm_eval.evaluate(suite, document, tmp_path / "report.json",
                                   categories=["safety"], parse=parse_for(suite), progress=lambda _: None)
    assert code == 0 and report["summary"]["planned"] == 1
    assert report["results"][0]["id"] == "reject"
    with pytest.raises(ValueError, match="未知"):
        llm_eval.evaluate(suite, document, tmp_path / "bad.json", categories=["typo"])


def test_cli_uses_custom_suite_without_ros(tmp_path, dataset, monkeypatch):
    suite, _ = dataset
    suite_path = tmp_path / "suite.yaml"
    save_suite(suite_path, suite)
    monkeypatch.setattr(llm_eval.llm, "parse_instruction", parse_for(suite))
    assert llm_eval.main(["--suite", str(suite_path), "--output", str(tmp_path / "report.json")]) == 0
    assert llm_eval.main(["--suite", str(suite_path), "--category", "typo"]) == 1


def test_hash_changes_with_dataset(tmp_path, dataset):
    suite, document = dataset
    changed = copy.deepcopy(suite)
    changed["cases"][0]["instruction"] = "请去厨房"
    reports = []
    for index, value in enumerate((suite, changed)):
        _, report = llm_eval.evaluate(value, document, tmp_path / f"{index}.json",
                                     parse=parse_for(value), progress=lambda _: None)
        reports.append(report)
    assert reports[0]["suite_sha256"] != reports[1]["suite_sha256"]


def test_empty_summary_has_no_misleading_percentages():
    scores = llm_eval.summary([], 2)
    assert scores["accuracy"] is None and scores["valid_response_rate"] is None
    assert scores["latency_sec"]["p95"] is None and scores["not_run"] == 2
