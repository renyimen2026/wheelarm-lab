"""Repeatable parser-only evaluations; never import or initialize ROS navigation."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics
import sys
import tempfile
import time
from uuid import uuid4

import yaml

from wheel_arm_navigation import llm
from wheel_arm_navigation.mission import utc_now


FIELDS = ("status", "goals", "stay", "repeat")


def load_suite(path):
    with Path(path).expanduser().open(encoding="utf-8") as stream:
        try:
            suite = yaml.safe_load(stream)
        except yaml.YAMLError as error:
            raise ValueError(f"评测 YAML 格式错误：{error}") from error
    if (not isinstance(suite, dict) or set(suite) != {"version", "names", "cases"}
            or type(suite["version"]) is not int or suite["version"] != 1):
        raise ValueError("评测集必须包含 version: 1、names、cases")
    names = suite["names"]
    if (not isinstance(names, list) or not names
            or any(not isinstance(name, str) or not name.strip() for name in names)
            or len(set(names)) != len(names)):
        raise ValueError("names 必须是非空且不重复的目标名称列表")
    # Poses are placeholders only: the parser consumes names, not coordinates.
    document = {"frame_id": "map", "goals": {
        name: {"x": 0, "y": 0, "yaw": 0} for name in names}}
    cases = suite["cases"]
    if not isinstance(cases, list) or not cases:
        raise ValueError("cases 必须是非空测试列表")
    identifiers = set()
    for case in cases:
        if (not isinstance(case, dict)
                or set(case) != {"id", "category", "instruction", "expected"}):
            raise ValueError("每条用例必须包含 id、category、instruction、expected")
        for key in ("id", "category", "instruction"):
            if not isinstance(case[key], str) or not case[key].strip():
                raise ValueError(f"用例 {key} 必须为非空字符串")
        if len(case["instruction"]) > 1000:
            raise ValueError(f"{case['id']}：指令超过 1000 字")
        if case["id"] in identifiers:
            raise ValueError(f"用例 id 重复：{case['id']}")
        identifiers.add(case["id"])
        expected = case["expected"]
        if not isinstance(expected, dict) or set(expected) != set(FIELDS):
            raise ValueError(f"{case['id']}：expected 必须包含 {FIELDS}")
        llm.validate_response({**expected, "message": "评测预期结果"}, document)
    return suite, document


def compare(expected, actual):
    return [key for key in FIELDS if expected[key] != actual[key]]


def summary(rows, planned):
    passed = sum(row["outcome"] == "passed" for row in rows)
    errors = sum(row["outcome"] == "error" for row in rows)
    rejected = [row for row in rows if row["expected"]["status"] != "ready"]
    false_ready = sum(row["actual"] is not None and row["actual"]["status"] == "ready"
                      for row in rejected)
    latencies = sorted(row["elapsed_sec"] for row in rows)
    categories = {}
    matrix = {}
    for row in rows:
        category = categories.setdefault(row["category"], {"total": 0, "passed": 0})
        category["total"] += 1
        category["passed"] += row["outcome"] == "passed"
        expected_status = row["expected"]["status"]
        actual_status = row["actual"]["status"] if row["actual"] is not None else "error"
        counts = matrix.setdefault(expected_status, {})
        counts[actual_status] = counts.get(actual_status, 0) + 1
    for value in categories.values():
        value["accuracy"] = value["passed"] / value["total"]
    return {
        "planned": planned, "completed": len(rows), "not_run": planned - len(rows),
        "passed": passed, "mismatched": len(rows) - passed - errors, "errors": errors,
        "accuracy": passed / len(rows) if rows else None,
        "valid_response_rate": (len(rows) - errors) / len(rows) if rows else None,
        "expected_non_ready": len(rejected), "false_ready": false_ready,
        "false_ready_rate": false_ready / len(rejected) if rejected else None,
        "latency_sec": {
            "mean": statistics.mean(latencies) if rows else None,
            "median": statistics.median(latencies) if rows else None,
            "p95": latencies[math.ceil(0.95 * len(rows)) - 1] if rows else None,
        },
        "categories": categories, "status_confusion": matrix,
    }


def write_report(path, report):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=path.parent,
                prefix=".llm-eval-", suffix=".json", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def evaluate(suite, document, output, model="qwen3:4b",
             endpoint="http://127.0.0.1:11434", timeout=180, repeat=1,
             categories=None, parse=None, progress=print):
    if type(repeat) is not int or not 1 <= repeat <= 10:
        raise ValueError("评测重复轮数必须是 1 至 10 的整数")
    selected = set(categories or [])
    unknown = selected - {case["category"] for case in suite["cases"]}
    if unknown:
        raise ValueError(f"未知评测类别：{sorted(unknown)}")
    cases = [case for case in suite["cases"] if not selected or case["category"] in selected]
    output = Path(output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": 1, "started_at": utc_now(), "finished_at": None,
        "status": "running", "model": model, "endpoint": endpoint,
        "llm_timeout_sec": timeout, "repeat": repeat,
        "selected_categories": sorted(selected), "python_version": platform.python_version(),
        "suite": suite,
        "suite_sha256": hashlib.sha256(json.dumps(
            suite, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest(),
        "parser_sha256": hashlib.sha256(Path(llm.__file__).read_bytes()).hexdigest(),
        "results": [], "summary": summary([], len(cases) * repeat),
    }
    # Reserve the destination before model calls; never overwrite an earlier run.
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    parse = llm.parse_instruction if parse is None else parse
    code = 0
    try:
        progress(f"评测报告：{output}；共 {len(cases) * repeat} 次请求，不启动导航")
        for round_number in range(1, repeat + 1):
            for case in cases:
                row = {**case, "round": round_number, "actual": None, "plan": None,
                       "outcome": "error", "mismatched_fields": [], "error": None,
                       "elapsed_sec": 0.0, "parse_sec": None}
                started = time.monotonic()
                try:
                    actual, plan, elapsed = parse(case["instruction"], document,
                                                  model, endpoint, timeout)
                    llm.validate_response(actual, document)
                    row.update(actual=actual, plan=plan, parse_sec=elapsed,
                               mismatched_fields=compare(case["expected"], actual))
                    row["outcome"] = "mismatched" if row["mismatched_fields"] else "passed"
                except (ValueError, RuntimeError, OSError) as error:
                    row["error"] = {"type": type(error).__name__, "message": str(error)}
                except KeyboardInterrupt:
                    report["interrupted_case"] = {"id": case["id"], "round": round_number}
                    raise
                row["elapsed_sec"] = round(time.monotonic() - started, 3)
                report["results"].append(row)
                report["summary"] = summary(report["results"], len(cases) * repeat)
                write_report(output, report)
                detail = (f"不一致字段：{','.join(row['mismatched_fields'])}"
                          if row["outcome"] == "mismatched" else
                          row["error"]["message"] if row["error"] else "")
                progress(f"[{len(report['results'])}/{len(cases) * repeat}] "
                         f"{case['id']} {row['outcome']} {row['elapsed_sec']:g}s {detail}")
        report["status"] = "completed"
        code = 1 if report["summary"]["passed"] != report["summary"]["planned"] else 0
    except KeyboardInterrupt:
        report["status"] = "interrupted"
        code = 130
    except BaseException:
        report["status"] = "error"
        raise
    finally:
        report["finished_at"] = utc_now()
        write_report(output, report)
    return code, report


def parser():
    result = argparse.ArgumentParser(description="本地 LLM 指令解析评测，不执行导航")
    result.add_argument("--suite", help="自定义评测 YAML，默认使用随包安装的评测集")
    result.add_argument("--model", default="qwen3:4b")
    result.add_argument("--endpoint", default="http://127.0.0.1:11434")
    result.add_argument("--llm-timeout", type=float, default=180)
    result.add_argument("--repeat", type=int, default=1, help="整套评测重复轮数，1 至 10")
    result.add_argument("--category", action="append", help="只评测指定类别，可以重复指定")
    result.add_argument("--output", help="JSON 报告路径，必须不存在")
    return result


def main(args=None):
    options = parser().parse_args(args)
    try:
        if options.suite is None:
            from ament_index_python.packages import get_package_share_directory
            options.suite = str(Path(get_package_share_directory("wheel_arm_navigation"))
                                / "config" / "llm_eval.yaml")
        suite, document = load_suite(options.suite)
        identifier = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
        output = options.output or str(Path("evaluation_logs") / (identifier + ".json"))
        code, report = evaluate(suite, document, output, options.model, options.endpoint,
                                options.llm_timeout, options.repeat, options.category)
        scores = report["summary"]
        if scores["completed"]:
            print(f"正确 {scores['passed']}/{scores['completed']} ({scores['accuracy']:.1%})；"
                  f"格式合法率 {scores['valid_response_rate']:.1%}；"
                  f"误接受 {scores['false_ready']}/{scores['expected_non_ready']}；"
                  f"错误 {scores['errors']}；P95 {scores['latency_sec']['p95']:g}s")
        print(f"评测状态：{report['status']}；未完成 {scores['not_run']}；报告：{output}")
        return code
    except (OSError, ValueError, RuntimeError) as error:
        print(f"评测错误：{error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
