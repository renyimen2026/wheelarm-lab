"""Fail-closed single navigation, stopped capture and image-question workflow."""

from pathlib import Path
import tempfile
import time

from wheel_arm_navigation.mission import Journal, check_goals, utc_now, validate_task
from wheel_arm_navigation.observation import save_rgb_capture
from wheel_arm_navigation.scene_describe import save_report
from wheel_arm_navigation import vision


def execute(task, document, goals_file, options, navigate, capture, infer,
            preflight, stopping, progress, source, preload_snapshot=None):
    if (not isinstance(task, dict) or set(task) != {"goal", "question"}
            or not isinstance(task["goal"], str) or not isinstance(task["question"], str)
            or not task["question"].strip() or len(task["question"]) > 500):
        raise ValueError("必须是单个精确地点和一个观察问题")
    plan = validate_task({"name": "navigate_observe", "goals": [task["goal"]],
                          "wait": options.nav_wait, "timeout": options.nav_timeout})
    check_goals(plan, document)
    journal = Journal(options.log_dir, plan, document, goals_file, source)
    observation = {"status": "not_run", "question": task["question"], "reason": "",
                   "directory": None, "report": None, "result": None}
    journal.data["observation"] = observation
    step = journal.data["steps"][0]
    journal.flush()
    progress(f"任务日志：{journal.path}")
    phase, status = "preflight", "preflight_failed"
    code = 1
    try:
        if stopping():
            raise KeyboardInterrupt()
        # Check output permissions and vision-model availability before sending a goal.
        root = Path(options.observations_dir).expanduser()
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryFile(dir=root):
            pass
        journal.data["vision_preflight"] = preflight()
        journal.flush()
        if stopping():
            raise KeyboardInterrupt()
        phase, status = "navigation", "failed"
        step.update(status="running", started_at=utc_now())
        journal.flush()
        started = time.monotonic()
        result = navigate(step["name"], step["pose"], document["frame_id"],
                          plan["wait"], plan["timeout"])
        step.update(status=result.status, reason=result.reason, action_status=result.action_status,
                    safe_to_continue=result.safe_to_continue, finished_at=utc_now(),
                    navigation_sec=round(time.monotonic() - started, 3))
        if preload_snapshot is not None:
            journal.data["vision_preload"] = preload_snapshot()
        journal.flush()
        if not result.safe_to_continue:
            status = "unsafe_stop"
            observation["reason"] = "导航终态未确认，不继续拍摄"
        elif stopping() or result.status == "canceled":
            raise KeyboardInterrupt()
        elif result.status != "succeeded":
            observation["reason"] = "导航未成功，不把当前位置画面当成目标地点"
        else:
            phase, status = "capture", "capture_failed"
            observation["status"] = "capturing"
            journal.flush()
            progress("导航成功，等待里程计停稳并检查拍摄时刻定位")
            image, metadata = capture(step["pose"], document["frame_id"])
            if stopping():
                raise KeyboardInterrupt()
            directory, metadata = save_rgb_capture(root, image, {**metadata, "mission_id": journal.data["mission_id"],
                                                                 "destination": task["goal"]})
            observation.update(status="captured", directory=str(directory.resolve()), metadata=metadata)
            journal.flush()
            phase, status = "vision", "vision_failed"
            observation["status"] = "reasoning"
            journal.flush()
            progress(f"画面已冻结：{directory / 'rgb.png'}；只回答当前视角，不判断整个房间")
            result, elapsed = infer(image, task["question"])
            if stopping():
                raise KeyboardInterrupt()
            vision.validate_response(result)
            report = {"schema_version": 1, "created_at": utc_now(), "mission_id": journal.data["mission_id"],
                      "observation_id": metadata["observation_id"], "image_sha256": vision.image_fingerprint(image),
                      "model": options.vision_model, "question": task["question"], "elapsed_sec": elapsed,
                      "source_mode": "after_navigation", "input_modalities": ["rgb"],
                      "image_stamps": metadata.get("stamps"), "result": result,
                      "limitations": ["仅当前单帧视角，没看到不等于房间不存在", "模型结果未经独立验证",
                                       "没有自主搜索、深度物体定位或抓取"]}
            report_path = save_report(directory / "descriptions", report)
            observation.update(status="completed", report=str(report_path), result=result, elapsed_sec=elapsed)
            status, code = "completed", 0
            progress("当前画面观察结果：" + result["summary"])
            progress("注意：单帧没看到某物，不代表目标地点不存在该物体。")
    except KeyboardInterrupt:
        status, code = "canceled", 130
        observation.update(status="canceled", reason=f"任务在 {phase} 阶段中断")
    except Exception as error:
        observation.update(status="failed" if phase != "navigation" else "not_run", reason=str(error))
        if phase == "navigation":
            status = "unsafe_stop"
            step.update(status="error", reason=str(error), safe_to_continue=False, finished_at=utc_now())
        progress(f"任务失败阶段：{phase}；{error}")
    finally:
        if preload_snapshot is not None:
            journal.data["vision_preload"] = preload_snapshot()
        journal.data.update(status=status, finished_at=utc_now(), final_phase=phase)
        journal.flush()
    progress(f"任务结束：{status}；日志：{journal.path}")
    return code, journal.path
