"""User-authorized batch02 simulation collection. No model training or upload.

Owns only the subprocesses it starts. Each slot gets a fresh simulation, an
exclusive recording directory, MoveIt planning, RGB-D lift/hold verification,
and causal time-grid diagnostics. Failed attempts are retained, never relabeled.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "wheel_arm_manipulation"))
INSTRUCTION = "拿起桌上的杯子并保持夹持"
SLOTS = [
    ("train", [-1.05, 2.10, -1.05], .060),
    ("train", [-1.02, 2.04, -1.02], .070),
    ("train", [-1.08, 2.16, -1.08], .060),
    ("train", [-1.00, 2.08, -1.08], .070),
    ("train", [-1.08, 2.08, -1.00], .080),
    ("train", [-1.02, 2.12, -1.10], .060),
    ("independent_test", [-1.04, 2.08, -1.04], .065),
    ("independent_test", [-1.06, 2.12, -1.06], .075),
]


def save(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def worker(mode, width, episode):
    import rclpy
    from rclpy.signals import SignalHandlerOptions
    from wheel_arm_manipulation import arm_goal, cup_grasp
    from wheel_arm_manipulation.planning import load_model, values_from_state
    from wheel_arm_manipulation.task_metadata import run_with_metadata
    rclpy.init(args=["collector", "--ros-args", "-p", "use_sim_time:=true"],
                signal_handler_options=SignalHandlerOptions.NO)
    if mode == "prepare":
        options = arm_goal.parser().parse_args([
            "gripper", "width", "--width", str(width), "--execute", "--wait", "60"])
        node = arm_goal.ArmGoalClient(options, load_model("gripper"))
    else:
        options = cup_grasp.parser().parse_args([
            "--execute", "--finish", "hold", "--hold-seconds", "10", "--wait", "30",
            "--episode-dir", str(episode), "--output-dir", str(episode / "cup_evidence")])
        node = cup_grasp.CupGraspClient(options)

    def stop(*_):
        node.stopping = True
        if not node.execution_pending:
            raise KeyboardInterrupt()
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    try:
        if mode == "prepare":
            from rclpy.action import ActionClient
            from control_msgs.action import FollowJointTrajectory
            clients = [ActionClient(node, FollowJointTrajectory,
                       f"/wheel_arm/{name}_controller/follow_joint_trajectory")
                       for name in ("arm", "gripper")]
            try:
                readiness_deadline=time.monotonic()+60.
                for client in clients:
                    remaining=readiness_deadline-time.monotonic()
                    if remaining<=0 or not client.wait_for_server(timeout_sec=remaining):
                        raise RuntimeError("控制器 action 服务未在共同60秒启动期限内就绪")
                discovery_until = time.monotonic()+2
                while time.monotonic() < discovery_until:
                    rclpy.spin_once(node, timeout_sec=.05)
            finally:
                for client in clients:
                    client.destroy()
            current = node.snapshot()
            values = values_from_state(current, node.model)
            if all(abs(value-width/2) <= .0002 for value in values.values()):
                print("[初始开口] 已到达，无需重复发送零运动轨迹。", flush=True)
            else:
                trajectory, start, goal = node.plan(current)
                node.execute(trajectory, start, goal)
        else:
            # Invoking this collector requests autonomous simulated demonstrations.
            return run_with_metadata(node, options, cup_grasp.run, lambda: True)
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return 0


def launch(command, log):
    return subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)


def stop(child, timeout=40):
    if child is not None and child.poll() is None:
        os.killpg(child.pid, signal.SIGINT)
        child.wait(timeout=timeout)


def run_child(command, log_path, timeout):
    with log_path.open("x") as log:
        child = launch(command, log)
        try:
            code = child.wait(timeout=timeout)
            if code:
                raise RuntimeError(f"exit={code}: {log_path}")
        finally:
            stop(child)


def attempt(root, number, retry, q, width):
    directory = root / f"slot_{number:02d}" / f"attempt_{retry:02d}"
    directory.mkdir(parents=True)
    # Per-attempt discovery isolation avoids stale endpoints across fast resets.
    # These variables affect only this collector and its owned subprocesses.
    os.environ["ROS_DOMAIN_ID"] = str(90+number*5+retry)
    os.environ["IGN_PARTITION"] = f"wheelarm_batch02_{number}_{retry}"
    save(directory / "runtime.json", {"ros_domain_id": os.environ["ROS_DOMAIN_ID"],
                                      "ign_partition": os.environ["IGN_PARTITION"]})
    simulation = capture = None
    sim_log = (directory / "simulation.log").open("x")
    capture_log = (directory / "capture.log").open("x")
    try:
        simulation = launch([
            "ros2", "launch", "wheel_arm_manipulation", "cup_test.launch.py",
            "camera_mount:=mast", "cup_model:=ceramic_cup", "gui:=false", "rviz:=false",
            f"initial_shoulder_lift:={q[0]}", f"initial_elbow:={q[1]}",
            f"initial_wrist_pitch:={q[2]}"], sim_log)
        run_child([sys.executable, "-s", str(Path(__file__).resolve()), "--worker", "prepare",
                   "--width", str(width)], directory / "prepare.log", 110)
        recordings = directory / "recordings"
        capture = launch(["ros2", "run", "wheel_arm_manipulation", "observation_capture",
                          "--record", "--record-references", "--hz", "16",
                          "--rgb-reliability", "reliable", "--instruction", INSTRUCTION,
                          "--output-dir", str(recordings)], capture_log)
        deadline = time.monotonic() + 35
        episode = None
        while time.monotonic() < deadline:
            files = list(recordings.glob("*/frames.jsonl")) if recordings.exists() else []
            if len(files) == 1 and files[0].stat().st_size > 0:
                episode = files[0].parent
                break
            if capture.poll() is not None:
                raise RuntimeError(f"采集提前退出: {directory / 'capture.log'}")
            time.sleep(.1)
        if episode is None:
            raise RuntimeError("等待首帧超时")
        print(f"[采集] slot={number} retry={retry} episode={episode}", flush=True)
        run_child([sys.executable, "-s", str(Path(__file__).resolve()), "--worker", "grasp",
                   "--episode", str(episode)], directory / "grasp.log", 220)
        time.sleep(1)
        stop(capture)
        metadata = json.loads((episode / "task_metadata.json").read_text())
        summary = json.loads((episode / "summary.json").read_text())
        events = [json.loads(x) for x in (episode / "task_events.jsonl").read_text().splitlines()]
        if (metadata.get("task_success") is not True or not metadata.get("lift_verified")
                or not any(e["event"] == "hold_verified" for e in events)
                or summary["actual_hz"] < 14 or summary["diagnostics"]["counts"]["conversion_failed"]):
            raise RuntimeError("抓起保持验证或采集质量未通过")
        run_child(["ros2", "run", "wheel_arm_manipulation", "action_alignment", str(episode),
                   "--task-segment", "grasp-hold", "--chunk-size", "4", "--chunk-step", ".1",
                   "--time-grid", "--image-policy", "available-past"], directory / "alignment.log", 100)
        report = json.loads((episode / "grasp_hold_available_time_grid_report.json").read_text())
        runs = [r["samples"] for r in report["valid_contiguous_runs"]]
        if report["accepted_slots"] / report["grid_slots"] < .97 or max(runs, default=0) < 300:
            raise RuntimeError("时间网格连续性未达到本轮标准")
        return {"episode_dir": str(episode), "task_id": metadata["task_id"],
                "frames": summary["frames"], "actual_hz": summary["actual_hz"],
                "accepted_grid_slots": report["accepted_slots"], "run_lengths": runs,
                "complete_50_step_starts": sum(max(0, n-49) for n in runs)}
    finally:
        try:
            stop(capture)
        finally:
            stop(simulation)
            # ros2 launch may exit after terminating its Ruby wrapper while an
            # owned Gazebo grandchild remains. It shares our explicit group.
            if simulation is not None:
                try:
                    os.killpg(simulation.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            sim_log.close()
            capture_log.close()
        time.sleep(2)


def collect(root, resume=False, slots=None):
    slots = SLOTS if slots is None else slots
    if (len(slots)!=8 or [s[0] for s in slots]!=['train']*6+['independent_test']*2
            or any(len(q)!=3 or not .05<=w<=.08 for _,q,w in slots)):
        raise ValueError('采集计划要求六条训练、两条独立测试，三关节初始姿态和50～80mm开口')
    plan = {"schema_version": 1, "instruction": INSTRUCTION,
         "capture_hz": 16, "training_grid_hz": 10, "hold_seconds": 10,
         "scope": "fixed_cup_and_camera_small_initial_arm_and_opening_variations",
         "slots": [{"slot": i, "split": split, "initial_arm_q": q, "initial_width_m": w}
                   for i, (split, q, w) in enumerate(slots, 1)]}
    if resume:
        if json.loads((root / "collection_plan.json").read_text()) != plan:
            raise ValueError("恢复时计划不一致")
    else:
        root.mkdir(parents=True, exist_ok=False)
        save(root / "collection_plan.json", plan)
    accepted = []
    for number, (split, q, width) in enumerate(slots, 1):
        accepted_file = root / f"accepted_{number:02d}.json"
        if resume and accepted_file.exists():
            entry = json.loads(accepted_file.read_text())
            metadata = json.loads((Path(entry['episode_dir']) / 'task_metadata.json').read_text())
            if metadata['task_id'] != entry['task_id'] or metadata['task_success'] is not True:
                raise ValueError("恢复的成功记录已变化")
            accepted.append(entry)
            continue
        previous = list((root / f"slot_{number:02d}").glob("attempt_*"))
        first = max((int(p.name.split('_')[-1]) for p in previous), default=0)+1
        for retry in (first, first+1):
            try:
                entry = attempt(root, number, retry, q, width)
            except (RuntimeError, subprocess.TimeoutExpired) as error:
                print(f"[保留失败记录] slot={number} retry={retry}: {error}", flush=True)
                save(root / f"slot_{number:02d}" / f"attempt_{retry:02d}" / "failure.json",
                     {"status": "rejected", "reason": str(error)})
                if retry == first+1:
                    raise
            else:
                entry.update(slot=number, split=split, initial_arm_q=q, initial_width_m=width)
                accepted.append(entry)
                save(root / f"accepted_{number:02d}.json", entry)
                print(f"[通过 {number}/8] {json.dumps(entry, ensure_ascii=False)}", flush=True)
                break
    save(root / "collection_report.json", {"status": "verified", "episodes": accepted,
         "training_started": False, "uploaded": False})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--plan", type=Path, help="采集前固定的八条初始状态及集合划分 JSON")
    parser.add_argument("--worker", choices=("prepare", "grasp"))
    parser.add_argument("--width", type=float, default=.06)
    parser.add_argument("--episode", type=Path)
    options = parser.parse_args()
    if options.worker:
        raise SystemExit(worker(options.worker, options.width, options.episode))
    if options.root is None:
        parser.error("需要 --root；此入口会自主执行仿真示范采集")
    slots = None
    if options.plan:
        slots = [(s['split'],s['initial_arm_q'],s['initial_width_m'])
                 for s in json.loads(options.plan.read_text())['slots']]
    collect(options.root, options.resume, slots)
