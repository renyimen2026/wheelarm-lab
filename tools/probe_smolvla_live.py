"""Predict from current ROS observations; optional bounded simulation execution."""

import argparse
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate_smolvla_validation as validation
from ros_probe_session import RosProbeSession
from vla_trial_summary import trial_limits, lift_evidence, hold_evidence

forward, torch = validation.forward, validation.torch


def observation_batch(directory, instruction):
    directory = Path(directory)
    meta = forward.read_json(directory/"observation.json")
    if meta.get("kind") != "observation_only":
        raise ValueError("需要实时单帧观测")
    state = meta["state"]
    if state["names"] != list(forward.STATE_NAMES) or state["units"] != list(forward.STATE_UNITS):
        raise ValueError("状态关节顺序或单位不一致")
    values = np.asarray(state["values"], dtype=np.float32)
    if values.shape != (5,) or not np.isfinite(values).all() or not 0 <= values[4] <= .080002:
        raise ValueError("实际状态无效")
    for key in ("image_age_ms", "joints_age_ms", "timestamp_difference_ms"):
        if not np.isfinite(meta[key]) or meta[key] < 0:
            raise ValueError("观测时间无效")
    if max(meta["image_age_ms"], meta["joints_age_ms"]) > 500 or meta["timestamp_difference_ms"] > 50:
        raise ValueError("观测过旧或配对超差")
    path = directory/meta["image"]["file"]
    if not path.resolve().is_relative_to(directory.resolve()):
        raise ValueError("图像路径越界")
    with Image.open(path) as image:
        rgb = np.array(image.convert("RGB"), copy=True)
    if rgb.shape != (480, 640, 3):
        raise ValueError("实时图像尺寸与训练数据不同")
    return {forward.IMAGE_KEY: torch.from_numpy(rgb).permute(2, 0, 1).float()[None, None]/255,
            forward.STATE_KEY: torch.from_numpy(values)[None, None], "task": [instruction]}, meta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--assets-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--execute", action="store_true", help="仿真执行；默认最多三轮，--trial-seconds 开启连续试验；每轮仅用首步并限幅")
    parser.add_argument("--trial-seconds", type=float, help="连续仿真试验时长上限，10～600 秒；到时不再开始新动作")
    parser.add_argument("--action-steps", type=int, choices=(1,4,10), default=1, help="每轮执行预测前几步；默认 1，动作块对照选 10")
    parser.add_argument('--action-stride',type=int,choices=(1,5),default=1,
                        help='从10Hz模型未来轨迹每1或5步取样；保持真实0.1/0.5秒间隔，所有物理速度/加速度限制不变')
    parser.add_argument('--gripper-start-source',choices=('actual','reference'),default='actual',
                        help='reference从当前夹爪插值参考衔接，避免持杯时每轮重置到实际开口；不改模型目标及容限')
    parser.add_argument("--allow-valid-prefix", action="store_true", help="动作块中后续步骤不合法时，只执行连续合法前缀；首步不合法仍拒绝")
    parser.add_argument("--verify-every", type=int, default=0, help="每 N 轮用 RGB-D 核对杯子抬升，确认后停止；0 为仅结束时核对")
    parser.add_argument("--max-cycles", type=int, default=60, help="连续试验最多轮数，默认 60")
    parser.add_argument("--hold-seconds", type=float, default=0,
                        help="抬升确认后停止模型动作，至少观察10秒保持；0 为原有只核对抬升模式")
    parser.add_argument('--seed',type=int,default=20261108,help='动作采样随机种子；多轮验收应使用不同种子')
    parser.add_argument('--ensemble-size',type=int,choices=(1,4),default=1,
                        help='同一观测采样1或4条模型轨迹；4条取平均并保留原始输出，不调用专家')
    options = parser.parse_args()
    output = Path(options.output_dir).expanduser().resolve()
    session = None
    records, cup_before, cup_after = [], None, None
    verifications = []
    hold_samples, hold_result = [], None
    trial_started, trial_deadline = None, None
    end_reason, completed, failure = "not_started", False, None
    try:
        if output.exists():
            raise ValueError("输出目录已存在，不会覆盖")
        if options.hold_seconds != 0 and (not 10 <= options.hold_seconds <= 30
                or not options.verify_every or not options.execute):
            raise ValueError("保持验证需周期抬升核对和执行模式，保持10～30秒")
        if not 0 <= options.verify_every <= 10 or (options.verify_every and (not options.execute or options.trial_seconds is None)):
            raise ValueError("周期核对须连续执行模式，间隔 1～10 轮；0 表示不做周期核对")
        if options.allow_valid_prefix and (not options.execute or options.action_steps == 1):
            raise ValueError("合法前缀选项需要 --execute 和多步动作块")
        options.samples = trial_limits(options.execute, options.trial_seconds, options.max_cycles, options.samples)
        for source in (options.dataset, options.checkpoint, options.assets_dir):
            if output.is_relative_to(Path(source).expanduser().resolve()):
                raise ValueError("输出应在数据和模型目录之外")
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise ValueError("需要支持 BF16 的 CUDA GPU")
        torch.set_num_threads(1)
        checkpoint, config, data, stats, _ = validation.prepare_inputs(options.dataset, options.checkpoint, options.assets_dir)
        horizon=config.chunk_size
        if options.action_steps*options.action_stride>horizon:
            raise ValueError('请求的执行步数和时间跨度超过模型预测范围')
        selected_indices=list(range(options.action_stride-1,options.action_steps*options.action_stride,options.action_stride))
        checkpoint_hash=validation.checkpoint_digest(checkpoint/'model.safetensors')
        instruction = data.manifest["instruction"]
        print("[模型] 加载已训练权重；随后采集实时观测，执行模式由 --execute 明确指定。", flush=True)
        policy = forward.load_policy(checkpoint, config)
        policy.to("cuda").eval()
        config.device = "cuda"
        pre, post = forward.make_smolvla_pre_post_processors(config, dataset_stats=stats)
        output.mkdir(parents=True)
        print("[模式] 仿真短时执行；每轮重新观察，按选定步数限幅后执行；请停止其他运动任务。" if options.execute else "[模式] 只预测，不发送动作。", flush=True)
        if options.execute:
            session = RosProbeSession(output/"ros_session")
        if options.trial_seconds is not None:
            cup_before = session.request({"operation": "measure_cup", "require_support": True, "label": "trial_start"})["measurement"]
            print(f"[杯子初始证据] {cup_before}", flush=True)
        trial_started = time.monotonic()
        trial_deadline = trial_started+options.trial_seconds if options.trial_seconds is not None else None
        end_reason = "cycle_limit"
        for i in range(options.samples):
            if trial_deadline is not None and time.monotonic() >= trial_deadline:
                end_reason = "time_limit"
                break
            capture = output/f"capture_{i:02d}"
            started = time.monotonic()
            if session is not None:
                captured = session.request({"operation": "capture", "output_dir": str(capture), "instruction": instruction})
                observation_directory = Path(captured["observation_directory"])
            else:
                subprocess.run(["ros2", "run", "wheel_arm_manipulation", "observation_capture",
                                "--rgb-reliability", "reliable", "--output-dir", str(capture),
                                "--instruction", instruction, "--timeout", "15"], check=True, timeout=30)
                candidates = list(capture.glob("*/observation.json"))
                if len(candidates) != 1:
                    raise ValueError("本轮应恰好保存一组观测")
                observation_directory = candidates[0].parent
            batch, meta = observation_batch(observation_directory, instruction)
            state = batch[forward.STATE_KEY][0, 0].clone()
            processed = {k: v for k, v in pre(batch).items() if k != "action" and not k.startswith("action_")}
            if options.ensemble_size>1:
                processed={k:(v.expand(options.ensemble_size,*v.shape[1:])
                              if isinstance(v,torch.Tensor) and v.ndim and v.shape[0]==1
                              else v*options.ensemble_size if isinstance(v,list) and len(v)==1 else v)
                           for k,v in processed.items()}
            torch.cuda.synchronize()
            inference_start = time.monotonic()
            noises=[]
            for member in range(options.ensemble_size):
                generator = torch.Generator(device="cuda").manual_seed(options.seed+i+member*1000003)
                noises.append(torch.randn((1, horizon, 32), generator=generator, device="cuda"))
            noise=torch.cat(noises,dim=0)
            policy.reset()
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                members = post(policy.predict_action_chunk(processed, noise=noise)).float().cpu()
            torch.cuda.synchronize()
            elapsed = time.monotonic()-inference_start
            forward.require_tensor(members, (options.ensemble_size, horizon, 5), "实时预测")
            prediction=members.mean(0,keepdim=True)
            p = prediction[0]
            record = {"sample": i, "observation_directory": str(observation_directory),
                      "observation": meta, "instruction": instruction,
                      "state_physical": state.tolist(), "prediction_physical": p.tolist(),
                      "physical_action_offsets_seconds": [.1*(j+1) for j in range(horizon)],
                      "inference_seconds": elapsed,
                      "capture_process_and_prediction_wall_seconds": time.monotonic()-started,
                      "gripper_out_of_model_range_steps": int(((p[:, 4] < 0) | (p[:, 4] > .08)).sum()),
                      "first_action_minus_actual": (p[0]-state).tolist()}
            record.update(executed_prediction_indices=selected_indices,
                          executed_physical_offsets_seconds=[(j+1)/10 for j in selected_indices],
                          first_executed_action_minus_actual=(p[selected_indices[0]]-state).tolist())
            if options.ensemble_size>1:
                record.update(ensemble_predictions_physical=members.tolist(),
                              ensemble_standard_deviation=members.std(0,unbiased=False).tolist())
            forward.write_json(output/f"prediction_{i:02d}.json", record)
            if trial_deadline is not None and time.monotonic() >= trial_deadline:
                end_reason = "time_limit_before_motion"
                break
            if options.execute:
                request_path, result_path = output/f"motion_request_{i:02d}.json", output/f"motion_result_{i:02d}.json"
                forward.write_json(request_path, {"scope": "simulation_short_probe", "created_wall_seconds": time.time(),
                    "trial_deadline_monotonic": trial_deadline, "allow_valid_prefix":options.allow_valid_prefix,
                    "observation_stamp_ns": min(meta["joints_stamp_ns"], meta["image"]["stamp_ns"]), "state_physical": state.tolist(),
                    "first_prediction_physical": p[selected_indices[0]].tolist(),
                    "action_step_seconds":options.action_stride/10,
                    "gripper_start_source":options.gripper_start_source,
                    "predictions_physical": p[selected_indices].tolist()})
                request = forward.read_json(request_path)
                record["motion_session_result_file"] = str(session.root/f"result_{session.sequence:03d}.json")
                record["motion"] = {"status": "unknown", "execution_attempted": None}
                records.append(record)
                record["motion"] = session.request(dict(request, operation="motion"))
                forward.write_json(result_path, record["motion"])
            record["trial_elapsed_wall_seconds"] = time.monotonic()-trial_started
            if not options.execute:
                records.append(record)
            with (output/"cycles.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(forward.json.dumps(record, ensure_ascii=False, allow_nan=False)+"\n")
            if record.get("motion", {}).get("motion_skipped"):
                end_reason = "time_limit_before_dispatch"
                break
            if options.execute:
                motion = record["motion"]
                print(f"[运动反馈] TCP(m)={motion.get('gripper_tcp_base_m')}；开口={motion['actual_after'][4]*1000:.2f}mm", flush=True)
            print(f"[实时预测 {i+1}/{options.samples}] {elapsed:.3f}s；首步减实际={record['first_action_minus_actual']}；夹爪越界步数={record['gripper_out_of_model_range_steps']}", flush=True)
            if options.verify_every and (i+1)%options.verify_every == 0:
                cup_check = session.request({"operation":"measure_cup", "require_support":False,
                    "label":f"cycle_{i+1:03d}"})['measurement']
                evidence = lift_evidence(cup_before,cup_check)
                verification = {"completed_cycles":i+1,"measurement":cup_check,"evidence":evidence}
                verifications.append(verification)
                with (output/"verifications.jsonl").open("a",encoding="utf-8") as stream:
                    stream.write(forward.json.dumps(verification,ensure_ascii=False,allow_nan=False)+"\n")
                print(f"[周期抬升核对] {i+1}轮；{evidence}",flush=True)
                if evidence['lift_verified'] is True and (not options.hold_seconds
                        or abs(evidence['cup_top_delta_m']-.08)<=.01):
                    end_reason = "lift_geometry_verified"
                    break
        if end_reason == "lift_geometry_verified" and options.hold_seconds:
            hold_start = time.monotonic()
            for sample_index in range(9):
                deadline = hold_start+options.hold_seconds*sample_index/2
                while time.monotonic()<deadline:
                    time.sleep(max(0.,min(.05,deadline-time.monotonic())))
                captured = session.request({'operation':'capture',
                    'output_dir':str(output/f'hold_capture_{sample_index:02d}'),'instruction':instruction})
                _, hold_meta = observation_batch(captured['observation_directory'],instruction)
                measurement = session.request({'operation':'measure_cup','require_support':False,
                    'label':f'hold_{sample_index:02d}'})['measurement']
                hold_samples.append({'elapsed_seconds':time.monotonic()-hold_start,
                    'joints_stamp_ns':hold_meta['joints_stamp_ns'],
                    'measurement':measurement,'state':hold_meta['state']['values']})
                print(f"[保持核对] {hold_samples[-1]}",flush=True)
                if sample_index==0:
                    # Count a full duration after the first complete observation.
                    hold_start=time.monotonic()
                    hold_samples[0]['elapsed_seconds']=0.
                if (sample_index>=2 and hold_samples[-1]['joints_stamp_ns']-hold_samples[0]['joints_stamp_ns']
                        >=options.hold_seconds*1e9):
                    break
            hold_result=hold_evidence(cup_before,hold_samples,options.hold_seconds)
            end_reason='lift_and_hold_verified' if hold_result['task_success'] else 'hold_not_verified'
        if options.trial_seconds is not None:
            cup_after = session.request({"operation": "measure_cup", "require_support": False, "label": "trial_end"})["measurement"]
            final_evidence = lift_evidence(cup_before, cup_after)
            print(f"[杯子抬升证据] {final_evidence}", flush=True)
            if end_reason == "lift_geometry_verified" and final_evidence['lift_verified'] is not True:
                end_reason = "lift_evidence_not_maintained_after_stop"
            if hold_result and (final_evidence['lift_verified'] is not True
                    or abs(final_evidence.get('cup_top_delta_m',0)-.08)>.01):
                hold_result.update(task_success=False,reason='final_height_not_verified')
                end_reason='hold_not_verified'
        completed = True
        finished = sum(not r.get("motion", {}).get("motion_skipped", False) for r in records)
        print(f"[保存] {output}；结束原因={end_reason}；完成 {finished} 轮。", flush=True)
    except (OSError, ValueError, RuntimeError, KeyError, subprocess.SubprocessError) as error:
        end_reason, failure = "error", f"{type(error).__name__}: {error}"
        print(f"[实时预测失败] {type(error).__name__}: {error}", flush=True)
        return 1
    except KeyboardInterrupt:
        end_reason, failure = "interrupted", "user_interrupt"
        print("[停止] 请求取消；保留已完成轮次。", flush=True)
        return 130
    finally:
        try:
            if session is not None:
                session.close()
        finally:
            if output.is_dir() and trial_started is not None:
                for record in records:
                    path = record.get("motion_session_result_file")
                    if path and Path(path).is_file():
                        record["motion"] = forward.read_json(Path(path))
                sent = [r.get("motion", {}).get("execution_attempted", False) for r in records]
                forward.write_json(output/"live_report.json", {"status": "completed" if completed else "stopped",
                    "checkpoint": str(checkpoint), "scope": "bounded_slow_simulation_motion_probe" if options.execute else "live_observation_prediction_only",
                    "checkpoint_sha256":checkpoint_hash,
                    "end_reason": end_reason, "error": failure, "completed_cycles": sum(
                        (not options.execute or r.get("motion", {}).get("status") == "completed")
                        and not r.get("motion", {}).get("motion_skipped", False) for r in records),
                    "trial_seconds_limit": options.trial_seconds, "action_steps_per_cycle": options.action_steps,
                    "action_stride":options.action_stride,
                    "gripper_start_source":options.gripper_start_source,
                    "allow_valid_prefix":options.allow_valid_prefix, "verify_every_cycles":options.verify_every, "periodic_verifications":verifications,
                    "elapsed_including_final_measurement_seconds": time.monotonic()-trial_started,
                    "samples": records, "cup_before": cup_before, "cup_after": cup_after,
                    "lift_evidence": lift_evidence(cup_before, cup_after),
                    "hold_samples":hold_samples,"hold_evidence":hold_result,
                    "sampling_seed":options.seed,
                    "policy_chunk_size":horizon,
                    "ensemble_size":options.ensemble_size,
                    "task_success":bool(hold_result and hold_result['task_success']),
                    "robot_commands_sent": True if any(sent) else (None if any(v is None for v in sent) else False),
                    "training_started": False})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
