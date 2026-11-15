"""Evaluate every complete validation window of a saved local checkpoint."""

import argparse
import hashlib
import numpy as np
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_smolvla_batch as pilot
from wheel_arm_manipulation.smolvla_batch_data import load_batch, SplitData
from wheel_arm_manipulation.lerobot_conversion import digest
from wheel_arm_manipulation.task_phases import parse_phases, phase_at

forward, single, torch = pilot.forward, pilot.single, pilot.torch


def checkpoint_digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            value.update(block)
    return value.hexdigest()


def physical_summary(errors, baseline):
    errors, baseline = np.asarray(errors), np.asarray(baseline)
    if (errors.ndim != 2 or errors.shape[1] != 5 or errors.shape != baseline.shape
            or not len(errors) or not np.isfinite(errors).all() or not np.isfinite(baseline).all()):
        raise ValueError("物理误差统计需要同形状的非空有限 [N,5] 数组")
    return {"action_comparisons": len(errors), "joint_mae_rad": errors[:, :4].mean(axis=0).tolist(),
            "joint_max_abs_error_rad": errors[:, :4].max(axis=0).tolist(),
            "gripper_mae_mm": float(errors[:, 4].mean()*1000),
            "gripper_p95_abs_error_mm": float(np.percentile(errors[:, 4], 95)*1000),
            "gripper_max_abs_error_mm": float(errors[:, 4].max()*1000),
            "hold_joint_mae_rad": baseline[:, :4].mean(axis=0).tolist(),
            "hold_gripper_mae_mm": float(baseline[:, 4].mean()*1000)}


def summarize(records):
    by_start, by_action = {}, {}
    errors, baselines = [], []
    for record in records:
        target = np.asarray(record["target_physical"])
        prediction = np.asarray(record["prediction_physical"])
        state = np.asarray(record["state_physical"])
        if target.shape not in ((10,5),(50,5)) or prediction.shape != target.shape or state.shape != (5,):
            raise ValueError("评估轨迹或状态形状错误")
        error, baseline = abs(prediction-target), abs(state[None, :]-target)
        if len(record["action_phases"]) != len(target):
            raise ValueError("动作阶段标签长度不一致")
        errors.extend(error.tolist())
        baselines.extend(baseline.tolist())
        by_start.setdefault(record["start_phase"], []).append(record)
        for phase, err, hold in zip(record["action_phases"], error, baseline):
            group = by_action.setdefault(phase, {"errors": [], "baseline": []})
            group["errors"].append(err.tolist())
            group["baseline"].append(hold.tolist())
    return {"evaluated_starts": len(records), "per_start_mean": pilot.aggregate(records),
            "all_actions": physical_summary(errors, baselines),
            "by_start_phase": {key: pilot.aggregate(group) for key, group in by_start.items()},
            "by_action_phase": {key: physical_summary(group["errors"], group["baseline"])
                                for key, group in by_action.items()}}


def external_validation(root, train):
    """Load held-out physical samples; never import their normalization stats."""
    report = forward.read_json(root/"batch_conversion_report.json")
    if report.get("status") != "verified" or report.get("split_unit") != "original_recording":
        raise ValueError("独立评估数据必须已完成按原始示范划分及读回验证")
    data = SplitData(root/"validation", "validation")
    train_paths = {str(Path(ep['source_episode']).resolve()) for ep in train.manifest['episodes']}
    validation_paths = {str(Path(ep['source_episode']).resolve()) for ep in data.manifest['episodes']}
    if set(train.groups) & set(data.groups) or train_paths & validation_paths:
        raise ValueError("独立评估示范与模型训练示范重叠")
    if data.features != train.features or data.manifest['instruction'] != train.manifest['instruction']:
        raise ValueError("独立评估的特征或任务指令与训练不一致")
    return data


def prepare_inputs(dataset, checkpoint, assets, evaluation_dataset=None):
    root, checkpoint = Path(dataset).expanduser().resolve(strict=True), Path(checkpoint).expanduser().resolve(strict=True)
    training_report = forward.read_json(checkpoint.parent/"training_report.json")
    plan = training_report["plan"]
    if (training_report.get("status") != "completed" or not training_report.get("checkpoint_saved")
            or Path(training_report["checkpoint"]).resolve() != checkpoint or Path(plan["dataset"]).resolve() != root):
        raise ValueError("权重与已完成训练报告/数据集不匹配")
    _, vlm, revisions = forward.load_assets(assets)
    if revisions != plan["revisions"]:
        raise ValueError("模型素材提交号与训练时不一致")
    train, validation, stats = load_batch(root)
    if (digest(root/"training_stats.json") != plan["normalization_sha256"]
            or digest(checkpoint.parent/"training_stats.json") != plan["normalization_sha256"]):
        raise ValueError("评估归一化统计与训练时不一致")
    evaluation_root = Path(evaluation_dataset).expanduser().resolve(strict=True) if evaluation_dataset else root
    if evaluation_root != root:
        validation = external_validation(evaluation_root, train)
    config = forward.PreTrainedConfig.from_pretrained(checkpoint, local_files_only=True)
    if config.input_features != forward.configure(train.base.meta)[0].input_features or config.output_features != forward.configure(train.base.meta)[0].output_features:
        raise ValueError("已训练策略的外部特征与数据集不一致")
    config, _ = forward.adapt_config(config, train.base.meta, vlm)
    if config.chunk_size!=plan.get('chunk_size',50):
        raise ValueError('策略动作时域与训练报告不一致')
    sources = forward.read_json(evaluation_root/"source_manifest.json")["sources"]
    phases = {}
    for source in sources:
        if source["split"] != "validation":
            continue
        path = Path(source["source_episode"])/"task_events.jsonl"
        if digest(path) != source["source_file_sha256"]["task_events.jsonl"]:
            raise ValueError("原始验证任务阶段文件已变化")
        phases[source["task_id"]] = parse_phases(forward.read_jsonl(path), source["task_id"])
    if set(phases) != set(validation.groups):
        raise ValueError("验证示范的阶段信息不完整")
    return checkpoint, config, validation, stats, phases


def evaluate(dataset, checkpoint, assets, output, evaluation_dataset=None):
    if forward.version("lerobot") != "0.4.4" or not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise ValueError("需要 LeRobot 0.4.4 和支持 BF16 的 CUDA GPU")
    checkpoint, config, data, stats, phases = prepare_inputs(dataset, checkpoint, assets, evaluation_dataset)
    output.mkdir(parents=True)
    plan = {"schema_version": 1, "dataset": str(Path(dataset).resolve()), "checkpoint": str(checkpoint),
            "evaluation_dataset": str(Path(evaluation_dataset or dataset).resolve()),
            "normalization_source": "checkpoint_training_dataset_only",
            "normalization_sha256": digest(Path(dataset)/"training_stats.json"),
            "checkpoint_sha256": checkpoint_digest(checkpoint/"model.safetensors"),
            "indices": data.starts, "normalization": "same_frozen_training_statistics",
            "prediction_horizon_steps":config.chunk_size,
            "physical_action_offsets_seconds":[(i+1)/10 for i in range(config.chunk_size)],
            "start_pool":"complete_50_step_starts_even_for_shorter_policy_horizon",
            "noise": "same_as_training_evaluation: private_seed_20261107_plus_index; flow_times_0.2_0.5_0.8",
            "start_phase": "observation_grid_time", "action_phase": "each_physical_reference_timestamp",
            "overlapping_windows_not_independent": True}
    forward.write_json(output/"evaluation_plan.json", plan)
    print(f"[模型] 严格加载已训练的最终权重；验证全部 {len(data.starts)} 个完整起点。", flush=True)
    policy = forward.load_policy(checkpoint, config)
    policy.to("cuda").eval()
    config.device = "cuda"
    preprocessor, postprocessor = forward.make_smolvla_pre_post_processors(config, dataset_stats=stats)
    versions = {n: p._version for n, p in policy.named_parameters()}
    records = []
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    with (output/"predictions.jsonl").open("x", encoding="utf-8", buffering=1) as stream:
        for completed, index in enumerate(data.starts, 1):
            raw = data.batch(index)
            horizon=config.chunk_size
            raw['action']=raw['action'][:,:horizon]
            raw['action_is_pad']=raw['action_is_pad'][:,:horizon]
            target, state = raw["action"].clone(), raw[forward.STATE_KEY][:, -1, :].clone()
            baseline = single.action_errors(state.unsqueeze(1).expand_as(target), target)
            processed = preprocessor(raw)
            generator = torch.Generator(device="cuda").manual_seed(20261107+index)
            probes = [(torch.randn((1, horizon, 32), device="cuda", generator=generator), torch.tensor([t], device="cuda"))
                      for t in (.2, .5, .8)]
            noise = torch.randn((1, horizon, 32), device="cuda", generator=generator)
            result = single.evaluate_fixed(policy, processed, target, postprocessor, probes, noise)
            provenance = data.mapping[index]
            grid, task = provenance["source_grid_row"], provenance["source_task_id"]
            result.update(dataset_index=index, source_task_id=task, dataset_episode_index=provenance["dataset_episode_index"],
                          source_grid_index=grid["grid_index"], source_stamp_ns=grid["stamp_ns"],
                          start_phase=phase_at(grid["stamp_ns"], phases[task]),
                          action_phases=[phase_at(grid["action_stamp_ns"]+i*100_000_000, phases[task]) for i in range(horizon)],
                          target_physical=target[0].tolist(), state_physical=state[0].tolist(), baseline=baseline)
            records.append(result)
            stream.write(forward.json.dumps(result, ensure_ascii=False, allow_nan=False)+"\n")
            if completed == 1 or completed % 20 == 0 or completed == len(data.starts):
                print(f"[完整验证] {completed}/{len(data.starts)}；当前起点 {index}，阶段 {result['start_phase']}。", flush=True)
    if any(p.grad is not None or p._version != versions[n] for n, p in policy.named_parameters()):
        raise ValueError("评估产生梯度或改变模型参数")
    report = {"schema_version": 1, "status": "completed", "scope": "all_complete_validation_starts_open_loop",
              "plan": plan, "summary": summarize(records),
              "peak_allocated_mib": torch.cuda.max_memory_allocated()/1024**2,
              "peak_reserved_mib": torch.cuda.max_memory_reserved()/1024**2,
              "parameters_unchanged_and_gradients_absent": True,
              "optimizer_steps": 0, "robot_commands_sent": False,
              "generalization_scope": "held_out_recordings_fixed_cup_and_camera; not_cup_position_generalization"}
    forward.write_json(output/"evaluation_report.json", report)
    return report


def main(args=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--assets-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--evaluation-dataset", help="可选：单独转换的独立测试数据集；只读取 validation，仍使用该模型训练时的归一化统计")
    options = parser.parse_args(args)
    output = Path(options.output_dir).expanduser().resolve()
    try:
        if output.exists():
            raise FileExistsError(f"输出已存在，不会覆盖：{output}")
        for directory in (options.dataset, options.checkpoint, options.assets_dir, options.evaluation_dataset):
            if directory is None:
                continue
            if output.is_relative_to(Path(directory).expanduser().resolve()):
                raise ValueError("评估输出应位于数据集、权重和素材目录之外")
        torch.set_num_threads(1)
        report = evaluate(options.dataset, options.checkpoint, options.assets_dir, output, options.evaluation_dataset)
    except (OSError, ValueError, RuntimeError, ImportError, KeyError) as error:
        print(f"[完整验证失败] {type(error).__name__}: {error}；已有计划和逐起点结果保留。", flush=True)
        return 1
    print(f"[完整验证完成] {report['summary']['evaluated_starts']} 个起点；{report['summary']['all_actions']}", flush=True)
    for phase, metrics in report["summary"]["by_action_phase"].items():
        print(f"[按动作所属阶段 {phase}] {metrics}", flush=True)
    plot = subprocess.run(["/usr/bin/python3", "-I", str(Path(__file__).with_name("plot_smolvla_validation.py")), str(output)],
                          text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(plot.stdout, end="", flush=True)
    if plot.returncode:
        print("[绘图未完成] 数值评估已完成且报告保留；绘图错误需单独处理。", flush=True)
    print(f"[保存] {output}；未更新参数、未发送机器人命令。", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
