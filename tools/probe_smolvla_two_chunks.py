"""Alternate two fixed training examples in one model/optimizer; report only."""

import argparse
from dataclasses import asdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import probe_smolvla_overfit as single

forward, training, torch = single.forward, single.training, single.torch
LABELS = ("arm", "gripper-close")


def probe(dataset, assets, per_sample, every, learning_rate):
    if forward.version("lerobot") != "0.4.4" or not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise ValueError("需要 LeRobot 0.4.4 和支持 BF16 的 CUDA GPU")
    policy_dir, vlm_dir, revisions = forward.load_assets(assets)
    root = Path(dataset).expanduser().resolve(strict=True)
    mapping = forward.read_jsonl(root/"source_mapping.jsonl")
    samples, raw_batches, indices = {}, [], []
    for label in LABELS:
        index, span = single.choose_moving_start(mapping, selection=label)
        config, stats, raw, provenance = forward.prepare_batch(root, policy_dir, vlm_dir, index)
        if raw_batches and (config.input_features != first_config.input_features
                            or config.output_features != first_config.output_features):
            raise ValueError("两个样本的策略特征不一致")
        if not raw_batches:
            first_config, first_stats = config, stats
        target = raw["action"].clone()
        state = raw[forward.STATE_KEY][:, -1, :].unsqueeze(1).expand_as(target)
        baseline = single.action_errors(state, target)
        samples[label] = {"input": provenance, "target_peak_to_peak": span,
                          "target_physical": target[0].tolist(), "hold_current_state_baseline": baseline}
        raw_batches.append(raw)
        indices.append(index)
        print(f"[样本 {label}] 起点 {index}；关节变化(rad)={span[:4]}；夹爪变化={span[4]*1000:.3f} mm。", flush=True)
        print(f"[保持状态基线 {label}] {baseline}", flush=True)
    if indices[0] == indices[1]:
        raise ValueError("两种选择策略选中了同一起点；本轮要求两个不同样本")
    config, stats = first_config, first_stats
    config.optimizer_lr = learning_rate
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    print("[模型] 仅加载一份本地完整预训练权重；两样本共享模型和 AdamW 状态。", flush=True)
    policy = forward.SmolVLAPolicy.from_pretrained(policy_dir, config=config, local_files_only=True, strict=True)
    policy.to("cuda")
    policy.train()
    scope = training.check_training_scope(policy)
    if policy.model.vlm_with_expert.vlm.training:
        raise ValueError("冻结主干应保持 eval 模式")
    config.device = "cuda"
    preprocessor, postprocessor = forward.make_smolvla_pre_post_processors(config, dataset_stats=stats)
    targets = [raw["action"].clone() for raw in raw_batches]
    batches = [preprocessor(raw) for raw in raw_batches]
    # Match the earlier single-chunk probes' fixed evaluation tensors for BOTH
    # samples. Evaluations restore RNG state and never advance training noise.
    generator = torch.Generator(device="cuda").manual_seed(20261107)
    noises = [(torch.randn((1, 50, 32), device="cuda", generator=generator),
               torch.tensor([t], device="cuda")) for t in (.2, .5, .8)]
    sampling_noise = torch.randn((1, 50, 32), device="cuda", generator=generator)

    def evaluate():
        return {"samples": {label: single.evaluate_fixed(policy, batch, target, postprocessor, noises, sampling_noise)
                            for label, batch, target in zip(LABELS, batches, targets)}}

    optimizer_config = config.get_optimizer_preset()
    print(f"[训练] arm → gripper-close 交替；每样本 {per_sample} 次，总计 {per_sample*2} 次；lr={learning_rate}。", flush=True)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    run = single.train_fixed_batch(policy, batches[0], optimizer_config, per_sample*2, every, evaluate,
                                   "cuda", alternate_batch=batches[1])
    counts = {label: sum(row["sample_index"] == i for row in run["history"]) for i, label in enumerate(LABELS)}
    if any(count != per_sample for count in counts.values()):
        raise ValueError("实际样本更新计数与计划不一致")
    return {
        "schema_version": 1, "status": "completed", "scope": "two_fixed_chunks_alternating_memorization_only",
        "dataset": str(root), "assets": str(Path(assets).resolve()), "revisions": revisions,
        "strict_checkpoint_load": True, "samples": samples, "sample_order": list(LABELS),
        "optimizer_steps_per_sample": counts, "optimizer": asdict(optimizer_config),
        "shared_model_and_optimizer": True, "scheduler_used": False, "training_scope": scope, "run": run,
        "gpu": torch.cuda.get_device_name(), "autocast_dtype": "bfloat16",
        "evaluation": {"fixed_noise_seed": 20261107, "flow_times": [.2, .5, .8],
                       "same_fixed_probes_as_single_chunk": True, "held_out_samples": 0},
        "loss_scope": "normalized_flow_matching_real_5d_valid_steps_only",
        "peak_allocated_mib": torch.cuda.max_memory_allocated()/1024**2,
        "peak_reserved_mib": torch.cuda.max_memory_reserved()/1024**2,
        "checkpoint_saved": False, "robot_commands_sent": False, "generalization_evaluated": False,
    }


def main(args=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset")
    parser.add_argument("--assets-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps-per-sample", type=int, default=300)
    parser.add_argument("--eval-every", type=int, default=40, help="按总更新次数计；40 次相当于每样本更新 20 次")
    parser.add_argument("--learning-rate", type=single.positive_learning_rate, default=3e-5)
    options = parser.parse_args(args)
    output = Path(options.output).expanduser().resolve()
    try:
        if not 1 <= options.steps_per_sample <= 500 or not 1 <= options.eval_every <= options.steps_per_sample*2:
            raise ValueError("每样本要求 1～500 次更新，评估间隔须介于 1 和总更新数之间")
        if output.exists():
            raise FileExistsError(f"报告已存在：{output}")
        for directory in (options.dataset, options.assets_dir):
            if output.is_relative_to(Path(directory).expanduser().resolve()):
                raise ValueError("报告应写入独立诊断目录")
        report = probe(options.dataset, options.assets_dir, options.steps_per_sample, options.eval_every, options.learning_rate)
        output.parent.mkdir(parents=True, exist_ok=True)
        forward.write_json(output, report)
    except (OSError, ValueError, RuntimeError, ImportError, KeyError) as error:
        print(f"[双样本实验失败] {type(error).__name__}: {error}", flush=True)
        return 1
    print(f"[实验完成] 更新计数 {report['optimizer_steps_per_sample']}；需分别检查两个样本。", flush=True)
    print(f"[显存] 张量峰值 {report['peak_allocated_mib']:.1f} MiB；缓存峰值 {report['peak_reserved_mib']:.1f} MiB。", flush=True)
    print(f"[保存] {output}", flush=True)
    print("[范围] 仅两个训练样本的记忆验证；未评估泛化或闭环；仅保存报告，更新权重随进程退出丢弃。", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
