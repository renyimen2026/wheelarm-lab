"""Bounded memorization probe on one moving chunk; report only, no checkpoint."""

import argparse
from contextlib import nullcontext
from dataclasses import asdict
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import probe_smolvla_train_step as training

forward = training.forward
torch = training.torch


def choose_moving_start(mapping, horizon=50, selection="arm"):
    if selection not in ("arm", "gripper-close"):
        raise ValueError("未知动作块选择策略")
    if len(mapping) < horizon:
        raise ValueError("没有完整 50 步动作块")
    targets = torch.tensor([row["source_grid_row"]["action"] for row in mapping], dtype=torch.float64)
    if targets.shape != (len(mapping), 5) or not torch.isfinite(targets).all():
        raise ValueError("来源动作应为有限五维绝对位置")
    # Rank arm motion in radians OR net closing in metres, never mixed units.
    # Labels only select the training example, never enter policy inputs.
    scores = []
    for start in range(len(mapping)-horizon+1):
        if selection == "arm":
            arm = targets[start:start+horizon, :4]
            scores.append(((arm.amax(dim=0)-arm.amin(dim=0))**2).sum().item())
        else:
            scores.append((targets[start, 4]-targets[start+horizon-1, 4]).item())
    index = max(range(len(scores)), key=scores.__getitem__)
    span = targets[index:index+horizon].amax(dim=0)-targets[index:index+horizon].amin(dim=0)
    if selection == "arm" and span[:4].max() < .02:
        raise ValueError("所有完整动作块的机械臂变化均小于 0.02 rad；请另选运动数据")
    if selection == "gripper-close" and scores[index] < .005:
        raise ValueError("没有夹爪净闭合至少 5 mm 的完整动作块；张开或静止不能作为闭合样本")
    forward.validate_window(mapping, index, horizon)
    return index, span.tolist()


def action_errors(prediction, target):
    if prediction.shape != target.shape or tuple(prediction.shape) not in ((1,10,5),(1,50,5)):
        raise ValueError("动作误差比较需要相同的 [1,10/50,5] 张量")
    if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
        raise ValueError("预测或目标含非有限值")
    error = (prediction-target).abs()
    mean, maximum = error.mean(dim=(0, 1)), error.amax(dim=(0, 1))
    return {"joint_mae_rad": mean[:4].tolist(), "gripper_mae_mm": mean[4].item()*1000,
            "joint_max_abs_error_rad": maximum[:4].tolist(), "gripper_max_abs_error_mm": maximum[4].item()*1000}


def evaluate_fixed(policy, batch, target, postprocessor, noises, sampling_noise):
    was_training = policy.training
    policy.eval()
    # Evaluation never consumes the RNG stream used by subsequent training.
    try:
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]), torch.no_grad(), \
                torch.autocast("cuda", dtype=torch.bfloat16):
            losses = []
            for noise, timestep in noises:
                loss, _ = training.robot_action_loss(policy, batch, noise=noise.clone(), time=timestep.clone())
                forward.require_tensor(loss, (), "固定评估 loss")
                losses.append(loss.item())
            observations = {k: v for k, v in batch.items() if k != "action" and not k.startswith("action_")}
            policy.reset()
            prediction = policy.predict_action_chunk(observations, noise=sampling_noise.clone())
            physical = postprocessor(prediction).detach().float().cpu()
            forward.require_tensor(physical, target.shape, "评估预测")
            return {"fixed_flow_loss": sum(losses)/len(losses), "fixed_flow_losses": losses,
                    "physical_errors": action_errors(physical, target), "prediction_physical": physical[0].tolist()}
    finally:
        policy.train(was_training)


def train_fixed_batch(policy, batch, optimizer_config, steps, every, evaluate, device,
                      loss_function=training.robot_action_loss, alternate_batch=None):
    limit = 500 if alternate_batch is None else 1000
    if not 1 <= steps <= limit or not 1 <= every <= steps:
        raise ValueError(f"本阶段要求 1～{limit} 次更新，评估间隔须介于 1 和总步数之间")
    named = dict(policy.named_parameters())
    parameters = [p for p in named.values() if p.requires_grad]
    frozen = {n: p for n, p in named.items() if not p.requires_grad}
    versions = {n: p._version for n, p in frozen.items()}
    if not parameters:
        raise ValueError("没有可训练参数")
    representatives = {}
    for name, p in named.items():
        if p.requires_grad and training.parameter_group(name) not in representatives:
            representatives[training.parameter_group(name)] = (name, p.detach().cpu().clone())
    optimizer = optimizer_config.build(parameters)  # One optimizer for the ENTIRE run.
    history, evaluations = [], []

    def record(step):
        result = evaluate()
        result["step"] = step
        evaluations.append(result)
        samples = result.get("samples", {"single": result})
        for i, (label, sample) in enumerate(samples.items()):
            errors = sample["physical_errors"]
            count = step if alternate_batch is None else (step+1-i)//2
            print(f"[评估 {step}/{steps}][{label} 已更新 {count} 次] 固定 loss={sample['fixed_flow_loss']:.6g}；"
                  f"关节 MAE(rad)={errors['joint_mae_rad']}；夹爪 MAE={errors['gripper_mae_mm']:.3f} mm。", flush=True)

    record(0)
    policy.train()
    for step in range(1, steps+1):
        optimizer.zero_grad(set_to_none=True)
        sample_index = 1 if alternate_batch is not None and step % 2 == 0 else 0
        active_batch = alternate_batch if sample_index == 1 else batch
        with torch.autocast("cuda", dtype=torch.bfloat16) if device == "cuda" else nullcontext():
            loss, _ = loss_function(policy, active_batch)
        forward.require_tensor(loss, (), "训练 loss")
        if not loss.requires_grad:
            raise ValueError("训练 loss 不连接自动求导图")
        loss.backward()
        if not any(p.grad is not None for p in parameters):
            raise ValueError("训练没有产生梯度")
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in parameters):
            raise ValueError(f"第 {step} 步梯度含 NaN/Inf，禁止更新")
        norm = torch.nn.utils.clip_grad_norm_(parameters, optimizer_config.grad_clip_norm, error_if_nonfinite=True)
        optimizer.step()
        history.append({"step": step, "sample_index": sample_index,
                        "random_training_loss": loss.item(), "gradient_norm_before_clip": norm.item()})
        if step % every == 0 or step == steps:
            if any(not torch.isfinite(p).all() for p in parameters):
                raise ValueError(f"第 {step} 步参数含 NaN/Inf")
            if any(p.grad is not None or p._version != versions[n] for n, p in frozen.items()):
                raise ValueError("冻结参数出现梯度或版本变更")
            record(step)
    changes = {}
    for group, (name, before) in representatives.items():
        delta = (named[name].detach().cpu()-before).abs()
        if not delta.any():
            raise ValueError(f"训练模块代表参数未变化：{name}")
        changes[group] = {"parameter": name, "max_abs_change": delta.max().item()}
    return {"optimizer_steps": steps, "history": history, "evaluations": evaluations,
            "representative_parameter_updates": changes, "frozen_parameter_versions_unchanged": True}


def probe(dataset, assets, steps, every, learning_rate=None, sample_selection="arm"):
    if forward.version("lerobot") != "0.4.4" or not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise ValueError("需要 LeRobot 0.4.4 和支持 BF16 的 CUDA GPU")
    policy_dir, vlm_dir, revisions = forward.load_assets(assets)
    root = Path(dataset).expanduser().resolve(strict=True)
    index, span = choose_moving_start(forward.read_jsonl(root/"source_mapping.jsonl"), selection=sample_selection)
    config, stats, raw_batch, input_report = forward.prepare_batch(root, policy_dir, vlm_dir, index)
    selection_description = "机械臂变化最大" if sample_selection == "arm" else "夹爪净闭合量最大"
    print(f"[样本] 自动选取{selection_description}的完整动作块；起点 {index}。", flush=True)
    print(f"[目标变化] 四关节范围(rad)={span[:4]}；夹爪范围={span[4]*1000:.3f} mm。", flush=True)
    target = raw_batch["action"].clone()
    if sample_selection == "gripper-close":
        print(f"[闭合目标] {target[0, 0, 4].item()*1000:.3f} → {target[0, -1, 4].item()*1000:.3f} mm；"
              "是控制器参考开口，不是杯子尺寸或实际接触测量。", flush=True)
    state = raw_batch[forward.STATE_KEY][:, -1, :].unsqueeze(1).expand_as(target)
    baseline = action_errors(state, target)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    policy = forward.SmolVLAPolicy.from_pretrained(policy_dir, config=config, local_files_only=True, strict=True)
    policy.to("cuda")
    policy.train()
    scope = training.check_training_scope(policy)
    if policy.model.vlm_with_expert.vlm.training:
        raise ValueError("冻结主干应处于 eval 模式")
    config.device = "cuda"
    preprocessor, postprocessor = forward.make_smolvla_pre_post_processors(config, dataset_stats=stats)
    batch = preprocessor(raw_batch)
    # Private generator creates probes without advancing the training RNG.
    generator = torch.Generator(device="cuda").manual_seed(20261107)
    noises = [(torch.randn((1, 50, 32), device="cuda", generator=generator),
               torch.tensor([t], device="cuda")) for t in (.2, .5, .8)]
    sampling_noise = torch.randn((1, 50, 32), device="cuda", generator=generator)
    preset_learning_rate = config.optimizer_lr
    if learning_rate is not None:
        if not math.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("学习率必须是有限正数")
        config.optimizer_lr = learning_rate
    optimizer_config = config.get_optimizer_preset()
    print(f"[训练] 固定一个样本，{steps} 步；AdamW lr={optimizer_config.lr}；每 {every} 步评估。", flush=True)
    print(f"[保持当前状态基线] {baseline}", flush=True)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    result = train_fixed_batch(policy, batch, optimizer_config, steps, every,
                               lambda: evaluate_fixed(policy, batch, target, postprocessor, noises, sampling_noise), "cuda")
    return {
        "schema_version": 1, "status": "completed", "scope": "one_moving_chunk_memorization_probe_only",
        "input": input_report, "revisions": revisions, "assets": str(Path(assets).resolve()),
        "sample_selection": sample_selection,
        "training_scope": scope, "optimizer": asdict(optimizer_config), "scheduler_used": False,
        "checkpoint_preset_learning_rate": preset_learning_rate,
        "learning_rate_source": "checkpoint_preset" if learning_rate is None else "command_line_override",
        "target_physical": target[0].tolist(), "target_peak_to_peak": span,
        "hold_current_state_baseline": baseline, "run": result,
        "evaluation": {"fixed_noise_seed": 20261107, "flow_times": [.2, .5, .8],
                       "flow_noise_fixed": True, "sampling_noise_fixed": True,
                       "training_noise_and_time": "fresh_random_each_update", "held_out_samples": 0},
        "gpu": torch.cuda.get_device_name(), "autocast_dtype": "bfloat16",
        "loss_scope": "normalized_flow_matching_real_5d_only; exclude_temporal_padding",
        "peak_allocated_mib": torch.cuda.max_memory_allocated()/1024**2,
        "peak_reserved_mib": torch.cuda.max_memory_reserved()/1024**2,
        "checkpoint_saved": False, "robot_commands_sent": False, "generalization_evaluated": False,
    }


def positive_learning_rate(value):
    rate = float(value)
    if not math.isfinite(rate) or rate <= 0:
        raise argparse.ArgumentTypeError("学习率必须是有限正数")
    return rate


def main(args=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset")
    parser.add_argument("--assets-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--learning-rate", type=positive_learning_rate, default=None,
                        help="覆盖固定学习率；省略时沿用预训练配置，不启用 scheduler")
    parser.add_argument("--sample-selection", choices=("arm", "gripper-close"), default="arm",
                        help="选择机械臂变化最大或夹爪净闭合量最大的完整动作块")
    options = parser.parse_args(args)
    output = Path(options.output).expanduser().resolve()
    try:
        if not 1 <= options.steps <= 500 or not 1 <= options.eval_every <= options.steps:
            raise ValueError("本阶段最多 500 步；评估间隔须介于 1 和总步数之间")
        if output.exists():
            raise FileExistsError(f"报告已存在：{output}")
        for directory in (options.dataset, options.assets_dir):
            if output.is_relative_to(Path(directory).expanduser().resolve()):
                raise ValueError("报告应写入独立诊断目录")
        report = probe(options.dataset, options.assets_dir, options.steps, options.eval_every,
                       options.learning_rate, options.sample_selection)
        output.parent.mkdir(parents=True, exist_ok=True)
        forward.write_json(output, report)
    except (OSError, ValueError, RuntimeError, ImportError, KeyError) as error:
        print(f"[过拟合实验失败] {type(error).__name__}: {error}", flush=True)
        return 1
    print(f"[实验完成] {options.steps} 次更新；需要结合评估曲线判断是否学会该样本。", flush=True)
    print(f"[显存] 张量峰值 {report['peak_allocated_mib']:.1f} MiB；缓存峰值 {report['peak_reserved_mib']:.1f} MiB。", flush=True)
    print(f"[保存] {output}", flush=True)
    print("[范围] 单样本记忆实验，无泛化或闭环验证；仅保存报告，更新权重随进程退出丢弃。", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
