"""One disposable optimizer update on one local batch; saves a report only."""

import argparse
from contextlib import nullcontext
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
import sys
import time

# The sibling module sets offline mode before importing HF/LeRobot. Explicit
# path insertion supports the documented `python -I tools/...` invocation.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import probe_smolvla_forward as forward
from wheel_arm_manipulation.smolvla_loss import robot_action_loss

torch = forward.torch
BACKBONE = "model.vlm_with_expert.vlm."
GROUPS = {
    "expert": "model.vlm_with_expert.lm_expert.",
    "state_projection": "model.state_proj.",
    "action_input": "model.action_in_proj.",
    "action_output": "model.action_out_proj.",
    "action_time_input": "model.action_time_mlp_in.",
    "action_time_output": "model.action_time_mlp_out.",
}


def parameter_group(name):
    return next((group for group, prefix in GROUPS.items() if name.startswith(prefix)), "other")


def check_training_scope(policy):
    config = policy.config
    if not (config.freeze_vision_encoder and config.train_expert_only and config.train_state_proj):
        raise ValueError("本阶段要求冻结 VLM 并训练专家和状态投影，禁止静默改变训练范围")
    named = dict(policy.named_parameters())
    frozen_backbone = {n: p for n, p in named.items() if n.startswith(BACKBONE)}
    if not frozen_backbone or any(p.requires_grad for p in frozen_backbone.values()):
        raise ValueError("视觉语言主干没有完全冻结")
    trainable = {n: p for n, p in named.items() if p.requires_grad}
    if {parameter_group(n) for n in trainable} != set(GROUPS):
        raise ValueError("可训练参数范围不是预期的专家、投影和时间编码六组模块")
    return {
        "total_parameters": sum(p.numel() for p in named.values()),
        "trainable_parameters": sum(p.numel() for p in trainable.values()),
        "frozen_parameters": sum(p.numel() for p in named.values() if not p.requires_grad),
        "trainable_parameters_by_group": {
            group: sum(p.numel() for n, p in trainable.items() if parameter_group(n) == group)
            for group in GROUPS
        },
    }


def one_update(policy, batch, optimizer_config, device, loss_function=None):
    """Shared update/check path; CPU toy tests exercise the real optimizer path."""
    named = dict(policy.named_parameters())
    trainable = {n: p for n, p in named.items() if p.requires_grad}
    frozen = {n: p for n, p in named.items() if not p.requires_grad}
    frozen_versions = {n: p._version for n, p in frozen.items()}
    if not trainable:
        raise ValueError("没有可训练参数")
    optimizer = optimizer_config.build(list(trainable.values()))
    optimizer.zero_grad(set_to_none=True)
    synchronize = torch.cuda.synchronize if device == "cuda" else lambda: None
    context = torch.autocast("cuda", dtype=torch.bfloat16) if device == "cuda" else nullcontext()
    synchronize()
    started = time.perf_counter()
    with context:
        loss, loss_details = (policy.forward(batch) if loss_function is None
                              else loss_function(policy, batch))
    forward.require_tensor(loss, (), "训练 loss")
    if not loss.requires_grad:
        raise ValueError("loss 不连接自动求导图")
    synchronize()
    forward_seconds = time.perf_counter()-started
    print(f"[反向] loss={loss.item():.6g}；计算梯度。", flush=True)
    started = time.perf_counter()
    loss.backward()
    synchronize()
    backward_seconds = time.perf_counter()-started
    representatives = {}
    missing, zero_gradients = [], []
    for name, parameter in trainable.items():
        gradient = parameter.grad
        if gradient is None:
            missing.append(name)
            continue
        if not torch.isfinite(gradient).all():
            raise ValueError(f"梯度含 NaN/Inf，禁止更新：{name}")
        if not gradient.any():
            zero_gradients.append(name)
            continue
        group = parameter_group(name)
        if group not in representatives:
            # CPU copies of one tensor per group avoid duplicating the model
            # on the GPU and prove value changes, not just version changes.
            representatives[group] = (name, parameter.detach().cpu().clone())
    expected_groups = {parameter_group(n) for n in trainable}
    if set(representatives) != expected_groups:
        raise ValueError(f"部分训练模块没有非零梯度：{sorted(expected_groups-set(representatives))}")
    if any(p.grad is not None for p in frozen.values()):
        raise ValueError("冻结参数出现梯度，禁止更新")
    norm = torch.nn.utils.clip_grad_norm_(list(trainable.values()), optimizer_config.grad_clip_norm,
                                         error_if_nonfinite=True)
    if not torch.isfinite(norm) or norm.item() <= 0:
        raise ValueError("整体梯度范数非有限或为零，禁止更新")
    print(f"[更新] 梯度范数 {norm.item():.6g}；执行一次 AdamW step。", flush=True)
    synchronize()
    started = time.perf_counter()
    optimizer.step()
    synchronize()
    step_seconds = time.perf_counter()-started
    updates = {}
    for group, (name, before) in representatives.items():
        after = named[name].detach().cpu()
        delta = (after-before).abs()
        if not torch.isfinite(after).all() or not delta.any():
            raise ValueError(f"代表参数未更新或产生非有限值：{name}")
        updates[group] = {"parameter": name, "changed_elements": int(torch.count_nonzero(delta)),
                          "max_abs_change": delta.max().item()}
    if any(p.grad is not None or p._version != frozen_versions[n] for n, p in frozen.items()):
        raise ValueError("冻结参数出现梯度或版本变更")
    if any(not torch.isfinite(p).all() for p in trainable.values()):
        raise ValueError("更新后的可训练参数包含非有限值")
    return {
        "optimizer_steps": 1, "loss_before_update": loss.item(), "loss_details": loss_details,
        "gradient_norm_before_clip": norm.item(), "grad_clip_norm": optimizer_config.grad_clip_norm,
        "trainable_tensors_with_gradient": len(trainable)-len(missing),
        "trainable_tensors_without_gradient": missing, "zero_gradient_tensors": zero_gradients,
        "representative_parameter_updates": updates,
        "all_trainable_parameters_finite_after_update": True,
        "frozen_parameters_have_no_gradient_and_versions_unchanged": True,
        "forward_seconds": forward_seconds, "backward_seconds": backward_seconds,
        "optimizer_step_seconds": step_seconds,
    }


def probe(dataset, assets, index, loss_scope="official"):
    if version("lerobot") != "0.4.4":
        raise ValueError("本阶段固定使用 LeRobot 0.4.4")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise ValueError("本阶段需要支持 BF16 的 CUDA GPU")
    policy_dir, vlm_dir, revisions = forward.load_assets(assets)
    config, stats, batch, input_report = forward.prepare_batch(dataset, policy_dir, vlm_dir, index)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    print("[模型] 本地完整权重严格加载；batch=1，chunk_size=50。", flush=True)
    policy = forward.SmolVLAPolicy.from_pretrained(policy_dir, config=config, local_files_only=True, strict=True)
    policy.to("cuda")
    policy.train()
    scope = check_training_scope(policy)
    if policy.model.vlm_with_expert.vlm.training:
        raise ValueError("冻结的视觉语言主干没有保持 eval 模式")
    config.device = "cuda"
    preprocessor, _ = forward.make_smolvla_pre_post_processors(config, dataset_stats=stats)
    processed = preprocessor(batch)
    forward.require_tensor(processed["action"], (1, 50, 5), "训练动作")
    print(f"[训练范围] 可训练 {scope['trainable_parameters']:,}；冻结 {scope['frozen_parameters']:,} 个参数。", flush=True)
    optimizer_config = config.get_optimizer_preset()
    print(f"[优化器] AdamW，lr={optimizer_config.lr}；BF16 autocast；只更新一次。", flush=True)
    if loss_scope not in ("official", "robot"):
        raise ValueError("未知 loss 范围")
    print("[loss 范围] " + ("实际五维；排除补齐维度和时间 padding。" if loss_scope == "robot"
                            else "官方内部 32 维；用于兼容性对照。"), flush=True)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    update = one_update(policy, processed, optimizer_config, "cuda",
                        loss_function=robot_action_loss if loss_scope == "robot" else None)
    return {
        "schema_version": 1, "status": "passed", "scope": "one_disposable_training_update_only",
        "input": input_report, "assets": str(Path(assets).resolve()), "revisions": revisions,
        "strict_checkpoint_load": True, "gpu": torch.cuda.get_device_name(),
        "packages": {name: version(name) for name in ("lerobot", "torch", "transformers")},
        "autocast_dtype": "bfloat16", "parameter_dtype": str(next(policy.parameters()).dtype),
        "training_scope": scope, "optimizer": asdict(optimizer_config),
        "scheduler_used": False, "learning_rate_scope": "fixed_checkpoint_preset_for_smoke_test_only",
        "update": update, "peak_allocated_mib": torch.cuda.max_memory_allocated()/1024**2,
        "peak_reserved_mib": torch.cuda.max_memory_reserved()/1024**2,
        "memory_scope": "batch_1_horizon_50_forward_backward_and_first_adamw_step",
        "loss_scope": ("robot_5d_valid_steps_normalized_flow_matching" if loss_scope == "robot" else
                       "official_0.4.4_loss_includes_32_padded_action_dimensions; not_grasp_accuracy_or_final_training_objective"),
        "training_started": True, "formal_training_started": False,
        "checkpoint_saved": False, "robot_commands_sent": False,
    }


def main(args=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset")
    parser.add_argument("--assets-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--loss-scope", choices=("official", "robot"), default="official",
                        help="official 保留原冒烟检查；robot 仅监督真实五维动作")
    options = parser.parse_args(args)
    output = Path(options.output).expanduser().resolve()
    try:
        if output.exists():
            raise FileExistsError(f"报告已存在：{output}")
        for directory in (options.dataset, options.assets_dir):
            if output.is_relative_to(Path(directory).expanduser().resolve()):
                raise ValueError("报告应写入独立诊断目录，不写入数据集或模型素材目录")
        report = probe(options.dataset, options.assets_dir, options.frame_index, options.loss_scope)
        output.parent.mkdir(parents=True, exist_ok=True)
        forward.write_json(output, report)
    except (OSError, ValueError, RuntimeError, ImportError, KeyError) as error:
        print(f"[训练冒烟检查失败] {type(error).__name__}: {error}", flush=True)
        return 1
    update = report["update"]
    print(f"[训练冒烟检查通过] loss={update['loss_before_update']:.6g}；反向传播和 1 次参数更新完成。", flush=True)
    if options.loss_scope == "robot":
        print(f"[同次前向 loss 对照] {update['loss_details']}", flush=True)
    print(f"[参数变化] {update['representative_parameter_updates']}", flush=True)
    print(f"[耗时] 前向 {update['forward_seconds']:.2f} s；反向 {update['backward_seconds']:.2f} s；"
          f"更新 {update['optimizer_step_seconds']:.2f} s。", flush=True)
    print(f"[显存] 张量峰值 {report['peak_allocated_mib']:.1f} MiB；缓存峰值 {report['peak_reserved_mib']:.1f} MiB。", flush=True)
    print(f"[保存] {output}", flush=True)
    print(f"[范围] 单批次训练链路验证，loss_scope={options.loss_scope}；仅保存报告，"
          "内存中的更新随进程退出丢弃。未连接机器人。", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
