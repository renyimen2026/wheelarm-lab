"""Bounded local multi-demonstration pilot; no ROS connection or uploading."""

import argparse
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
import sys
import subprocess

sys.path.insert(0, str(Path(__file__).resolve().parent))
import probe_smolvla_overfit as single
from wheel_arm_manipulation.smolvla_batch_data import (
    BalancedStarts, PhaseBalancedStarts, TransitionMixedStarts, load_batch,
    training_phase_groups, training_transition_groups,active_training_groups)

forward, training, torch = single.forward, single.training, single.torch


def aggregate(records):
    if not records:
        raise ValueError("没有评估起点")
    count = len(records)
    return {"evaluated_starts": count,
            "fixed_flow_loss": sum(r["fixed_flow_loss"] for r in records)/count,
            "joint_mae_rad": [sum(r["physical_errors"]["joint_mae_rad"][i] for r in records)/count for i in range(4)],
            "gripper_mae_mm": sum(r["physical_errors"]["gripper_mae_mm"] for r in records)/count,
            "hold_joint_mae_rad": [sum(r["baseline"]["joint_mae_rad"][i] for r in records)/count for i in range(4)],
            "hold_gripper_mae_mm": sum(r["baseline"]["gripper_mae_mm"] for r in records)/count}


def run_updates(policy, next_batch, evaluate, optimizer_config, steps, every, device,
                loss_function=training.robot_action_loss, on_update=None, on_evaluation=None,
                gradient_accumulation_steps=1):
    if not 1 <= steps <= 2000 or not 1 <= every <= steps:
        raise ValueError("首轮要求 1～2000 次更新，评估间隔须介于 1 和总步数之间")
    if type(gradient_accumulation_steps) is not int or not 1 <= gradient_accumulation_steps <= 16:
        raise ValueError("梯度累积次数须为 1～16 的整数")
    named = dict(policy.named_parameters())
    parameters = [p for p in named.values() if p.requires_grad]
    frozen = {n: p for n, p in named.items() if not p.requires_grad}
    frozen_versions = {n: p._version for n, p in frozen.items()}
    if not parameters:
        raise ValueError("没有可训练参数")
    optimizer = optimizer_config.build(parameters)
    evaluations, counts, sample_counts = [], {}, {}

    def record(step):
        optimizer.zero_grad(set_to_none=True)
        versions = {n: p._version for n, p in named.items()}
        was_training = policy.training
        devices = [torch.cuda.current_device()] if device == "cuda" else []
        try:
            policy.eval()
            with torch.random.fork_rng(devices=devices), torch.no_grad():
                result = evaluate(step)
        finally:
            policy.train(was_training)
        if any(p.grad is not None or p._version != versions[n] for n, p in named.items()):
            raise ValueError("评估过程产生梯度或改变模型参数")
        if any(p._version != frozen_versions[n] for n, p in frozen.items()):
            raise ValueError("冻结参数版本发生变化")
        if any(not torch.isfinite(p).all() for p in parameters):
            raise ValueError("训练后参数含 NaN/Inf")
        result["step"] = step
        evaluations.append(result)
        if on_evaluation is not None:
            on_evaluation(result)
        for label, value in result["splits"].items():
            summary = value["summary"]
            print(f"[评估 {step}/{steps}][{label} 固定 {summary['evaluated_starts']} 起点] "
                  f"loss={summary['fixed_flow_loss']:.6g}；关节 MAE(rad)={summary['joint_mae_rad']}；"
                  f"夹爪 MAE={summary['gripper_mae_mm']:.3f} mm。", flush=True)
            if 'coordination' in value:
                c=value['coordination']['summary']
                print(f"[闭合抬升 {step}/{steps}][{label}] 窗口 {c['evaluated_windows']}；"
                    f"先闭合再抬升 {c['closed_before_lift_windows']}；抬升时开口未达标 {c['lift_without_closure_windows']}；"
                    f"（过大 {c['lift_with_open_gripper_windows']}、过小 {c['lift_with_overclosed_gripper_windows']}、不稳定 {c['lift_with_unstable_gripper_windows']}）；"
                    f"未预测抬升 {c['no_lift_predicted_windows']}；张开时最大升高 {c['max_rise_while_open_mm']:.2f}mm。", flush=True)

    record(0)
    policy.train()
    for step in range(1, steps+1):
        optimizer.zero_grad(set_to_none=True)
        samples = []
        for micro_step in range(gradient_accumulation_steps):
            task, index, batch = next_batch()
            with torch.autocast("cuda", dtype=torch.bfloat16) if device == "cuda" else nullcontext():
                loss, loss_details = loss_function(policy, batch)
            forward.require_tensor(loss, (), "训练 loss")
            if not loss.requires_grad:
                raise ValueError("训练 loss 未连接自动求导图")
            # Each microbatch contains one complete, unpadded 50-step sample.
            # Backward immediately frees its graph; only parameter gradients accumulate.
            (loss/gradient_accumulation_steps).backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in parameters):
                raise ValueError("梯度含 NaN/Inf，禁止更新")
            samples.append({"micro_step": micro_step+1, "source_task_id": task,
                            "dataset_index": index, "random_training_loss": loss.detach().item()})
            if isinstance(loss_details,dict):
                samples[-1]['loss_components']={key:loss_details[key] for key in
                    ('base_flow_loss','geometry_loss','temporal_error_loss') if key in loss_details}
            del loss, batch
        if not any(p.grad is not None for p in parameters):
            raise ValueError("训练没有产生梯度")
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in parameters):
            raise ValueError("梯度含 NaN/Inf，禁止更新")
        if any(p.grad is not None for p in frozen.values()):
            raise ValueError("冻结参数出现梯度")
        norm = torch.nn.utils.clip_grad_norm_(parameters, optimizer_config.grad_clip_norm, error_if_nonfinite=True)
        optimizer.step()
        for task in {sample["source_task_id"] for sample in samples}:
            counts[task] = counts.get(task, 0)+1
        for sample in samples:
            task = sample["source_task_id"]
            sample_counts[task] = sample_counts.get(task, 0)+1
        average_loss = sum(sample["random_training_loss"] for sample in samples)/gradient_accumulation_steps
        if on_update is not None:
            row = {"step": step, "gradient_accumulation_steps": gradient_accumulation_steps,
                   "effective_batch_size": gradient_accumulation_steps,
                   "sample_draws_so_far": step*gradient_accumulation_steps, "samples": samples,
                   "random_training_loss": average_loss, "gradient_norm_before_clip": norm.item()}
            if gradient_accumulation_steps == 1:
                row.update(source_task_id=samples[0]["source_task_id"], dataset_index=samples[0]["dataset_index"])
            on_update(row)
        if step % 20 == 0:
            print(f"[训练 {step}/{steps}] {gradient_accumulation_steps} 样本平均 loss={average_loss:.6g}；"
                  f"梯度范数={norm.item():.6g}；累计抽取 {step*gradient_accumulation_steps} 次。", flush=True)
        if step % every == 0 or step == steps:
            record(step)
    return {"optimizer_steps": steps, "updates_per_original_recording": counts, "evaluations": evaluations,
            "updates_per_original_recording_semantics": "updates_containing_recording; counts_can_overlap",
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "total_sample_draws": steps*gradient_accumulation_steps,
            "sample_draws_per_original_recording": sample_counts,
            "evaluation_has_no_gradients_or_parameter_updates": True, "frozen_parameter_versions_unchanged": True}


def train(dataset, assets, output, steps, every, learning_rate, sampling="recording", coordination_eval=False,
          gradient_accumulation_steps=1, geometry_loss=False, initial_checkpoint=None,horizon=50,state_noise_rad=0.,
          gripper_loss_weight=0., parameter_precision='fp32'):
    if not 0<=state_noise_rad<=.03:
        raise ValueError('训练状态扰动标准差须为0～0.03rad')
    if not 0<=gripper_loss_weight<=2 or (gripper_loss_weight and not geometry_loss):
        raise ValueError('夹爪附加损失权重须为0～2且启用几何训练指标')
    if horizon not in (10,50) or (horizon!=50 and (coordination_eval or sampling=='transition')):
        raise ValueError('训练时域可选10或50步；五秒过渡诊断及抽样需要50步')
    if sampling not in ("recording", "phase", "transition",'active'):
        raise ValueError("未知抽样策略")
    if sampling == "transition" and gradient_accumulation_steps != 4:
        raise ValueError("过渡混合抽样要求 --gradient-accumulation-steps 4，每次更新两份全流程、两份过渡样本")
    if forward.version("lerobot") != "0.4.4" or not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise ValueError("需要 LeRobot 0.4.4 和支持 BF16 的 CUDA GPU")
    policy_dir, vlm_dir, revisions = forward.load_assets(assets)
    train_data, validation_data, stats = load_batch(dataset)
    config, _ = forward.adapt_config(forward.PreTrainedConfig.from_pretrained(policy_dir, local_files_only=True),
                                      train_data.base.meta, vlm_dir)
    config.optimizer_lr = learning_rate
    config.chunk_size=horizon
    config.n_action_steps=min(config.n_action_steps,horizon)
    initialization=policy_dir
    if initial_checkpoint is not None:
        initialization=Path(initial_checkpoint).resolve(strict=True)
        previous=forward.read_json(initialization.parent/'training_report.json')
        if (previous.get('status')!='completed' or not previous.get('checkpoint_saved')
                or Path(previous['checkpoint']).resolve()!=initialization
                or previous['plan']['normalization_sha256']!=forward.read_json(
                    Path(dataset)/'batch_conversion_report.json')['normalization_stats']['sha256']
                or previous['plan']['revisions']!=revisions):
            raise ValueError('微调初始权重需要已验证报告、相同冻结统计及素材来源')
    instruction = train_data.manifest["instruction"]
    tokenizer = forward.AutoTokenizer.from_pretrained(vlm_dir, local_files_only=True)
    token_count = len(tokenizer(instruction.rstrip("\n")+"\n", truncation=False)["input_ids"])
    if token_count > config.tokenizer_max_length:
        raise ValueError("任务文本超过分词上限，禁止静默截断")
    indices = {"train": train_data.evaluation_indices(horizon), "validation": validation_data.evaluation_indices(horizon)}
    plans = {label: {"full_50_step_starts": len(data.starts), "original_recordings": len(data.groups),
                     "fixed_evaluation_indices": indices[label]}
             for label, data in (("train", train_data), ("validation", validation_data))}
    phase_groups, phase_labels, phase_hashes = training_phase_groups(dataset, train_data) if sampling in ("phase",'active') else ({}, {}, {})
    active_groups=active_training_groups(train_data,phase_labels,horizon) if sampling=='active' else {}
    candidate_counts = {task: {phase: len(rows) for phase, rows in phases.items()}
                        for task, phases in phase_groups.items()}
    plan = {"schema_version": 1, "dataset": str(Path(dataset).resolve()), "assets": str(Path(assets).resolve()),
            "revisions": revisions, "steps": steps, "batch_size": 1, "chunk_size": horizon,
            "micro_batch_size": 1, "gradient_accumulation_steps": gradient_accumulation_steps,
            "effective_batch_size": gradient_accumulation_steps,
            "planned_sample_draws": steps*gradient_accumulation_steps,
            "gradient_reduction": "mean_across_equal_size_microbatches_then_clip_then_one_optimizer_step",
            "learning_rate": learning_rate, "eval_every": every, "splits": plans,
            "sampling": {"phase": "uniform_original_training_recording_then_uniform_available_start_phase_then_uniform_complete_start",
                         "recording": "uniform_original_training_recording_then_uniform_complete_start",
                         "active":"uniform_original_training_recording_then_active_or_lift_hold_start; no_time_compression",
                         "transition": "two_full_pool_and_two_reference_transition_pool_draws_per_four_sample_update; uniform_recording_in_each_pool"}[sampling],
            "phase_candidate_counts": candidate_counts, "training_phase_event_sha256": phase_hashes,
            "phase_bucket_semantics": "observation_grid_time; between-phase bucket retained; future phases may differ",
            "normalization_source": "training_stats.json", "normalization_sha256": forward.read_json(
                Path(dataset)/"batch_conversion_report.json")["normalization_stats"]["sha256"],
            "evaluation_scope": "fixed_subset; same_fixed_scene_repeatability; not_full_validation_or_closed_loop",
            "checkpoint_policy": "save_final_model_and_processors_only; no_best_selection_or_optimizer_resume"}
    plan.update(initial_checkpoint=str(initialization),optimizer_state_restored=False,
                geometry_loss=geometry_loss)
    plan['training_only_proprioception_augmentation']={'enabled':bool(state_noise_rad),
        'physical_standard_deviation':[state_noise_rad*.1]+[state_noise_rad]*3+[state_noise_rad*.05],
        'images_and_target_actions_unchanged':True,'evaluation_and_live_observations_unmodified':True}
    if active_groups:
        plan['active_sampling']={'selected_counts':{k:len(v) for k,v in active_groups.items()},
            'indices':active_groups,'definition':'lift phase retained; otherwise future reference span >3mrad or1mm, or final_reference-current_state >6mrad or2mm; no timestamps/chunks rewritten'}
    output.mkdir(parents=True)
    print(f"[输入] {plans}；仅训练集参与更新，验证集不参与归一化统计。", flush=True)
    print(f"[训练配置] 微批量 1；梯度累积 {gradient_accumulation_steps}；有效批量 {gradient_accumulation_steps}；"
          f"参数更新 {steps} 次；计划抽取 {steps*gradient_accumulation_steps} 次样本。", flush=True)
    if sampling == "phase":
        print(f"[阶段候选] {candidate_counts}；阶段按动作块起点划分，保留阶段间等待。", flush=True)
    coordination_plans, transition_groups, fk = {}, {}, None
    if coordination_eval or sampling == "transition" or geometry_loss:
        from wheel_arm_manipulation.smolvla_coordination import TCPKinematics, coordination_plan
        specification = output/"tcp_kinematics.json"
        workspace = Path(__file__).resolve().parents[3]
        subprocess.run(["bash", "-e", "-c", 'source "$1"\nsource "$2"\nexec /usr/bin/python3 -s "$3" --output "$4"',
            "vla-kinematics", "/opt/ros/humble/setup.bash", str(workspace/"install/setup.bash"),
            str(Path(__file__).with_name("export_vla_kinematics.py")), str(specification)], check=True)
        fk = TCPKinematics(forward.read_json(specification))
    loss_function=training.robot_action_loss
    if geometry_loss:
        from wheel_arm_manipulation.smolvla_geometry_loss import GeometryFlowLoss
        loss_function=GeometryFlowLoss(stats,forward.read_json(specification),gripper_weight=gripper_loss_weight)
        plan['geometry_loss_configuration']={'geometry_weight':.2,'smoothness_weight':.1,
            'position_scale_m':.01,'pitch_scale_rad':.02,
            'additional_gripper_weight':gripper_loss_weight,
            'definition':'official flow loss + target-Jacobian Cartesian/pitch velocity-error metric + temporal error differences',
            'inference_unchanged':True}
    if sampling == "transition":
        transition_groups, transition_details = training_transition_groups(dataset, train_data, fk)
        plan["transition_sampling"] = {"plans": transition_details,
            "kinematics": forward.read_json(specification),
            "ordinary_pool_contains_all_complete_starts_including_transitions": True,
            "model_predictions_used_for_selection": False, "validation_used_for_selection": False,
            "phase_or_pool_labels_added_to_model_input": False,
            "planned_draws": {"all": steps*2, "transition": steps*2},
            "definition": "observation_in_close; actual_TCP_within_5mm_of_reference_grasp_plane; future_reference_closed_within_2mm_for_two_steps_before_sustained_10mm_TCP_lift_for_three_steps; full_unpadded_50_step_window"}
        forward.write_json(output/"transition_sampling_plan.json", plan["transition_sampling"])
        print(f"[过渡候选/仅训练] { {task:len(rows) for task,rows in transition_groups.items()} }", flush=True)
        print("[混合采样] 每次更新 2 份全流程样本 + 2 份闭合抬升过渡样本；两类均按原始示范均衡抽取，允许重复。", flush=True)
    if coordination_eval:
        coordination_plans = {label: coordination_plan(dataset, data, fk)
            for label, data in (("train",train_data),("validation",validation_data))}
        plan["coordination_evaluation"] = {"plans":coordination_plans,
            "kinematics":forward.read_json(specification),"lift_rise_m":.010,"sustained_steps":3,
            "closed_band_tolerance_m":.002,"evaluation_only_no_loss_or_sampling_changes":True}
        forward.write_json(output/"coordination_plan.json",plan["coordination_evaluation"])
        print(f"[闭合抬升窗口] { {label:{task:r['indices'] for task,r in groups.items()} for label,groups in coordination_plans.items()} }", flush=True)
    plan['requested_parameter_precision'] = parameter_precision
    forward.write_json(output/"run_plan.json", plan)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    policy = forward.load_policy(initialization, config)
    from smolvla_precision import trainable_precision
    precision = trainable_precision(policy, parameter_precision)
    print(f'[参数精度] {precision}', flush=True)
    plan['parameter_precision'] = precision
    forward.write_json(output/"parameter_precision.json", precision)
    policy.to("cuda")
    policy.train()
    scope = training.check_training_scope(policy)
    if policy.model.vlm_with_expert.vlm.training:
        raise ValueError("冻结主干应保持 eval 模式")
    config.device = "cuda"
    preprocessor, postprocessor = forward.make_smolvla_pre_post_processors(config, dataset_stats=stats)
    if sampling == "phase":
        sampler = PhaseBalancedStarts(phase_groups, seed=0)
    elif sampling == "transition":
        sampler = TransitionMixedStarts(train_data.groups, transition_groups, seed=0)
    elif sampling=='active':
        sampler=BalancedStarts(active_groups,seed=0)
    else:
        sampler = BalancedStarts(train_data.groups, seed=0)
    phase_counts = {task: dict.fromkeys(phases, 0) for task, phases in phase_groups.items()}
    transition_sets = {task: set(rows) for task, rows in transition_groups.items()}
    pool_counts = {"all": 0, "transition": 0} if sampling == "transition" else {}
    per_task_pools = {task: {"all": 0, "transition": 0} for task in transition_groups}
    pending_pools, transition_sample_count = [], 0

    def next_batch():
        nonlocal transition_sample_count
        task, index = sampler.draw()
        if sampling in ("phase",'active'):
            phase_counts[task][phase_labels[index]] += 1
        if sampling == "transition":
            pending_pools.append(sampler.last_pool)
            pool_counts[sampler.last_pool] += 1
            per_task_pools[task][sampler.last_pool] += 1
            transition_sample_count += index in transition_sets[task]
        raw=train_data.batch(index)
        raw['action']=raw['action'][:,:horizon]
        raw['action_is_pad']=raw['action_is_pad'][:,:horizon]
        if state_noise_rad:
            raw[forward.STATE_KEY]=augment_proprioception(raw[forward.STATE_KEY],state_noise_rad)
        return task, index, preprocessor(raw)

    def evaluate(step):
        results = {}
        for label, data in (("train", train_data), ("validation", validation_data)):
            records = []
            extra = sorted({index for task in coordination_plans.get(label,{}).values() for index in task['indices']})
            cache = {}
            for index in sorted(set(indices[label]+extra)):
                raw = data.batch(index)
                raw['action']=raw['action'][:,:horizon]
                raw['action_is_pad']=raw['action_is_pad'][:,:horizon]
                target = raw["action"].clone()
                state = raw[forward.STATE_KEY][:, -1, :].unsqueeze(1).expand_as(target)
                baseline = single.action_errors(state, target)
                processed = preprocessor(raw)
                generator = torch.Generator(device="cuda").manual_seed(20261107+index)
                probes = [(torch.randn((1, horizon, 32), device="cuda", generator=generator),
                           torch.tensor([t], device="cuda")) for t in (.2, .5, .8)]
                noise = torch.randn((1, horizon, 32), device="cuda", generator=generator)
                result = single.evaluate_fixed(policy, processed, target, postprocessor, probes, noise)
                result.update(dataset_index=index, source_task_id=data.mapping[index]["source_task_id"], baseline=baseline)
                cache[index] = dict(result, target_physical=target[0].tolist(),
                    state_physical=state[0,0].tolist())
                if index not in indices[label]:
                    continue
                result = dict(result)
                if step in (0, steps):
                    result["target_physical"] = target[0].tolist()
                else:
                    del result["prediction_physical"]
                records.append(result)
            groups = {task: aggregate([r for r in records if r["source_task_id"] == task]) for task in data.groups}
            results[label] = {"summary": aggregate(records), "per_original_recording": groups, "samples": records}
            if coordination_eval:
                from wheel_arm_manipulation.smolvla_coordination import coordination_metrics, summarize_coordination
                checks=[]
                for task, specification in coordination_plans[label].items():
                    for index in specification['indices']:
                        value=cache[index]
                        value['coordination']=coordination_metrics(value['prediction_physical'],value['target_physical'],
                            value['state_physical'],fk,specification['grasp_z'],specification['closed_width'])
                        checks.append(value)
                results[label]['coordination']={'summary':summarize_coordination(checks),'samples':checks}
        return {"splits": results}

    optimizer_config = config.get_optimizer_preset()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    with (output/"updates.jsonl").open("x", encoding="utf-8", buffering=1) as updates, \
            (output/"evaluations.jsonl").open("x", encoding="utf-8", buffering=1) as evaluations:
        def log(stream, value):
            stream.write(forward.json.dumps(value, ensure_ascii=False, allow_nan=False)+"\n")
        def log_update(row):
            if sampling == "transition":
                if len(pending_pools) != len(row["samples"]):
                    raise ValueError("混合采样追踪与本次更新的样本不一致")
                row = dict(row, samples=[dict(sample, sampling_pool=pool,
                    is_transition_window=sample["dataset_index"] in transition_sets[sample["source_task_id"]])
                    for sample, pool in zip(row["samples"], pending_pools)])
                pending_pools.clear()
            if sampling in ("phase",'active'):
                row = dict(row, samples=[dict(sample, start_phase=phase_labels[sample["dataset_index"]])
                                         for sample in row["samples"]])
                if "dataset_index" in row:
                    row["start_phase"] = phase_labels[row["dataset_index"]]
            log(updates, row)
        run = run_updates(policy, next_batch, evaluate, optimizer_config, steps, every, "cuda",
                          on_update=log_update, on_evaluation=lambda row: log(evaluations, row),
                          gradient_accumulation_steps=gradient_accumulation_steps,loss_function=loss_function)
    run["sample_draws_per_original_recording_and_start_phase"] = phase_counts
    if sampling == "transition":
        if pool_counts != {"all": steps*2, "transition": steps*2} or pending_pools:
            raise ValueError("混合采样未完成预定的两类样本抽取")
        run["sample_draws_per_sampling_pool"] = pool_counts
        run["sample_draws_per_original_recording_and_sampling_pool"] = per_task_pools
        run["transition_window_sample_draws_including_ordinary_pool"] = transition_sample_count
    if gradient_accumulation_steps == 1:
        run["updates_per_original_recording_and_start_phase"] = phase_counts
    print("[保存] 最后一步模型、预处理器和后处理器；未按验证集挑选最佳权重。", flush=True)
    checkpoint = output/"checkpoint_final"
    policy.save_pretrained(checkpoint, push_to_hub=False)
    preprocessor.save_pretrained(checkpoint, push_to_hub=False)
    postprocessor.save_pretrained(checkpoint, push_to_hub=False)
    with (output/"training_stats.json").open("xb") as stream:
        stream.write((Path(dataset)/"training_stats.json").read_bytes())
    report = {"schema_version": 1, "status": "completed", "scope": "bounded_multi_demonstration_offline_pilot",
        "plan": plan, "training_scope": scope, "parameter_precision": precision,
        "optimizer": asdict(optimizer_config), "scheduler_used": False,
        "run": run, "checkpoint": str(checkpoint), "checkpoint_saved": True, "resume_supported": False,
        "loss_scope": ("normalized_flow_matching_real_5d_plus_cartesian_pitch_temporal_error_metric"
                       if geometry_loss else "normalized_flow_matching_real_5d_only"), "gpu": torch.cuda.get_device_name(),
        "peak_allocated_mib": torch.cuda.max_memory_allocated()/1024**2,
        "peak_reserved_mib": torch.cuda.max_memory_reserved()/1024**2,
        "robot_commands_sent": False, "uploaded": False}
    forward.write_json(output/"training_report.json", report)
    return report


def augment_proprioception(state,standard_deviation):
    """Simulate small joint measurement errors; targets/RGB remain ground truth."""
    scales=state.new_tensor([standard_deviation*.1]+[standard_deviation]*3+[standard_deviation*.05])
    augmented=state+torch.randn_like(state)*scales
    lower=state.new_tensor([-3.14,-1.57,-2.2,-2.6,0.])
    upper=state.new_tensor([3.14,1.57,2.2,2.6,.08])
    return torch.maximum(lower,torch.minimum(upper,augmented))


def main(args=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset")
    parser.add_argument("--assets-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--learning-rate", type=single.positive_learning_rate, default=3e-5)
    parser.add_argument("--coordination-eval", action="store_true", help="增加闭合到抬升窗口的离线评估，不改变训练损失和抽样")
    parser.add_argument('--geometry-loss',action='store_true',help='训练时加入TCP位置、腕部姿态及时间误差约束；不改变推理控制')
    parser.add_argument('--initial-checkpoint',help='从相同冻结归一化统计的已训练权重微调；新建优化器')
    parser.add_argument('--horizon',type=int,choices=(10,50),default=50,help='预测动作块长度；10步为未来一秒，采样起点仍取完整50步池')
    parser.add_argument('--state-noise-rad',type=float,default=0.,help='仅训练输入的关节测量扰动；保持真实图像及动作标签')
    parser.add_argument('--gripper-loss-weight',type=float,default=0.,help='夹爪附加flow损失权重，0～2；原五维及几何指标同时保留')
    parser.add_argument('--parameter-precision',choices=('fp32','checkpoint'),default='fp32',
                        help='可训练参数默认FP32、前向仍BF16 autocast；checkpoint仅供旧精度对照')
    parser.add_argument("--sampling", choices=("recording", "phase", "transition",'active'), default="recording",
                        help="transition 每次更新抽两份全流程、两份参考闭合抬升窗口，需要梯度累积 4")
    parser.add_argument("--gradient-accumulation-steps", type=int, choices=range(1, 17), default=1,
                        help="每次参数更新累积多少个单样本梯度并取平均；默认 1，显存中的微批量始终为 1")
    options = parser.parse_args(args)
    output = Path(options.output_dir).expanduser().resolve()
    try:
        if not 1 <= options.steps <= 2000 or not 1 <= options.eval_every <= options.steps:
            raise ValueError("首轮最多 2000 次更新，评估间隔须介于 1 和总步数之间")
        if output.exists():
            raise FileExistsError(f"输出目录已存在，不会覆盖：{output}")
        for directory in (options.dataset, options.assets_dir):
            if output.is_relative_to(Path(directory).expanduser().resolve()):
                raise ValueError("训练输出应位于数据集和模型素材目录之外")
        torch.set_num_threads(1)
        report = train(options.dataset, options.assets_dir, output, options.steps, options.eval_every,
                       options.learning_rate, options.sampling, options.coordination_eval,
                       options.gradient_accumulation_steps,options.geometry_loss,options.initial_checkpoint,options.horizon,
                       options.state_noise_rad,options.gripper_loss_weight,options.parameter_precision)
    except (OSError, ValueError, RuntimeError, ImportError, KeyError, subprocess.SubprocessError) as error:
        print(f"[离线训练失败] {type(error).__name__}: {error}；已有计划和日志保留，不自动覆盖。", flush=True)
        return 1
    print(f"[训练完成] {report['run']['optimizer_steps']} 次更新；显存张量峰值 {report['peak_allocated_mib']:.1f} MiB。", flush=True)
    print(f"[保存] {output}", flush=True)
    print("[范围] 独立示范的固定子集评估，限于相同场景重复表现；未验证闭环或杯位泛化。", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
