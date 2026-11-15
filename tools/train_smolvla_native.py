"""Run pinned LeRobot's native SmolVLA trainer on verified local training data.

Reuse its batch loader, 32D flow loss, time-padding mask, AdamW, cosine warmup,
checkpointing and Accelerate BF16 execution. Adapt only local features/assets,
frozen normalization and FP32 storage of optimizer-owned parameters. No ROS,
teacher policy, uploads, or validation samples participate in training.
"""
import argparse
from dataclasses import asdict
from pathlib import Path
import shutil
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_smolvla_batch as pilot
from smolvla_precision import restore_trainable_precision, trainable_precision

forward, torch = pilot.forward, pilot.torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path)
    parser.add_argument('--assets-dir', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--batch-size', type=int, choices=(2,4,8), default=8)
    parser.add_argument('--save-every', type=int, default=500)
    parser.add_argument('--initial-checkpoint',type=Path,
                        help='从已完成、相同冻结统计和素材的50步权重微调；重新创建优化器')
    parser.add_argument('--learning-rate',type=float,default=1e-4)
    parser.add_argument('--project-delta-columns',action='store_true',
                        help='按所需列读取未来动作；保留原始当前RGB/状态/文本及官方采样和padding')
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if not 1 <= args.steps <= 20000 or args.save_every < 1 or output.exists():
        raise ValueError('Use 1–20000 steps and a new output directory')
    if not 2.5e-6 <= args.learning_rate <= 1e-4:
        raise ValueError('Native cosine recipe requires peak LR 2.5e-6–1e-4')
    for source in (args.dataset.resolve(), args.assets_dir.resolve()):
        if output.is_relative_to(source):
            raise ValueError('Training output must be outside input sources')
    if forward.version('lerobot') != '0.4.4' or not torch.cuda.is_bf16_supported():
        raise ValueError('Requires pinned LeRobot 0.4.4 and a BF16 CUDA GPU')
    torch.set_num_threads(1)
    policy_dir, vlm, revisions = forward.load_assets(args.assets_dir)
    train_data, validation_data, stats = pilot.load_batch(args.dataset)
    initialization=policy_dir
    previous=None
    if args.initial_checkpoint is not None:
        initialization=args.initial_checkpoint.resolve(strict=True)
        previous=forward.read_json(initialization.parent/'training_report.json')
        if (previous.get('status')!='completed' or not previous.get('checkpoint_saved')
                or Path(previous['checkpoint']).resolve()!=initialization
                or previous['plan'].get('chunk_size')!=50
                or previous['plan']['revisions']!=revisions
                or previous['plan']['normalization_sha256']!=forward.read_json(
                    args.dataset/'batch_conversion_report.json')['normalization_stats']['sha256']):
            raise ValueError('Fine-tuning requires verified 50-step weights, matching revisions and frozen statistics')
    config, _ = forward.adapt_config(forward.PreTrainedConfig.from_pretrained(
        initialization, local_files_only=True), train_data.base.meta, vlm)
    config.chunk_size = 50
    config.n_action_steps = 10
    config.device = 'cuda'
    config.pretrained_path = initialization
    config.optimizer_lr = args.learning_rate
    from accelerate import Accelerator
    from lerobot.configs.default import DatasetConfig
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.scripts import lerobot_train as native
    from lerobot.utils.train_utils import get_step_checkpoint_dir

    dataset = DatasetConfig(repo_id=train_data.manifest['repo_id'], root=str(train_data.root),
                            use_imagenet_stats=False)
    dataset.image_transforms.enable = False
    cfg = TrainPipelineConfig(dataset=dataset, policy=config, output_dir=output,
        seed=0, batch_size=args.batch_size, steps=args.steps, num_workers=0,
        eval_freq=0, log_freq=20, save_freq=min(args.steps,args.save_every), job_name='wheelarm_smolvla_native')
    cfg.wandb.enable = False
    plan = {'schema_version':1, 'dataset':str(args.dataset.resolve()),
        'assets':str(args.assets_dir.resolve()), 'revisions':revisions, 'steps':args.steps,
        'batch_size':args.batch_size, 'effective_batch_size':args.batch_size, 'chunk_size':50,
        'normalization_sha256':forward.read_json(args.dataset/'batch_conversion_report.json')['normalization_stats']['sha256'],
        'normalization_source':'same_frozen_training_stats_only',
        'sampling':'native_shuffled_training_frames_including_masked_episode_tails',
        'source_recordings':len(train_data.groups), 'source_training_rows':len(train_data.base),
        'loss':'official_LeRobot_0.4.4_32d_flow_with_time_padding_mask',
        'custom_geometry_loss':False, 'state_augmentation':False,
        'autocast':'BF16', 'trainable_storage':'FP32', 'validation_used_for_updates':False,
        'scheduler':asdict(config.get_scheduler_preset())}
    plan.update(initial_checkpoint=str(initialization),optimizer_state_restored=False,
                learning_rate=args.learning_rate,
                delta_column_projection=args.project_delta_columns)
    if args.project_delta_columns:
        from smolvla_dataset_projection import project_delta_columns
        original_make_dataset=native.make_dataset
        def projected_dataset(*a,**kw):
            return project_delta_columns(original_make_dataset(*a,**kw))
        native.make_dataset=projected_dataset
    original_make_policy = native.make_policy
    context = {'evaluations':[]}

    def evaluate(policy, step):
        pre, post = forward.make_smolvla_pre_post_processors(config, dataset_stats=stats)
        was_training = policy.training
        versions = {n:p._version for n,p in policy.named_parameters()}
        records = []
        try:
            policy.eval()
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]), torch.no_grad():
                for index in validation_data.evaluation_indices(50):
                    raw = validation_data.batch(index)
                    target = raw['action'].clone()
                    processed = pre(raw)
                    generator = torch.Generator(device='cuda').manual_seed(20261107+index)
                    probes = [(torch.randn((1,50,32),device='cuda',generator=generator),
                               torch.tensor([t],device='cuda')) for t in (.2,.5,.8)]
                    noise = torch.randn((1,50,32),device='cuda',generator=generator)
                    value = pilot.single.evaluate_fixed(policy,processed,target,post,probes,noise)
                    value.update(dataset_index=index,target_physical=target[0].tolist())
                    prediction = target.new_tensor(value['prediction_physical'])[None]
                    value['first_second_errors'] = pilot.single.action_errors(prediction[:,:10],target[:,:10])
                    records.append(value)
        finally:
            policy.train(was_training)
        if any(p._version != versions[n] or p.grad is not None for n,p in policy.named_parameters()):
            raise ValueError('Read-only development evaluation changed parameters or gradients')
        row = {'step':step,'scope':'fixed_held_out_development_anchors_not_full_validation',
            'robot_5d_flow_loss':sum(v['fixed_flow_loss'] for v in records)/len(records),
            'first_second_gripper_mae_mm':sum(v['first_second_errors']['gripper_mae_mm'] for v in records)/len(records),
            'full_horizon_gripper_mae_mm':sum(v['physical_errors']['gripper_mae_mm'] for v in records)/len(records),
            'samples':records}
        if (step==0 and previous is not None and previous.get('native_trainer')=='lerobot.scripts.lerobot_train.train@0.4.4'
                and Path(previous['plan']['dataset']).resolve()==args.dataset.resolve()):
            prior={v['dataset_index']:v['prediction_physical'] for v in previous['run']['evaluations'][-1]['samples']}
            if set(prior)!={v['dataset_index'] for v in records}:
                raise ValueError('Initial-checkpoint evaluation anchors changed')
            difference=max(float(abs(torch.tensor(v['prediction_physical'])-
                torch.tensor(prior[v['dataset_index']])).max()) for v in records)
            if difference>1e-5:
                raise ValueError(f'Loaded fine-tuning predictions differ from saved checkpoint: {difference}')
            row['initialization_parity_max_abs_difference']=difference
        context['evaluations'].append(row)
        with (output/'native_evaluations.jsonl').open('a') as stream:
            stream.write(forward.json.dumps(row,allow_nan=False)+'\n')
        print(f'[原生验证 {step}] 首秒夹爪MAE={row["first_second_gripper_mae_mm"]:.3f}mm；'
              f'完整50步MAE={row["full_horizon_gripper_mae_mm"]:.3f}mm。',flush=True)

    def adapted_policy(*a, **kw):
        policy = original_make_policy(*a, **kw)
        restore_trainable_precision(policy, initialization)
        context['precision'] = trainable_precision(policy)
        context['scope'] = pilot.training.check_training_scope(policy)
        output.mkdir(parents=True, exist_ok=True)
        forward.write_json(output/'run_plan.json', plan)
        forward.write_json(output/'parameter_precision.json', context['precision'])
        print('[原生训练适配] 本地完整权重；可训练参数FP32；官方32维flow loss、批训练和cosine预热。', flush=True)
        evaluate(policy,0)
        return policy

    native.make_policy = adapted_policy
    # The downloaded base has weights/config only. Build robot-specific processors
    # from the already frozen training statistics rather than base-robot features.
    native.make_pre_post_processors = lambda **kw: forward.make_smolvla_pre_post_processors(
        config, dataset_stats=stats)
    original_save = native.save_checkpoint
    def save_and_evaluate(*a, **kw):
        result = original_save(*a, **kw)
        evaluate(kw['policy'],kw['step'])
        return result
    native.save_checkpoint = save_and_evaluate
    accelerator = Accelerator(mixed_precision='bf16', step_scheduler_with_optimizer=False)
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    native.train(cfg, accelerator=accelerator)
    saved = get_step_checkpoint_dir(output, args.steps, args.steps)/'pretrained_model'
    checkpoint = output/'checkpoint_final'
    shutil.copytree(saved, checkpoint)
    shutil.copyfile(args.dataset/'training_stats.json', output/'training_stats.json')
    report = {'schema_version':1, 'status':'completed', 'checkpoint_saved':True,
        'checkpoint':str(checkpoint), 'plan':plan, 'training_scope':context['scope'],
        'parameter_precision':context['precision'], 'optimizer':asdict(cfg.optimizer),
        'scheduler_used':True, 'native_trainer':'lerobot.scripts.lerobot_train.train@0.4.4',
        'loss_scope':'official_32d_flow_matching_time_masked',
        'run':{'optimizer_steps':args.steps,'evaluations':context['evaluations']},
        'elapsed_wall_seconds':time.monotonic()-started,
        'peak_allocated_mib':torch.cuda.max_memory_allocated()/1024**2,
        'robot_commands_sent':False, 'uploaded':False, 'resume_supported':False,
        'native_resume_checkpoint':str(saved.parent)}
    report['task_finetuning_updates_in_lineage']=args.steps+(0 if previous is None else
        previous.get('task_finetuning_updates_in_lineage',previous['run']['optimizer_steps']))
    forward.write_json(output/'training_report.json', report)
    print(f'[原生训练完成] {args.steps} 次更新；批量{args.batch_size}；峰值{report["peak_allocated_mib"]:.1f}MiB；{output}', flush=True)


if __name__ == '__main__':
    main()
