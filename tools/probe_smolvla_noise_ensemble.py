"""Compare single/4-sample predictions on recorded live observations, read-only."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent))
import probe_smolvla_live as live

forward,torch=live.forward,live.torch


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset',type=Path)
    parser.add_argument('--checkpoint',required=True,type=Path)
    parser.add_argument('--assets-dir',required=True,type=Path)
    parser.add_argument('--rollout',required=True,type=Path)
    parser.add_argument('--indices',type=int,nargs='+',required=True)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--compare-checkpoint',action='store_true',
                        help='允许不同的已验证训练权重重放同一输入；不宣称复现原预测')
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    torch.set_num_threads(1)
    checkpoint,config,data,stats,_=live.validation.prepare_inputs(args.dataset,args.checkpoint,args.assets_dir)
    old=forward.read_json(args.rollout/'live_report.json')
    same_checkpoint=Path(old['checkpoint']).resolve()==checkpoint
    if ((not same_checkpoint and not args.compare_checkpoint) or old['ensemble_size']!=1
            or old['checkpoint_sha256']!=live.validation.checkpoint_digest(Path(old['checkpoint'])/'model.safetensors')):
        raise ValueError('Need unchanged weights and a single-sample source rollout')
    policy=forward.load_policy(checkpoint,config).to('cuda').eval()
    config.device='cuda'
    pre,post=forward.make_smolvla_pre_post_processors(config,dataset_stats=stats)
    versions={n:p._version for n,p in policy.named_parameters()}
    records=[]
    for i in args.indices:
        source=old['samples'][i]
        raw,meta=live.observation_batch(source['observation_directory'],data.manifest['instruction'])
        base={k:v for k,v in pre(raw).items() if k!='action' and not k.startswith('action_')}
        predictions={}
        for size in (1,4):
            processed={k:(v.expand(size,*v.shape[1:]) if isinstance(v,torch.Tensor) and v.ndim and v.shape[0]==1
                          else v*size if isinstance(v,list) and len(v)==1 else v) for k,v in base.items()}
            noises=[]
            for j in range(size):
                generator=torch.Generator(device='cuda').manual_seed(old['sampling_seed']+i+j*1000003)
                noises.append(torch.randn((1,config.chunk_size,32),device='cuda',generator=generator))
            policy.reset()
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                members=post(policy.predict_action_chunk(processed,noise=torch.cat(noises))).float().cpu()
            forward.require_tensor(members,(size,config.chunk_size,5),'ensemble predictions')
            predictions[str(size)]={'members_physical':members.tolist(),'mean_physical':members.mean(0).tolist(),
                'last_executed_width_mm':float(members.mean(0)[9,4]*1000),
                'member_last_executed_widths_mm':(members[:,9,4]*1000).tolist()}
        original=torch.tensor(source['prediction_physical'])
        single=torch.tensor(predictions['1']['mean_physical'])
        error=float(abs(original-single).max())
        if same_checkpoint and error>1e-5:raise ValueError(f'Single prediction differs from logged rollout: {error}')
        row={'sample_index':i,'state_physical':source['state_physical'],
             'single_reproduction_max_abs_difference':error if same_checkpoint else None,
             'different_checkpoint_prediction_max_abs_difference':error if not same_checkpoint else None,
             'predictions':predictions}
        records.append(row)
        print(f'[采样对照] {i}；实际{raw[forward.STATE_KEY][0,0,4]*1000:.3f}mm；'
              f'单条末步{predictions["1"]["last_executed_width_mm"]:.3f}mm；'
              f'四条均值{predictions["4"]["last_executed_width_mm"]:.3f}mm。',flush=True)
    if any(p._version!=versions[n] or p.grad is not None for n,p in policy.named_parameters()):
        raise ValueError('Read-only probe changed parameters or gradients')
    forward.write_json(args.output,{'scope':'same_recorded_inputs_noise_ensemble_only_no_teacher_or_ROS',
        'checkpoint':str(checkpoint),'checkpoint_sha256':live.validation.checkpoint_digest(checkpoint/'model.safetensors'),
        'source_checkpoint':old['checkpoint'],'source_checkpoint_sha256':old['checkpoint_sha256'],
        'same_checkpoint':same_checkpoint,
        'records':records,'optimizer_steps':0,'robot_commands_sent':False})


if __name__=='__main__':main()
