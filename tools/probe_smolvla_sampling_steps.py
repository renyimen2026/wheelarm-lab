"""Compare flow integration accuracy on fixed development validation inputs.

No parameter updates, no ROS connection, and no external test set is used.
"""
import argparse
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent))
import evaluate_smolvla_validation as evaluation
from wheel_arm_manipulation.smolvla_coordination import TCPKinematics

forward,torch=evaluation.forward,evaluation.torch


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset')
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--assets-dir',required=True)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    torch.set_num_threads(1)
    checkpoint,config,data,stats,phases=evaluation.prepare_inputs(args.dataset,args.checkpoint,args.assets_dir)
    policy=forward.load_policy(checkpoint,config).to('cuda').eval()
    config.device='cuda'
    pre,post=forward.make_smolvla_pre_post_processors(config,dataset_stats=stats)
    fk=TCPKinematics(forward.read_json(checkpoint.parent/'tcp_kinematics.json'))
    versions={name:p._version for name,p in policy.named_parameters()}
    records={str(n):[] for n in [10,20,40]}
    for index in data.evaluation_indices(config.chunk_size):
        raw=data.batch(index)
        target=raw['action'][0,:config.chunk_size].numpy()
        batch={k:v for k,v in pre(raw).items() if k!='action' and not k.startswith('action_')}
        generator=torch.Generator(device='cuda').manual_seed(20261107+index)
        noise=torch.randn((1,config.chunk_size,32),generator=generator,device='cuda')
        for steps in [10,20,40]:
            config.num_steps=steps
            policy.reset()
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                prediction=post(policy.predict_action_chunk(batch,noise=noise.clone())).float().cpu()[0].numpy()
            error=prediction-target
            tcp=fk.positions(prediction)-fk.positions(target)
            value={'index':index,'target':target.tolist(),'prediction':prediction.tolist(),
                'gripper_first_second_mae_mm':float(abs(error[:10,4]).mean()*1000),
                'tcp_first_second_mae_mm':(abs(tcp[:10]).mean(0)*1000).tolist(),
                'pitch_first_second_mae_rad':float(abs(error[:10,1:4].sum(-1)).mean())}
            records[str(steps)].append(value)
    if any(p._version!=versions[n] or p.grad is not None for n,p in policy.named_parameters()):
        raise ValueError('Read-only sampling probe changed model')
    summaries={n:{key:np.mean([r[key] for r in rows],axis=0).tolist() for key in rows[0] if key.endswith(('_mm','_rad'))}
               for n,rows in records.items()}
    forward.write_json(args.output,{'checkpoint':str(checkpoint),'scope':'fixed_development_validation_only',
        'summary':summaries,'records':records,'optimizer_steps':0,'robot_commands_sent':False})
    print(forward.json.dumps(summaries,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
