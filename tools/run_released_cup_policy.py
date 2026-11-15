"""Run the exact locally validated cup policy configuration in an existing simulation."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
import uuid

sys.path.insert(0,str(Path(__file__).resolve().parent))
from validate_smolvla_release import digest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release',required=True,type=Path)
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--seed',type=int,help='默认使用第一轮预先指定且通过验收的采样种子')
    args=parser.parse_args()
    release=json.loads(args.release.read_text())
    if (release.get('status')!='ready_for_user_simulation_acceptance'
            or len(release.get('verified_trials',[]))<5
            or digest(Path(release['checkpoint'])/'model.safetensors')!=release['checkpoint_sha256']):
        raise ValueError('需要通过完整验收且权重未变化的本地发布清单')
    if (not release.get('runtime_source_sha256') or any(digest(path)!=value
            for path,value in release['runtime_source_sha256'].items())):
        raise ValueError('执行代码与验收版本不同，需要重新验证')
    output=args.output_dir
    if output is None:
        workspace=Path(__file__).resolve().parents[3]
        stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8]
        output=workspace/'observations/diagnostics'/f'cup_vla_user_{stamp}'
    print(f'[交付配置] {release["checkpoint"]}；固定杯子/桌面/高位相机场景；请从全新仿真起点运行。',flush=True)
    sys.argv=['probe_smolvla_live',release['dataset'],'--checkpoint',release['checkpoint'],
        '--assets-dir',release['assets_dir'],'--execute','--action-steps',str(release['action_steps']),
        '--action-stride',str(release.get('action_stride',1)),
        '--gripper-start-source',release.get('gripper_start_source','actual'),
        '--ensemble-size',str(release['ensemble_size']),
        '--seed',str(args.seed if args.seed is not None else release.get('default_seed',20261108)),
        '--trial-seconds','600','--max-cycles','80','--verify-every','3','--hold-seconds','10',
        '--output-dir',str(output)]
    import probe_smolvla_live
    return probe_smolvla_live.main()


if __name__=='__main__':raise SystemExit(main())
