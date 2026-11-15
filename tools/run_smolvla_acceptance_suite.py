"""Five predeclared fresh simulation cases, after a successful regression trial.

Run with system Python after sourcing ROS/workspace. A failed case ends this
suite; no replacement runs or success-only selection are permitted.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from collect_smolvla_demonstrations import save
from validate_smolvla_release import digest,validate
from vla_trial_summary import hold_evidence

REPO=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',required=True,type=Path)
    parser.add_argument('--checkpoint',required=True,type=Path)
    parser.add_argument('--assets-dir',required=True,type=Path)
    parser.add_argument('--regression-trial',required=True,type=Path)
    parser.add_argument('--output-dir',required=True,type=Path)
    parser.add_argument('--first-seed',type=int,default=20261126)
    parser.add_argument('--ensemble-size',type=int,choices=(1,4),default=4)
    parser.add_argument('--action-stride',type=int,choices=(1,5),default=1)
    parser.add_argument('--gripper-start-source',choices=('actual','reference'),default='actual')
    args=parser.parse_args()
    root=args.output_dir.resolve()
    root.mkdir(parents=True,exist_ok=False)
    cases=[{'case':i+1,'seed':args.first_seed+i,
        'initial_arm':[-1.05,2.10,-1.05] if i%2==0 else [-1.03,2.08,-1.05],
        'directory':str(root/f'{root.name}_case_{i+1:02d}')} for i in range(5)]
    weight_hash=digest(args.checkpoint/'model.safetensors')
    plan={'schema_version':1,'scope':'all_five_predeclared_cases_no_replacements',
        'checkpoint':str(args.checkpoint.resolve()),'checkpoint_sha256':weight_hash,
        'dataset':str(args.dataset.resolve()),'assets_dir':str(args.assets_dir.resolve()),
        'ensemble_size':args.ensemble_size,'action_stride':args.action_stride,'action_steps':10,
        'gripper_start_source':args.gripper_start_source,
        'cases':cases,'regression_trial':str(args.regression_trial.resolve()),
        'required_lift_m':.08,'height_tolerance_m':.01,'hold_seconds':10}
    save(root/'suite_plan.json',plan)
    result={'status':'waiting_for_regression','cases':[]}
    def status():
        temporary=root/'suite_result.tmp'
        temporary.write_text(json.dumps(result,ensure_ascii=False,indent=2))
        temporary.replace(root/'suite_result.json')
    status()
    deadline=time.monotonic()+900
    regression=args.regression_trial/'trial_result.json'
    while not regression.exists() and time.monotonic()<deadline:time.sleep(5)
    if not regression.exists() or json.loads(regression.read_text()).get('success') is not True:
        result.update(status='not_ready_regression_failed_or_incomplete');status();return 1
    report=json.loads((args.regression_trial/'policy/live_report.json').read_text())
    prior_plan=json.loads((args.regression_trial/'trial_plan.json').read_text())
    if (report['checkpoint_sha256']!=weight_hash or prior_plan['dataset']!=str(args.dataset.resolve())
            or report['ensemble_size']!=args.ensemble_size
            or report.get('action_stride',1)!=args.action_stride
            or report.get('gripper_start_source','actual')!=args.gripper_start_source
            or prior_plan.get('gripper_start_source','actual')!=args.gripper_start_source
            or not hold_evidence(report['cup_before'],report['hold_samples'],10.)['task_success']):
        raise ValueError('Regression evidence/configuration differs from this fixed suite')
    result['regression_verified']=True
    for case in cases:
        destination=Path(case['directory'])
        result.update(status='running_case',active_case=case['case']);status()
        command=[sys.executable,'-s',str(REPO/'tools/run_smolvla_trial.py'),
            '--dataset',str(args.dataset),'--checkpoint',str(args.checkpoint),'--assets-dir',str(args.assets_dir),
            '--ensemble-size',str(args.ensemble_size),'--action-stride',str(args.action_stride),
            '--gripper-start-source',args.gripper_start_source,
            '--seed',str(case['seed']),'--initial-arm',*[str(q) for q in case['initial_arm']],
            '--output-dir',str(destination)]
        with (root/f'case_{case["case"]:02d}.log').open('x') as stream:
            completed=subprocess.run(command,cwd=REPO,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
        path=destination/'trial_result.json'
        outcome=json.loads(path.read_text()) if path.exists() else {'success':False,'error':'missing_trial_result'}
        result['cases'].append(dict(case,returncode=completed.returncode,result=outcome));status()
        if completed.returncode!=0 or outcome.get('success') is not True:
            result.update(status='not_ready_predeclared_case_failed',failed_case=case['case']);status();return 1
    release=validate([Path(c['directory']) for c in cases],args.assets_dir)
    release.update(acceptance_plan=str(root/'suite_plan.json'),acceptance_plan_sha256=digest(root/'suite_plan.json'))
    save(root/'release_manifest.json',release)
    result.update(status='ready_for_user_simulation_acceptance',release_manifest=str(root/'release_manifest.json'))
    status()
    print('[五轮验收通过]',root/'release_manifest.json',flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
