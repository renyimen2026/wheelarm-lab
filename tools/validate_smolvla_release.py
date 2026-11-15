"""Create a local release manifest only after independent simulated grasp/hold trials."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from vla_trial_summary import hold_evidence,lift_evidence


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def runtime_source_hashes(repo):
    paths=['tools/probe_smolvla_live.py','tools/execute_vla_short_motion.py',
           'tools/ros_probe_session.py','tools/vla_trial_summary.py',
           'tools/probe_smolvla_forward.py','tools/smolvla_precision.py']
    paths += ['wheel_arm_manipulation/wheel_arm_manipulation/'+name+'.py' for name in
              ['arm_goal','vla_short_motion','planning','observation_capture','controller_reference','image_qos']]
    return {str((repo/path).resolve()):digest(repo/path) for path in paths}


def validate(trials,assets):
    if len(trials)<5 or len({str(Path(t).resolve()) for t in trials})!=len(trials):
        raise ValueError('至少五个不同的新仿真测试目录')
    cases=[]
    for trial in trials:
        root=Path(trial).resolve(strict=True)
        plan=json.loads((root/'trial_plan.json').read_text())
        result=json.loads((root/'trial_result.json').read_text())
        report_path=root/'policy/live_report.json'
        report=json.loads(report_path.read_text())
        sources=plan.get('runtime_source_sha256',{})
        if not sources or any(digest(path)!=value for path,value in sources.items()):
            raise ValueError('测试运行代码缺少版本记录或已被修改')
        if plan.get('action_stride',1)!=report.get('action_stride',1):
            raise ValueError('测试计划和实际动作时间间隔不一致')
        if plan.get('gripper_start_source','actual')!=report.get('gripper_start_source','actual'):
            raise ValueError('测试计划和实际夹爪衔接方式不一致')
        hold=hold_evidence(report['cup_before'],report.get('hold_samples',[]),10.)
        final=lift_evidence(report['cup_before'],report.get('cup_after'))
        if (result.get('success') is not True or report.get('task_success') is not True
                or report.get('status')!='completed' or report.get('error')
                or report.get('end_reason')!='lift_and_hold_verified'
                or not hold['task_success'] or final['lift_verified'] is not True
                or abs(final['cup_top_delta_m']-.08)>.01):
            raise ValueError(f'尚未完成目标高度及十秒保持：{root}')
        weight=Path(report['checkpoint'])/'model.safetensors'
        if report.get('checkpoint_sha256')!=digest(weight):
            raise ValueError(f'测试后的权重已变化或缺少哈希：{root}')
        cases.append({'trial_dir':str(root),'plan':plan,'report_sha256':digest(report_path),
            'runtime_source_sha256':sources,
            'checkpoint_sha256':report['checkpoint_sha256'],'hold_evidence':hold,
            'seed':report['sampling_seed'],'ensemble_size':report['ensemble_size']})
    first=cases[0]['plan']
    if any(c['runtime_source_sha256']!=cases[0]['runtime_source_sha256'] for c in cases):
        raise ValueError('验收不能混用不同执行代码版本')
    if (len({c['seed'] for c in cases})<3 or len({tuple(c['plan']['initial_arm']) for c in cases})<2
            or len({c['plan']['ign_partition'] for c in cases})!=len(cases)):
        raise ValueError('验收需要不同随机种子、初始姿态和独立仿真实例')
    if any((c['checkpoint_sha256'],c['plan']['dataset'],c['ensemble_size'],c['plan']['action_steps'],c['plan'].get('action_stride',1),c['plan'].get('gripper_start_source','actual'))
           !=(cases[0]['checkpoint_sha256'],first['dataset'],cases[0]['ensemble_size'],first['action_steps'],first.get('action_stride',1),first.get('gripper_start_source','actual')) for c in cases):
        raise ValueError('验收不能混用不同权重、数据归一化或推理配置')
    return {'schema_version':1,'status':'ready_for_user_simulation_acceptance',
        'scope':'fixed_ceramic_cup_mast_camera_table; no cup-position generalization or real-hardware claim',
        'checkpoint':first['checkpoint'],'checkpoint_sha256':cases[0]['checkpoint_sha256'],
        'dataset':first['dataset'],'assets_dir':str(Path(assets).resolve(strict=True)),
        'ensemble_size':cases[0]['ensemble_size'],'action_steps':first['action_steps'],
        'action_stride':first.get('action_stride',1),
        'gripper_start_source':first.get('gripper_start_source','actual'),
        'runtime_source_sha256':cases[0]['runtime_source_sha256'],
        'default_seed':cases[0]['seed'],
        'verified_trials':cases,'uploaded':False,'committed':False}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trials',nargs='+',required=True,type=Path)
    parser.add_argument('--assets-dir',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    report=validate(args.trials,args.assets_dir)
    with args.output.open('x') as stream:json.dump(report,stream,ensure_ascii=False,indent=2)
    print(f'[验收通过] {len(report["verified_trials"])} 次独立仿真抓取与保持；{args.output}')


if __name__=='__main__':main()
