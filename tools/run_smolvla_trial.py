"""Run one policy in an owned, freshly started, isolated cup simulation.

No MoveIt demonstration runs during the policy trial. Existing simulations are
untouched. All outputs and failed attempts are retained in a new directory.
Run with the system Python after sourcing ROS and the workspace.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from collect_smolvla_demonstrations import launch, run_child, save, stop
from validate_smolvla_release import runtime_source_hashes

REPO = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--dataset', required=True, type=Path)
    parser.add_argument('--assets-dir', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--initial-arm', type=float, nargs=3, default=[-1.05, 2.10, -1.05])
    parser.add_argument('--width', type=float, default=.06)
    parser.add_argument('--domain', type=int, default=181)
    parser.add_argument('--action-steps', choices=(1,4,10), type=int, default=10)
    parser.add_argument('--action-stride',choices=(1,5),type=int,default=1)
    parser.add_argument('--gripper-start-source',choices=('actual','reference'),default='actual')
    parser.add_argument('--seconds', type=int, default=600)
    parser.add_argument('--seed',type=int,default=20261108)
    parser.add_argument('--ensemble-size',type=int,choices=(1,4),default=1)
    parser.add_argument('--gui',action='store_true',help='Also exercise the Gazebo client used in the normal user launch')
    parser.add_argument('--rviz',action='store_true',help='Also exercise RViz visualization load')
    args = parser.parse_args()
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    os.environ['ROS_DOMAIN_ID'] = str(args.domain)
    os.environ['IGN_PARTITION'] = 'wheelarm_trial_'+root.name
    os.environ['PYTHONUNBUFFERED'] = '1'
    plan = {k: str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    plan.update(scope='fresh_fixed_cup_simulation_policy_only',
                runtime_source_sha256=runtime_source_hashes(REPO),
                ros_domain_id=args.domain, ign_partition=os.environ['IGN_PARTITION'])
    save(root/'trial_plan.json', plan)
    # Independent jobs may finish training/evaluation together. Serialize owned
    # simulations on the same ROS domain so controllers cannot cross-discover.
    domain_lock=open(f'/tmp/wheelarm_owned_trial_domain_{args.domain}.lock','a')
    waiting=time.monotonic()
    while True:
        try:
            fcntl.flock(domain_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            break
        except BlockingIOError:
            if time.monotonic()-waiting>1800:
                domain_lock.close()
                raise RuntimeError('Owned simulation domain still occupied after 30 minutes')
            time.sleep(1)
    simulation = None
    started = time.monotonic()
    outcome = {'status':'failed', 'success':False}
    phase='simulation_startup'
    try:
        with (root/'simulation.log').open('x') as log:
            q = args.initial_arm
            simulation = launch(['ros2','launch','wheel_arm_manipulation','cup_test.launch.py',
                'camera_mount:=mast','cup_model:=ceramic_cup',
                f'gui:={str(args.gui).lower()}',f'rviz:={str(args.rviz).lower()}',
                f'initial_shoulder_lift:={q[0]}',f'initial_elbow:={q[1]}',
                f'initial_wrist_pitch:={q[2]}'],log)
            phase='controller_preparation'
            run_child(['/usr/bin/python3','-s',str(REPO/'tools/collect_smolvla_demonstrations.py'),
                '--worker','prepare','--width',str(args.width)],root/'prepare.log',110)
            phase='policy_trial'
            run_child([str(REPO/'.venv-smolvla/bin/python'),'-I',str(REPO/'tools/probe_smolvla_live.py'),
                str(args.dataset),'--checkpoint',str(args.checkpoint),'--assets-dir',str(args.assets_dir),
                '--execute','--action-steps',str(args.action_steps),'--action-stride',str(args.action_stride),
                '--gripper-start-source',args.gripper_start_source,
                '--trial-seconds',str(args.seconds),
                '--max-cycles','80','--verify-every','3','--hold-seconds','10',
                '--seed',str(args.seed),
                '--ensemble-size',str(args.ensemble_size),
                '--output-dir',str(root/'policy')],
                root/'policy.log',args.seconds+160)
        outcome['status']='completed'
    except (RuntimeError,subprocess.TimeoutExpired) as error:
        outcome['error']=str(error)
        outcome['failed_stage']=phase
    finally:
        try:
            stop(simulation)
        finally:
            if simulation is not None:
                try: os.killpg(simulation.pid,signal.SIGTERM)
                except ProcessLookupError: pass
        report_path=root/'policy/live_report.json'
        if report_path.exists():
            report=json.loads(report_path.read_text())
            outcome.update(completed_cycles=report['completed_cycles'],
                end_reason=report['end_reason'],lift_evidence=report['lift_evidence'],
                policy_error=report['error'])
            # Geometry is reported separately; a maintained grasp still requires
            # the explicit hold test before this harness can call it success.
            outcome['lift_verified']=report['lift_evidence'].get('lift_verified')
            outcome['success']=report.get('task_success',False)
            motions=[r['motion'] for r in report['samples'] if r.get('motion',{}).get('status')=='completed'
                     and r['motion'].get('gripper_tcp_base_m')]
            if motions:
                plane=report.get('cup_before',{}).get('center_base_m',[0,0,0])[2]
                opened=[m for m in motions if m['actual_after'][4]>.053]
                outcome['motion_summary']={'last_tcp_m':motions[-1]['gripper_tcp_base_m'],
                    'last_width_mm':motions[-1]['actual_after'][4]*1000,
                    'maximum_tcp_rise_with_open_gripper_mm':max([0.]+[
                        (m['gripper_tcp_base_m'][2]-plane)*1000 for m in opened])}
        outcome['elapsed_wall_seconds']=time.monotonic()-started
        save(root/'trial_result.json',outcome)
        brief={k:v for k,v in outcome.items() if k!='policy_error'}
        if outcome.get('policy_error'):brief['policy_error']=outcome['policy_error'][:240]
        print(json.dumps(brief,ensure_ascii=False),flush=True)
        domain_lock.close()
    return 0 if outcome['status']=='completed' else 1


if __name__=='__main__':
    raise SystemExit(main())
