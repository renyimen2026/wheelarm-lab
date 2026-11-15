"""Controlled hold experiment: repeated actual-start versus reference-start chunks.

A MoveIt expert first lifts the cup. Then arm targets and gripper closing target
stay fixed for all equal-duration chunks. No VLA model or training data is used.
The owned simulation and both controller reference/actual traces are retained.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'wheel_arm_manipulation'))
from collect_smolvla_demonstrations import launch,run_child,save,stop


def worker(args):
    import rclpy
    from rclpy.signals import SignalHandlerOptions
    from control_msgs.msg import JointTrajectoryControllerState
    from wheel_arm_manipulation import cup_grasp
    from wheel_arm_manipulation.observation_capture import extract_state
    from wheel_arm_manipulation.planning import values_from_state
    from wheel_arm_manipulation.vla_short_motion import chunk_trajectory,gripper_reference_start
    from execute_vla_short_motion import fresh_gripper_reference
    rclpy.init(args=['reference_probe','--ros-args','-p','use_sim_time:=true'],
                signal_handler_options=SignalHandlerOptions.NO)
    options=cup_grasp.parser().parse_args(['--execute','--finish','hold','--hold-seconds','10',
        '--wait','30','--output-dir',str(args.output_dir/'cup_evidence')])
    node=cup_grasp.CupGraspClient(options)
    report={'mode':args.mode,'scope':'expert_lift_then_static_hold_boundary_probe_not_VLA',
            'status':'failed','chunks':[],'measurements':{}}
    trace=[]
    node.gripper_reference_message=None
    segment=['expert']
    def received(message):
        node.gripper_reference_message=message
        point=getattr(message,'reference',None)
        if point is None or not point.positions:point=message.desired
        trace.append({'stamp_ns':message.header.stamp.sec*10**9+message.header.stamp.nanosec,
            'segment':segment[0],'names':list(message.joint_names),
            'reference':list(point.positions),'actual':list(message.actual.positions)})
    subscription=node.create_subscription(JointTrajectoryControllerState,
        '/wheel_arm/gripper_controller/controller_state',received,5)
    original_capture=node.observer.capture
    def capture(*a,**kw):
        cup=original_capture(*a,**kw)
        report['measurements'][kw.get('label','unlabelled')]={'center_base_m':cup.center.tolist(),
            'top_base_m':float(cup.center[2]+cup.height/2),'radius_m':float(cup.radius)}
        return cup
    node.observer.capture=capture
    def interrupted(*_):
        node.stopping=True
        if not node.execution_pending:raise KeyboardInterrupt()
    signal.signal(signal.SIGINT,interrupted);signal.signal(signal.SIGTERM,interrupted)
    try:
        if cup_grasp.run(node,options,confirm=lambda:True)!=0:raise RuntimeError('Expert lift failed')
        node.observer.capture(require_support=False,label='held_before_boundary_test')
        model=copy.deepcopy(node.arm)
        model.joints=node.arm.joints+node.gripper.joints
        for field in ('bounds','velocities','accelerations'):
            getattr(model,field).update(getattr(node.gripper,field))
        model.group='';model.start_tolerance=model.reached_tolerance=.002
        node.model=model
        first=node.snapshot()
        fixed_arm=values_from_state(first,node.arm)
        message=fresh_gripper_reference(node)
        _,reference=gripper_reference_start(values_from_state(first,model),message,node.gripper,
                                           node.get_clock().now().nanoseconds)
        closing_width=reference['gripper_width_m']
        predictions=[[fixed_arm[name] for name in node.arm.joints]+[closing_width] for _ in range(10)]
        report.update(fixed_closing_reference_width_m=closing_width,
                      fixed_arm_goal=fixed_arm,nominal_duration_per_chunk_seconds=5.)
        for i in range(args.cycles):
            segment[0]=f'boundary_{i:02d}'
            current=node.snapshot()
            actual=extract_state(current)
            start=values_from_state(current,model)
            node.valid_state(node.state(current))
            if args.mode=='reference':
                start,reference=gripper_reference_start(start,fresh_gripper_reference(node),
                    node.gripper,node.get_clock().now().nanoseconds)
            trajectory,goal,details=chunk_trajectory(actual,predictions,node.arm,node.gripper,start)
            duration=details['trajectory_duration_seconds']
            stretch=max(1.,5./duration)
            for point in trajectory.joint_trajectory.points:
                ns=round((point.time_from_start.sec+point.time_from_start.nanosec/1e9)*stretch*1e9)
                point.time_from_start.sec=ns//10**9;point.time_from_start.nanosec=ns%10**9
                point.velocities=[v/stretch for v in point.velocities]
            node.options.timeout=max(15.,duration*stretch+5.)
            node.execute(trajectory,start,goal)
            row={'cycle':i+1,'actual_width_before_m':actual[4],
                'first_command_width_m':sum(trajectory.joint_trajectory.points[0].positions[-2:]),
                'closing_target_width_m':closing_width,'duration_seconds':duration*stretch,
                'actual_after':extract_state(node.joints)}
            report['chunks'].append(row)
            print('[边界保持]',json.dumps(row),flush=True)
        node.observer.capture(require_support=False,label='held_after_boundary_test')
        before=report['measurements']['held_before_boundary_test']['top_base_m']
        after=report['measurements']['held_after_boundary_test']['top_base_m']
        report.update(status='completed',cup_height_loss_during_probe_m=before-after)
    except (RuntimeError,ValueError,KeyboardInterrupt) as error:
        report['error']=f'{type(error).__name__}: {error}'
    finally:
        save(args.output_dir/'reference_trace.json',trace)
        save(args.output_dir/'probe_report.json',report)
        node.destroy_node()
        if rclpy.ok():rclpy.shutdown()
    return 0 if report['status']=='completed' else 1


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('actual','reference'),required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--cycles',type=int,default=12)
    parser.add_argument('--worker',action='store_true')
    args=parser.parse_args()
    if not 1<=args.cycles<=20:raise ValueError('Use 1–20 hold chunks')
    if args.worker:return worker(args)
    root=args.output_dir.resolve();root.mkdir(parents=True,exist_ok=False)
    os.environ['ROS_DOMAIN_ID']='181'
    os.environ['IGN_PARTITION']='wheelarm_boundary_'+hashlib.sha256(str(root).encode()).hexdigest()[:12]
    simulation=None
    try:
        with (root/'simulation.log').open('x') as log:
            simulation=launch(['ros2','launch','wheel_arm_manipulation','cup_test.launch.py',
                'camera_mount:=mast','cup_model:=ceramic_cup','gui:=false','rviz:=false'],log)
            run_child(['/usr/bin/python3','-s',str(REPO/'tools/collect_smolvla_demonstrations.py'),
                       '--worker','prepare','--width','.06'],root/'prepare.log',110)
            run_child(['/usr/bin/python3','-s',str(Path(__file__).resolve()),'--worker',
                '--mode',args.mode,'--output-dir',str(root),'--cycles',str(args.cycles)],root/'worker.log',600)
    finally:
        try:stop(simulation)
        finally:
            if simulation is not None:
                try:os.killpg(simulation.pid,signal.SIGTERM)
                except ProcessLookupError:pass
    return 0


if __name__=='__main__':raise SystemExit(main())
