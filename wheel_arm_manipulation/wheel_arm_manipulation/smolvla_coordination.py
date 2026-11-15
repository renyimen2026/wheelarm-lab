"""Offline closure/lift diagnostics; no ROS or robot control dependencies."""
import json
from pathlib import Path

import numpy as np
from wheel_arm_manipulation.lerobot_conversion import digest
from wheel_arm_manipulation.action_alignment import read_jsonl
from wheel_arm_manipulation.task_phases import parse_phases, phase_at


class TCPKinematics:
    def __init__(self, spec):
        self.spec = spec
        self.names = spec['joint_names']
        if len(self.names)!=4 or spec['frame']!='base_footprint' or spec['tip']!='gripper_tcp':
            raise ValueError('TCP chain does not match four-joint wheelarm')
        if [r['name'] for r in spec['chain'] if r['kind']=='revolute'] != self.names:
            raise ValueError('TCP joint order differs from state/action order')

    def positions(self, vectors):
        q=np.asarray(vectors,dtype=float)
        if q.shape[-1] not in (4,5) or not np.isfinite(q).all():
            raise ValueError('FK requires finite [...,4/5] physical joints')
        transform=np.broadcast_to(np.eye(4),q.shape[:-1]+(4,4)).copy()
        for link in self.spec['chain']:
            transform=transform@np.asarray(link['origin'])
            if link['kind']=='revolute':
                axis=np.asarray(link['axis'],dtype=float)
                if not np.isclose(np.linalg.norm(axis),1):
                    raise ValueError('Invalid URDF joint axis')
                x,y,z=axis
                skew=np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
                theta=q[...,self.names.index(link['name'])]
                rotation=np.broadcast_to(np.eye(4),q.shape[:-1]+(4,4)).copy()
                rotation[...,:3,:3]=np.eye(3)+np.sin(theta)[...,None,None]*skew+(1-np.cos(theta))[...,None,None]*(skew@skew)
                transform=transform@rotation
            elif link['kind']!='fixed':
                raise ValueError('Unsupported TCP chain joint')
        return transform[...,:3,3]


def onset(heights, plane, rise=.010, consecutive=3):
    mask=np.asarray(heights)>=plane+rise
    for i in range(len(mask)-consecutive+1):
        if mask[i:i+consecutive].all():
            return i
    return None


def coordination_metrics(prediction, target, state, fk, grasp_z, closed_width):
    prediction,target,state=map(lambda v:np.asarray(v,dtype=float),(prediction,target,state))
    if prediction.shape!=(50,5) or target.shape!=prediction.shape or state.shape!=(5,):
        raise ValueError('Coordination needs physical [50,5] trajectories and [5] state')
    if not all(np.isfinite(v).all() for v in (prediction,target,state)):
        raise ValueError('Nonfinite coordination inputs')
    if not np.isfinite([grasp_z,closed_width]).all() or not 0 < closed_width <= .08:
        raise ValueError('Invalid grasp-plane or closed-width reference')
    pred_z,ref_z=fk.positions(prediction)[:,2],fk.positions(target)[:,2]
    p,t=onset(pred_z,grasp_z),onset(ref_z,grasp_z)
    lower,upper=closed_width-.002,closed_width+.002
    def closed_at(values,index):
        if index is None:return None
        history=np.r_[state[4],values[:index+1,4]]
        return bool(len(history)>=2 and ((history[-2:]>=lower)&(history[-2:]<=upper)).all())
    passed=closed_at(prediction,p)
    if p is None:
        closure_state='no_lift'
    elif passed:
        closure_state='within_band'
    elif prediction[p,4]>upper:
        closure_state='too_open'
    elif prediction[p,4]<lower:
        closure_state='too_closed'
    else:
        closure_state='not_stable'
    opened=prediction[:,4]>upper
    return {'predicted_lift_offset_s':None if p is None else (p+1)/10,
        'reference_lift_offset_s':None if t is None else (t+1)/10,
        'predicted_width_at_lift_mm':None if p is None else float(prediction[p,4]*1000),
        'closed_before_lift':passed, 'closure_state_at_lift':closure_state,
        'reference_closed_before_lift':closed_at(target,t),
        'early_lift_seconds':None if p is None or t is None else (t-p)/10,
        'max_rise_while_open_mm':float(max(0.,np.max(pred_z[opened]-grasp_z) if opened.any() else 0.)*1000),
        'grasp_plane_z_m':grasp_z,'closed_reference_width_mm':closed_width*1000,
        'closed_band_mm':[lower*1000,upper*1000], 'task_success':None}


def coordination_plan(root,data,fk):
    sources=json.loads((Path(root)/'source_manifest.json').read_text())['sources']
    result={}
    for source in sources:
        if source['split']!=data.split:continue
        task=source['task_id']
        events=Path(source['source_episode'])/'task_events.jsonl'
        if digest(events)!=source['source_file_sha256']['task_events.jsonl']:
            raise ValueError('Source task phases changed')
        phases=parse_phases(read_jsonl(events),task)
        rows=[r['source_grid_row'] for r in data.mapping if r['source_task_id']==task]
        closing=[r for r in rows if phase_at(r['action_stamp_ns'],phases)=='close']
        if not closing:raise ValueError('Missing reference closure phase')
        actions=np.asarray([r['action'] for r in closing])
        plane=float(np.median(fk.positions(actions)[:,2]))
        closed=float(np.percentile(actions[:,4],10))
        eligible=[]
        for index in data.groups[task]:
            row=data.mapping[index]['source_grid_row']
            if phase_at(row['stamp_ns'],phases)!='close':continue
            target=np.array([r['source_grid_row']['action'] for r in data.mapping[index:index+50]])
            state=row['state']
            if abs(fk.positions(state)[2]-plane)>.005:continue
            metric=coordination_metrics(target,target,state,fk,plane,closed)
            if metric['reference_closed_before_lift'] is True:
                eligible.append(index)
        selected=sorted(set([eligible[0],eligible[-1]])) if eligible else []
        result[task]={'indices':selected,'eligible_indices':eligible,'eligible_starts':len(eligible),'grasp_z':plane,'closed_width':closed,
                      'events_sha256':digest(events),'status':'eligible' if selected else 'no_full_transition_window'}
    if set(result)!=set(data.groups):raise ValueError('Missing phase source for split')
    if not any(r['indices'] for r in result.values()):raise ValueError('No complete closure-to-lift windows in split')
    return result


def summarize_coordination(samples):
    values=[r['coordination'] for r in samples]
    lift=[v for v in values if v['predicted_lift_offset_s'] is not None]
    return {'evaluated_windows':len(values),'predicted_lift_windows':len(lift),
        'closed_before_lift_windows':sum(v['closed_before_lift'] is True for v in values),
        'lift_without_closure_windows':sum(v['closed_before_lift'] is False for v in lift),
        'lift_with_open_gripper_windows':sum(v.get('closure_state_at_lift')=='too_open' for v in lift),
        'lift_with_overclosed_gripper_windows':sum(v.get('closure_state_at_lift')=='too_closed' for v in lift),
        'lift_with_unstable_gripper_windows':sum(v.get('closure_state_at_lift')=='not_stable' for v in lift),
        'no_lift_predicted_windows':len(values)-len(lift),
        'max_rise_while_open_mm':max((v['max_rise_while_open_mm'] for v in values),default=None),
        'scope':'teacher_forced_offline_windows; not_closed_loop_success'}
