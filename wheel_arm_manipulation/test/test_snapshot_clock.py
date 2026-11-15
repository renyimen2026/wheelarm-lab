"""Arrival-order skew must not fabricate movement or accept stale/future feedback."""
import copy
from types import SimpleNamespace as S
from unittest.mock import patch

import pytest
from wheel_arm_manipulation import arm_goal


def run_snapshot(mode='skew'):
    clock={'wall':0.,'ros':1_000_000_000}
    model=S(joints=['joint'],group='arm',frame='base')
    node=S(stopping=False,options=S(wait=.8),model=model,stationary_models=(model,),
           joints=None,odom=None,joint_received=0.,odom_received=0.)
    node.get_clock=lambda:S(now=lambda:S(nanoseconds=clock['ros']))
    node.fresh=lambda *a:arm_goal.ArmGoalClient.fresh(node,*a)
    def message(stamp):
        return S(header=S(stamp=S(sec=stamp//10**9,nanosec=stamp%10**9)))
    def spin(_,timeout_sec):
        clock['wall']+=.003 if timeout_sec==.002 else .01
        if mode!='frozen_clock':clock['ros']=1_000_000_000+round(clock['wall']*1e9)
        if timeout_sec==.002:
            if mode=='motion_during_catchup' and clock['wall']>=.30:node.joints.velocity=[.1]
            return
        stamp=clock['ros']+(200_000_000 if mode=='future' else -1_000_000_000 if mode=='stale' else 2_000_000)
        node.joints=message(stamp);node.joints.name=['joint'];node.joints.position=[0.]
        node.joints.velocity=[float('nan') if mode=='nan_velocity' else .1 if mode=='moving' else 0.]
        node.odom=message(stamp);node.odom.child_frame_id='wrong' if mode=='wrong_frame' else 'base'
        node.odom.twist=S(twist=S(linear=S(x=.1 if mode=='moving_base' else 0.,y=0.,z=0.),
                                 angular=S(x=0.,y=0.,z=0.)))
        node.joint_received=node.odom_received=clock['wall']
    with patch.object(arm_goal.time,'monotonic',side_effect=lambda:clock['wall']), \
         patch.object(arm_goal.rclpy,'ok',return_value=True), \
         patch.object(arm_goal.rclpy,'spin_once',side_effect=spin), \
         patch.object(arm_goal,'values_from_state',return_value={'joint':0.}):
        result=arm_goal.ArmGoalClient.snapshot(node)
    return result,node,clock


def test_skewed_callbacks_preserve_real_quiet_interval_without_returning_future_data():
    result,node,clock=run_snapshot()
    stamp=result.header.stamp.sec*10**9+result.header.stamp.nanosec
    assert clock['wall']>=.3
    assert 0<=clock['ros']-stamp<=500_000_000
    assert node.snapshot_clock_timing['recoveries']>0
    assert node.snapshot_clock_timing['max_future_ms']==2.


@pytest.mark.parametrize('mode',['future','stale','frozen_clock','moving','nan_velocity',
                                  'wrong_frame','moving_base','motion_during_catchup'])
def test_invalid_feedback_or_new_motion_never_qualifies_as_continuously_stopped(mode):
    with pytest.raises(RuntimeError,match='新鲜关节反馈或连续停稳'):
        run_snapshot(mode)
