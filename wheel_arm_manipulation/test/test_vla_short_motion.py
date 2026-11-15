import math
import pytest

from wheel_arm_manipulation.vla_short_motion import bounded_target, valid_prefix,gripper_reference_start
from types import SimpleNamespace


def test_preserves_small_model_actions_and_bounds_larger_ones():
    state = [0., -.6, .9, -.3, .06]
    assert bounded_target(state, [.01, -.59, .89, -.29, .061]) == [.01, -.59, .89, -.29, .061]
    goal = bounded_target(state, [.1, -.5, .8, -.4, .07])
    assert goal == pytest.approx([.03, -.57, .87, -.33, .062])


def test_gripper_model_limit_is_applied_before_execution():
    result = bounded_target([0., 0., 0., 0., .079], [0., 0., 0., 0., .082])
    assert result[4] == .08


def test_half_second_rate_bound_preserves_same_rate_and_raw_jump_rejection():
    state=[0.,-.6,.9,-.3,.06]
    target=[.1,-.5,.8,-.4,.07]
    assert bounded_target(state,target,.5)==pytest.approx(target)
    with pytest.raises(ValueError,match='跳变'):
        bounded_target(state,[0.,-.6,.9,-.3,.08],.5)
    for interval in [0.,.2,1.,float('nan')]:
        with pytest.raises(ValueError,match='时间间隔'):
            bounded_target(state,target,interval)


def test_reference_start_keeps_actual_model_state_and_existing_contact_tolerance():
    names=('left_finger_joint','right_finger_joint')
    model=SimpleNamespace(joints=names,goal_tolerance=.001)
    actual={names[0]:.025,names[1]:.025,'elbow_joint':1.6}
    message=SimpleNamespace(joint_names=list(names),reference=None,
        desired=SimpleNamespace(positions=[.02455,.02455],velocities=[0.,0.]),
        header=SimpleNamespace(stamp=SimpleNamespace(sec=1,nanosec=0)))
    start,reference=gripper_reference_start(actual,message,model,1_000_000_000)
    assert actual[names[0]]==.025
    assert start['elbow_joint']==actual['elbow_joint']
    assert start[names[0]]+start[names[1]]==pytest.approx(.0491)
    assert sum(reference['actual_positions'])==pytest.approx(.05)
    message.desired.positions=[.0239,.0239]
    with pytest.raises(ValueError,match='既有目标容限'):
        gripper_reference_start(actual,message,model,1_000_000_000)
    message.desired.positions=[.02455,.02455]
    with pytest.raises(ValueError,match='timestamp'):
        gripper_reference_start(actual,message,model,1_600_000_000)
    message.desired.velocities=[.001,0.]
    with pytest.raises(ValueError,match='参考尚在运动'):
        gripper_reference_start(actual,message,model,1_000_000_000)


@pytest.mark.parametrize('state,prediction', [
    ([0., 0., 0., 0., .06], [0., 0., .2, 0., .06]),
    ([0., 0., 0., 0., .06], [0., 0., 0., 0., .08]),
    ([0., 1.56, 0., 0., .06], [0., 1.58, 0., 0., .06]),
    ([0., 0., 0., 0., .06], [0., 0., 0., 0., math.nan]),
    ([0., 0., 0., 0., .09], [0., 0., 0., 0., .08]),
    ([0., 0., 0., 0., .08], [0., 0., 0., 0., .096]),
    ([0.], [0.]),
])
def test_rejects_invalid_or_large_actions(state, prediction):
    with pytest.raises(ValueError):
        bounded_target(state, prediction)


def test_outside_raw_width_is_projected_without_relaxing_step_or_physical_limits():
    assert bounded_target([0.,0.,0.,0.,.079], [0.,0.,0.,0.,.084813])[4] == .08
    assert bounded_target([0.,0.,0.,0.,.075731], [0.,0.,0.,0.,.084813])[4] == pytest.approx(.077731)
    assert bounded_target([0.,0.,0.,0.,.001], [0.,0.,0.,0.,-.004])[4] == 0.


def test_prefix_stops_at_first_invalid_point_and_never_skips_to_later_points():
    state=[0.,-.6,.9,-.3,.0747]
    predictions=[[0.,-.6,.9,-.3,w] for w in [.0778,.0686,.0691,.0529,.0594]]
    prefix, reason=valid_prefix(state,predictions)
    assert prefix == predictions[:3]
    assert reason['first_rejected_step']==4
    assert reason['executed_prefix_steps']==3
    with pytest.raises(ValueError,match='第 1/'):
        valid_prefix(state,[[0.,-.6,.9,-.3,.03]])
    assert valid_prefix(state,predictions[:3]) == (predictions[:3],None)
