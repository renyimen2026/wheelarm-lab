import numpy as np
import pytest
from scipy.interpolate import CubicHermiteSpline

from wheel_arm_manipulation.planning import load_model
from wheel_arm_manipulation.vla_short_motion import chunk_trajectory


def build(predictions, state=None,step_seconds=.1):
    arm, gripper = load_model(), load_model('gripper')
    state = state or [0., -.6, .9, -.3, .06]
    start = dict(zip(arm.joints, state[:4]))
    start.update({n:state[4]/2 for n in gripper.joints})
    return chunk_trajectory(state, predictions, arm, gripper, start,step_seconds=step_seconds), arm, gripper


def test_original_ten_targets_and_time_order_survive_retiming():
    targets = [[0., -.6-.001*(i+1), .9, -.3, .06+.001*(i+1)] for i in range(10)]
    (trajectory, goal, report), arm, gripper = build(targets)
    assert report['bounded_waypoints'] == targets
    assert report['nominal_horizon_seconds'] == 1.
    assert report['uniform_time_scale'] >= 1
    times=[]
    for point in trajectory.joint_trajectory.points:
        times.append(point.time_from_start.sec+point.time_from_start.nanosec/1e9)
        assert not point.accelerations  # Controller must use cubic, not quintic.
    assert np.diff(times).min() > 0
    for i, target in enumerate(targets):
        point = trajectory.joint_trajectory.points[2*(i+1)]
        assert point.positions == pytest.approx([*target[:4],target[4]/2,target[4]/2])
        assert times[2*(i+1)] == pytest.approx((i+1)*.1*report['uniform_time_scale'])
    assert goal[gripper.joints[0]] == pytest.approx(.035)


@pytest.mark.parametrize('step_seconds',[.1,.5])
def test_controller_interpolation_respects_bounds_speed_and_acceleration(step_seconds):
    targets = [[0.,-.6,.9,-.3,w] for w in [.08,.081,.079,.08,.081,.08,.08,.079,.081,.08]]
    (trajectory, _, report), arm, gripper = build(targets, [0.,-.6,.9,-.3,.079],step_seconds)
    points = trajectory.joint_trajectory.points
    times = [p.time_from_start.sec+p.time_from_start.nanosec/1e9 for p in points]
    spline = CubicHermiteSpline(times,[p.positions for p in points],[p.velocities for p in points])
    sample = np.linspace(0,times[-1],10001)
    bounds = {**arm.bounds,**gripper.bounds}
    velocity = {**arm.velocities,**gripper.velocities}
    acceleration = {**arm.accelerations,**gripper.accelerations}
    for j, name in enumerate(trajectory.joint_trajectory.joint_names):
        assert spline(sample)[:,j].min() >= bounds[name][0]-1e-8
        assert spline(sample)[:,j].max() <= bounds[name][1]+1e-8
        assert abs(spline(sample,1)[:,j]).max() <= velocity[name]*.2+1e-6
        assert abs(spline(sample,2)[:,j]).max() <= acceleration[name]*.2+1e-5
    assert np.max(abs(spline([0,times[-1]],1))) < 1e-8
    assert report['nominal_horizon_seconds']==pytest.approx(10*step_seconds)


def test_half_second_forecasts_keep_their_physical_times_and_width_targets():
    targets=[[0.,-.6-.01*(i+1),.9,-.3,.079-.003*(i+1)] for i in range(10)]
    (trajectory,_,report),_,_=build(targets,[0.,-.6,.9,-.3,.079],.5)
    assert np.array(report['bounded_waypoints'])==pytest.approx(np.array(targets))
    times=[p.time_from_start.sec+p.time_from_start.nanosec/1e9 for p in trajectory.joint_trajectory.points]
    assert times[-1]==pytest.approx(5*report['uniform_time_scale'])
    assert len(times)>=101
    assert np.max(np.diff(times))/report['uniform_time_scale']<=.050000001
    for i,target in enumerate(targets):
        point=trajectory.joint_trajectory.points[10*(i+1)]
        assert point.positions==pytest.approx([*target[:4],target[4]/2,target[4]/2])


def test_continuous_closing_reference_does_not_inject_opening_at_chunk_start():
    arm,gripper=load_model(),load_model('gripper')
    state=[0.,-.6,.9,-.3,.05]
    actual=dict(zip(arm.joints,state[:4]));actual.update({n:.025 for n in gripper.joints})
    command=dict(actual);command.update({n:.02455 for n in gripper.joints})
    predictions=[[0.,-.6,.9,-.3,.0491] for _ in range(10)]
    baseline,_,old=chunk_trajectory(state,predictions,arm,gripper,actual)
    continuous,_,new=chunk_trajectory(state,predictions,arm,gripper,command)
    assert old['bounded_waypoints']==new['bounded_waypoints']  # Policy targets remain identical.
    assert sum(baseline.joint_trajectory.points[0].positions[-2:])==pytest.approx(.05)
    assert all(sum(point.positions[-2:])==pytest.approx(.0491) for point in continuous.joint_trajectory.points)


def test_chunk_does_not_bypass_step_limits():
    targets = [[0.,-.5,.9,-.3,.07] for _ in range(4)]
    (_, _, report), _, _ = build(targets)
    path = np.array([[0.,-.6,.9,-.3,.06]]+report['bounded_waypoints'])
    assert np.max(abs(np.diff(path[:,:4],axis=0))) <= .030000001
    assert np.max(abs(np.diff(path[:,4]))) <= .002000001
    assert report['nominal_horizon_seconds'] == .4


@pytest.mark.parametrize('targets', [[], [[0.,0.,0.,0.,.06]]*2,
    [[0.,-.6,float('nan'),-.3,.06]]*10, [[0.,-.6,1.2,-.3,.06]]*10])
def test_invalid_chunks_rejected(targets):
    with pytest.raises(ValueError):
        build(targets)


def test_range_projection_is_reported_and_bad_waypoint_is_identified():
    targets = [[0.,-.6,.9,-.3,.08] for _ in range(10)]
    targets[8][4]=.084813
    targets[9][4]=.083785
    (_,_,report),_,_=build(targets,[0.,-.6,.9,-.3,.079])
    assert report['raw_gripper_out_of_range_steps']==[9,10]
    assert report['max_raw_gripper_range_violation_mm']==pytest.approx(4.813)
    assert max(row[4] for row in report['bounded_waypoints'])<=.08
    targets[8][4]=.11
    with pytest.raises(ValueError,match='第 9/10 步'):
        build(targets,[0.,-.6,.9,-.3,.079])
