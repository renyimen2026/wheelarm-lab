"""Pure bounded-target calculation for the first simulated VLA motion probe."""

import math


def gripper_reference_start(actual_start, message, gripper, now_ns):
    """Keep the actuator command continuous while retaining measured arm state.

    Contact can leave measured fingers outside a valid closing reference. Do
    not inject a new opening command at each chunk boundary. This changes no
    policy target, controller tolerance, or measured model input.
    """
    from .controller_reference import controller_reference
    row=controller_reference(message,'gripper',now_ns,'ros_sim_time')
    point=getattr(message,row['source_field'])
    velocities=list(point.velocities)
    if velocities and (len(velocities)!=2 or not all(math.isfinite(v) and abs(v)<=1e-4 for v in velocities)):
        raise ValueError('夹爪位置参考尚在运动，不能作为动作块衔接起点')
    reference=dict(zip(row['names'],row['positions']))
    errors={name:reference[name]-actual_start[name] for name in gripper.joints}
    if any(abs(value)>gripper.goal_tolerance+1e-9 for value in errors.values()):
        raise ValueError('夹爪参考与实测差超过既有目标容限，拒绝衔接')
    result=dict(actual_start)
    result.update(reference)
    row.update(actual_positions=[actual_start[name] for name in row['names']],
               reference_minus_actual=errors,goal_tolerance_per_finger_m=gripper.goal_tolerance)
    return result,row


def bounded_target(state, prediction, step_seconds=.1):
    if step_seconds not in (.1,.5):
        raise ValueError('动作时间间隔仅支持0.1或0.5秒')
    if len(state) != 5 or len(prediction) != 5 or not all(math.isfinite(x) for x in [*state, *prediction]):
        raise ValueError("动作与状态需要有限五维数值")
    if any(abs(a-b) > .15 for a, b in zip(state[:4], prediction[:4])) or abs(state[4]-prediction[4]) > .015:
        raise ValueError("预测相对当前状态跳变过大（机械臂上限0.15rad、夹爪上限15mm），停止执行")
    bounds = [(-3.14, 3.14), (-1.57, 1.57), (-2.2, 2.2), (-2.6, 2.6), (0., .08)]
    result = []
    for i, (current, proposed, (low, high)) in enumerate(zip(state, prediction, bounds)):
        if not low-1e-6 <= current <= high+1e-6:
            raise ValueError("当前状态超出机器人限位")
        if i < 4 and not low <= proposed <= high:
            raise ValueError("首步机械臂预测越过关节限位")
        if i == 4:
            # Keep the raw-jump rejection above, then project onto physical
            # width limits before applying the per-step rate bound.
            proposed = min(high, max(low, proposed))
        limit = (.002 if i == 4 else .03)*(step_seconds/.1)
        result.append(min(high, max(low, current+min(limit, max(-limit, proposed-current)))))
    return result



def valid_prefix(state, predictions, step_seconds=.1):
    """Return the longest prefix before the first invalid waypoint; never skip it."""
    if not predictions:
        raise ValueError("动作块为空")
    previous=list(state)
    for i,row in enumerate(predictions):
        try:
            previous=bounded_target(previous,row,step_seconds)
        except ValueError as error:
            if i == 0:
                raise ValueError(f"动作块第 1/{len(predictions)} 步：{error}") from error
            return predictions[:i], {"requested_action_steps":len(predictions),
                "executed_prefix_steps":i,"first_rejected_step":i+1,"rejection_reason":str(error)}
    return predictions, None


def chunk_trajectory(state, predictions, arm, gripper, start, scaling=.2,step_seconds=.1):
    """Preserve physical waypoint times, then uniformly retime a cubic.

    Analytic polynomial extrema bound the entire spline, not only message points.
    Each requested waypoint keeps the same per-step bounds as the first-step probe.
    """
    import numpy as np
    from scipy.interpolate import PchipInterpolator, CubicHermiteSpline
    from moveit_msgs.msg import RobotTrajectory
    from trajectory_msgs.msg import JointTrajectoryPoint
    from builtin_interfaces.msg import Duration
    from wheel_arm_manipulation.planning import validate_trajectory

    if not 2 <= len(predictions) <= 10 or not 0 < scaling <= 1:
        raise ValueError("动作块支持 2～10 步")
    targets, previous = [], list(state)
    for index, row in enumerate(predictions):
        try:
            previous = bounded_target(previous, row,step_seconds)
        except ValueError as error:
            raise ValueError(f"动作块第 {index+1}/{len(predictions)} 步：{error}") from error
        targets.append(previous)
    names = arm.joints+gripper.joints
    values = [[start[n] for n in names]]
    for target in targets:
        values.append([*target[:4], target[4]/2, target[4]/2])
    times = np.arange(len(values))*step_seconds
    slopes = PchipInterpolator(times, values).derivative()(times)
    slopes[0], slopes[-1] = 0., 0.
    spline = CubicHermiteSpline(times, values, slopes)
    checks = set(times.tolist())
    # Position extrema: roots of velocity. Velocity extrema: roots of acceleration.
    for i in range(len(times)-1):
        for j in range(len(names)):
            c = spline.c[:, i, j]
            for coefficients in ([3*c[0], 2*c[1], c[2]], [6*c[0], 2*c[1]]):
                coefficients = np.trim_zeros(coefficients, 'f')
                for root in np.roots(coefficients) if len(coefficients) > 1 else []:
                    if abs(root.imag) < 1e-10 and 0 < root.real < step_seconds:
                        checks.add(float(times[i]+root.real))
    checks = sorted(checks)
    positions = spline(checks)
    for j, name in enumerate(names):
        low, high = (arm.bounds if name in arm.joints else gripper.bounds)[name]
        if positions[:, j].min() < low-1e-9 or positions[:, j].max() > high+1e-9:
            raise ValueError(f"动作块插值越过关节限位：{name}")
    vmax = np.array([(arm.velocities if n in arm.joints else gripper.velocities)[n]*scaling for n in names])
    amax = np.array([(arm.accelerations if n in arm.joints else gripper.accelerations)[n]*scaling for n in names])
    # Acceleration can jump at knots: include both polynomial side limits.
    accelerations = np.concatenate((2*spline.c[1], 6*spline.c[0]*step_seconds+2*spline.c[1]), axis=0)
    scale = max(1., float(np.max(abs(spline(checks, 1))/vmax)),
                float(np.sqrt(np.max(abs(accelerations)/amax))))*1.05
    duration = float(times[-1]*scale)
    if not math.isfinite(duration) or duration > 15:
        raise ValueError(f"动作块需要放慢至 {duration:.2f}s，超过本轮 15s 上限")
    trajectory = RobotTrajectory()
    trajectory.joint_trajectory.joint_names = list(names)
    # Preserve the original collision-check density even with 0.5s target
    # spacing. Exact positions/derivatives preserve the same cubic controller
    # path between samples. Every original knot is included on this grid.
    subdivisions=max(2*len(predictions),round(times[-1]/.05))
    for t in np.linspace(0, times[-1], subdivisions+1):
        ns = round(float(t*scale)*1e9)
        trajectory.joint_trajectory.points.append(JointTrajectoryPoint(
            positions=spline(t).tolist(), velocities=(spline(t, 1)/scale).tolist(),
            accelerations=(spline(t, 2)/scale**2).tolist(),
            time_from_start=Duration(sec=ns//10**9, nanosec=ns%10**9)))
    goal = dict(zip(names, values[-1]))
    # Validate against a combined model with the same limits.
    import copy
    combined = copy.deepcopy(arm)
    combined.joints = names
    for field in ('bounds', 'velocities', 'accelerations'):
        getattr(combined, field).update(getattr(gripper, field))
    validate_trajectory(trajectory, combined, start, goal, scaling, scaling)
    # Positions+velocities select cubic interpolation in the controller. Supplying
    # accelerations would select quintic interpolation across acceleration jumps.
    for point in trajectory.joint_trajectory.points:
        point.accelerations = []
    return trajectory, goal, {"action_steps":len(predictions), "bounded_waypoints":targets,
        "nominal_horizon_seconds":float(times[-1]),"action_step_seconds":step_seconds,"uniform_time_scale":scale,
        "collision_sampling_nominal_step_seconds":float(times[-1]/subdivisions),
        "trajectory_duration_seconds":duration,
        "raw_gripper_out_of_range_steps":[i+1 for i,row in enumerate(predictions) if not 0 <= row[4] <= .08],
        "max_raw_gripper_range_violation_mm":max(max(0.,-row[4],row[4]-.08)*1000 for row in predictions)}
