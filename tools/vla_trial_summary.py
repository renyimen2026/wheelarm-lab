"""Bounded-trial accounting and RGB-D lift evidence; no robot/model imports."""
import math


def trial_limits(execute, seconds, maximum, samples):
    if seconds is None:
        if not 1 <= samples <= (3 if execute else 10):
            raise ValueError("短时执行 1～3 轮，只预测 1～10 轮")
        return samples
    if not execute or not math.isfinite(seconds) or not 10 <= seconds <= 600 or not 1 <= maximum <= 100:
        raise ValueError("连续试验须 --execute，时长 10～600 秒，最多 1～100 轮")
    return maximum


def lift_evidence(before, after):
    result = {"method": "stable_rgbd_cup_geometry_before_and_after", "lift_verified": None,
              "task_success": None, "reason": "missing_or_unavailable_measurement"}
    if not before or not after or before.get("status") != "detected" or after.get("status") != "detected":
        return result
    delta = after["top_base_m"]-before["top_base_m"]
    xy = math.dist(before["center_base_m"][:2], after["center_base_m"][:2])
    radius = abs(before["radius_m"]-after["radius_m"])
    if not all(math.isfinite(v) for v in (delta, xy, radius)):
        raise ValueError("杯子几何证据含非有限值")
    verified = xy <= .025 and radius <= .002 and abs(delta-.08) <= .018
    result.update(lift_verified=verified, reason="lift_geometry_matches" if verified else "lift_not_verified",
                  cup_top_delta_m=delta, cup_xy_delta_m=xy, cup_radius_delta_m=radius)
    # Lift geometry alone does not prove maintained gripping or full task success.
    return result


def hold_evidence(before, samples, required_seconds=10.):
    """Assess observations after policy stop; never supply a control command."""
    result = {'task_success':False, 'method':'rgbd_lift_and_stationary_gripper_hold',
              'required_seconds':required_seconds, 'sample_count':len(samples)}
    if len(samples)<3:
        return dict(result,reason='insufficient_hold_samples')
    times=[s['elapsed_seconds'] for s in samples]
    widths=[s['state'][4] for s in samples]
    if (not all(math.isfinite(v) for v in times+widths)
            or any(b<=a for a,b in zip(times,times[1:]))
            or times[-1]-times[0]<required_seconds):
        return dict(result,reason='insufficient_hold_duration')
    evidence=[lift_evidence(before,s['measurement']) for s in samples]
    result.update(duration_seconds=times[-1]-times[0],
                  width_range_m=max(widths)-min(widths),lift_checks=evidence)
    if all('joints_stamp_ns' in s for s in samples):
        sim_span=(samples[-1]['joints_stamp_ns']-samples[0]['joints_stamp_ns'])/1e9
        result['duration_sim_seconds']=sim_span
        if sim_span<required_seconds:
            return dict(result,reason='insufficient_sim_hold_duration')
    if any(e['lift_verified'] is not True or abs(e['cup_top_delta_m']-.08)>.01 for e in evidence):
        return dict(result,reason='target_height_not_maintained')
    # This acceptance range is specific to the fixed 50 mm ceramic cup.
    if min(widths)<.047 or max(widths)>.053 or max(widths)-min(widths)>.002:
        return dict(result,reason='gripper_hold_not_verified')
    return dict(result,task_success=True,reason='lift_and_hold_verified')
