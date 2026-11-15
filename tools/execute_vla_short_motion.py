"""ROS-only bounded simulation motion, using the existing MoveIt execution/cancel path."""

import argparse
import copy
import json
import math
from pathlib import Path
import signal
import sys
import time
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"wheel_arm_manipulation"))
import message_filters
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, JointState
from control_msgs.msg import JointTrajectoryControllerState
from rclpy.qos import qos_profile_sensor_data
from wheel_arm_manipulation.image_qos import image_qos
from wheel_arm_manipulation.observation_capture import observation, save_observation
import rclpy
from rclpy.parameter import Parameter
from rclpy.signals import SignalHandlerOptions
from moveit_msgs.msg import RobotTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from builtin_interfaces.msg import Duration
from wheel_arm_manipulation.arm_goal import ArmGoalClient
from wheel_arm_manipulation.observation_capture import extract_state
from wheel_arm_manipulation.planning import load_model, values_from_state
from wheel_arm_manipulation.vla_short_motion import bounded_target, chunk_trajectory, valid_prefix,gripper_reference_start


def fresh_gripper_reference(node):
    deadline=time.monotonic()+.5
    while getattr(node,'gripper_reference_message',None) is None and time.monotonic()<deadline:
        rclpy.spin_once(node,timeout_sec=.01)
    message=getattr(node,'gripper_reference_message',None)
    if message is None:raise RuntimeError('夹爪控制器参考尚未收到')
    stamp=message.header.stamp.sec*1_000_000_000+message.header.stamp.nanosec
    catchup=time.monotonic()+.1
    while node.get_clock().now().nanoseconds<stamp and time.monotonic()<catchup:
        rclpy.spin_once(node,timeout_sec=.002)
    return message


def require_single_execution_server(node):
    count = node.count_publishers("/execute_trajectory/_action/status")
    if count != 1:
        raise RuntimeError(f"/execute_trajectory 服务端状态发布端数量为 {count}，需要恰好 1 个；请检查重复 MoveIt/仿真")


def stationary_base(node):
    """Drain a brief clock/odom callback skew; never accept a future/stale sample."""
    started = time.monotonic()
    deadline = started+.1
    timing = getattr(node, "base_feedback_timing", None)
    if timing is None:
        timing = node.base_feedback_timing = {"checks":0, "recovery_attempts":0, "recoveries":0,
            "max_future_ms":0., "max_ros_age_ms":0., "max_wall_age_ms":0., "max_recovery_wait_ms":0.}
    timing["checks"] += 1
    recovering = False
    while True:
        if getattr(node, "stopping", False):
            raise KeyboardInterrupt()
        wall = time.monotonic()
        now = node.get_clock().now().nanoseconds
        odom = node.odom
        if odom is None:
            raise RuntimeError("未收到底盘反馈")
        stamp = odom.header.stamp.sec*10**9+odom.header.stamp.nanosec
        age, wall_age = (now-stamp)/1e9, wall-node.odom_received
        timing["max_future_ms"] = max(timing["max_future_ms"], -age*1000)
        timing["max_ros_age_ms"] = max(timing["max_ros_age_ms"], age*1000)
        timing["max_wall_age_ms"] = max(timing["max_wall_age_ms"], wall_age*1000)
        twist = odom.twist.twist
        components = [twist.linear.x, twist.linear.y, twist.linear.z,
                      twist.angular.x, twist.angular.y, twist.angular.z]
        if not all(math.isfinite(v) for v in components) or math.hypot(*components[:3]) > .01 or math.hypot(*components[3:]) > .02:
            raise RuntimeError("底盘正在移动或速度反馈无效")
        detail = f"ROS年龄={age*1000:.1f}ms，墙钟接收年龄={wall_age*1000:.1f}ms，ROS当前={now}，消息时间={stamp}"
        if min(now, stamp) <= 0 or age < -.1 or wall_age < 0:
            raise RuntimeError("底盘反馈时钟不一致："+detail)
        if 0 <= age <= .5 and wall_age <= .5:
            if recovering:
                timing["recoveries"] += 1
                timing["max_recovery_wait_ms"] = max(timing["max_recovery_wait_ms"], (wall-started)*1000)
            return
        if wall >= deadline:
            raise RuntimeError("底盘反馈未能在100ms内恢复同步："+detail)
        if not recovering:
            recovering = True
            timing["recovery_attempts"] += 1
        # execute() calls this outside a ROS callback, so it is safe to process
        # queued /clock and odom callbacks here, with a strict wall-time budget.
        rclpy.spin_once(node, timeout_sec=min(.01, deadline-wall))


def fresh_stationary_snapshot(node):
    """Startup/previous action already settled; reject motion instead of waiting again."""
    message = node.snapshot(settled=False)
    stationary_base(node)
    if len(message.velocity) != len(message.name):
        raise ValueError("缺少关节速度，无法检查静止状态")
    for name in node.model.joints:
        velocity = message.velocity[message.name.index(name)]
        limit = .005 if name in ("left_finger_joint", "right_finger_joint") else .02
        if not math.isfinite(velocity) or abs(velocity) > limit:
            raise RuntimeError("执行前关节尚未静止，拒绝动作")
    return message


def execute_request(node, arm, gripper, request, result):
    age_wall = time.time()-request["created_wall_seconds"]
    if request.get("scope") != "simulation_short_probe" or not 0 <= age_wall <= 2:
        raise ValueError("执行请求过旧或范围不匹配")
    message = node.pre_execution_snapshot()
    now = node.get_clock().now().nanoseconds
    age = (now-request["observation_stamp_ns"])/1e9
    result["observation_age_before_checks_seconds"] = age
    if not 0 <= age <= 2:
        raise ValueError(f"预测所用观测已超过两秒或来自未来，拒绝执行：{age:.3f}s")
    model = node.model
    actual = extract_state(message)
    start=values_from_state(message,model)
    start_source=request.get('gripper_start_source','actual')
    if start_source not in ('actual','reference'):raise ValueError('未知夹爪起点策略')
    result['gripper_start_source']=start_source
    if start_source=='reference':
        start,reference=gripper_reference_start(start,fresh_gripper_reference(node),gripper,
                                               node.get_clock().now().nanoseconds)
        result['gripper_start_reference']=reference
        node.valid_state(node.state(message))  # Validate measured geometry as well as commanded path.
    observed = request["state_physical"]
    if len(observed) != 5 or any(abs(a-b) > (.003 if i == 4 else .01) for i, (a, b) in enumerate(zip(actual, observed))):
        raise ValueError("推理期间机器人状态发生变化，拒绝执行")
    predictions = request.get("predictions_physical", [request["first_prediction_physical"]])
    step_seconds=request.get('action_step_seconds',.1)
    if step_seconds not in (.1,.5):
        raise ValueError('动作时间间隔无效')
    if request.get("allow_valid_prefix") and len(predictions)>1:
        predictions, truncation = valid_prefix(actual,predictions,step_seconds)
        if truncation is not None:
            result['prefix_truncation'] = truncation
            print(f"[动作截短] 请求 {truncation['requested_action_steps']} 步；仅执行合法前 {len(predictions)} 步；"
                  f"第 {truncation['first_rejected_step']} 步及以后丢弃：{truncation['rejection_reason']}", flush=True)
    if len(predictions) > 1:
        trajectory, goal, details = chunk_trajectory(actual, predictions, arm, gripper, start,step_seconds=step_seconds)
        result.update(details)
        duration, target = details["trajectory_duration_seconds"], details["bounded_waypoints"][-1]
        node.options.timeout = max(10., duration+5)
    else:
        target = bounded_target(actual, predictions[0],step_seconds)
        goal = dict(zip(arm.joints, target[:4]))
        goal.update({name: target[4]/2 for name in gripper.joints})
        model.check(goal)
        duration = max([.5]+[1.5*abs(goal[n]-start[n])/(.2*model.velocities[n]) for n in model.joints]
                       +[math.sqrt(6*abs(goal[n]-start[n])/(.2*model.accelerations[n])) for n in model.joints])
        trajectory = RobotTrajectory()
        trajectory.joint_trajectory.joint_names = list(model.joints)
        # Dense samples of a smooth zero-endpoint-velocity interpolation.
        for i in range(11):
            u = i/10
            positions = [start[n]+(3*u*u-2*u*u*u)*(goal[n]-start[n]) for n in model.joints]
            velocities = [6*u*(1-u)*(goal[n]-start[n])/duration for n in model.joints]
            ns = round(u*duration*1e9)
            trajectory.joint_trajectory.points.append(JointTrajectoryPoint(
                positions=positions, velocities=velocities,
                time_from_start=Duration(sec=ns//10**9, nanosec=ns%10**9)))
        result.update(action_steps=1, action_step_seconds=step_seconds,
                      nominal_horizon_seconds=step_seconds, uniform_time_scale=duration/step_seconds)
    def guard():
        stationary_base(node)
    def before_send():
        if request.get("trial_deadline_monotonic") is not None and time.monotonic() >= request["trial_deadline_monotonic"]:
            raise TimeoutError("trial_deadline_reached")
        require_single_execution_server(node)
        latest=fresh_stationary_snapshot(node)
        if start_source=='reference':
            current_reference,_=gripper_reference_start(values_from_state(latest,model),
                fresh_gripper_reference(node),gripper,node.get_clock().now().nanoseconds)
            if any(abs(current_reference[name]-start[name])>1e-6 for name in gripper.joints):
                raise ValueError('检查期间夹爪参考已改变，拒绝发送旧轨迹')
        age = (node.get_clock().now().nanoseconds-request["observation_stamp_ns"])/1e9
        result["observation_age_at_dispatch_seconds"] = age
        print(f"[时序] 发送前观测年龄 {age:.3f}s（上限 2s）。", flush=True)
        if not 0 <= age <= 2:
            raise ValueError("发送前观测已过期；未发送动作")
        result["execution_attempted"] = True
    node.pre_execution_guard = before_send
    node.motion_guard = guard
    result.update(actual_before=actual, bounded_target=target, raw_prediction=request["first_prediction_physical"],
                  trajectory_duration_seconds=duration, execution_attempted=False)
    print(f"[动作执行] {len(predictions)} 步；目标 {target}；轨迹 {duration:.2f}s；名义 {result['nominal_horizon_seconds']:.1f}s。", flush=True)
    if result.get('raw_gripper_out_of_range_steps'):
        print(f"[夹爪投影] 原始越界步 {result['raw_gripper_out_of_range_steps']}；最大越界 "
              f"{result['max_raw_gripper_range_violation_mm']:.2f}mm；下发范围0～80mm、每步变化≤{20*step_seconds:g}mm。", flush=True)
    node.execute(trajectory, start, goal)
    tip = node.model.tip
    try:
        node.model.tip = "gripper_tcp"
        pose = node.fk(node.state(node.joints)).pose.position
        result["gripper_tcp_base_m"] = [pose.x, pose.y, pose.z]
    finally:
        node.model.tip = tip
    result.update(status="completed", actual_after=extract_state(node.joints))
    return result


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    with temporary.open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
    temporary.rename(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-dir", required=True)
    parser.add_argument("--execute", action="store_true", required=True)
    args = parser.parse_args()
    root = Path(args.session_dir).resolve(strict=True)
    node = None
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    try:
        arm, gripper = load_model(), load_model("gripper")
        model = copy.deepcopy(arm)
        model.joints = arm.joints+gripper.joints
        for field in ("bounds", "velocities", "accelerations"):
            getattr(model, field).update(getattr(gripper, field))
        model.group = ""
        model.start_tolerance, model.reached_tolerance = .002, .002
        options = SimpleNamespace(wait=5., timeout=10., command="vla",
                                  joint_states_topic="/wheel_arm/joint_states", odom_topic="/odometry/filtered")
        node = ArmGoalClient(options, model)
        node.set_parameters([Parameter("use_sim_time", value=True)])
        node.gripper_reference_message=None
        node.gripper_reference_subscription=node.create_subscription(JointTrajectoryControllerState,
            '/wheel_arm/gripper_controller/controller_state',
            lambda message:setattr(node,'gripper_reference_message',message),5)
        signal.signal(signal.SIGINT, lambda *_: setattr(node, "stopping", True))
        signal.signal(signal.SIGTERM, lambda *_: setattr(node, "stopping", True))
        bridge, pair = CvBridge(), [None]
        subscribers = [message_filters.Subscriber(node, Image, "/camera/image", qos_profile=image_qos("reliable")),
                       message_filters.Subscriber(node, JointState, options.joint_states_topic, qos_profile=qos_profile_sensor_data)]
        synchronizer = message_filters.ApproximateTimeSynchronizer(subscribers, queue_size=50, slop=.05)
        synchronizer.registerCallback(lambda rgb, joints: pair.__setitem__(0, (rgb, joints)))
        node.snapshot()  # Startup/discovery/settling happen BEFORE the first observation.
        require_single_execution_server(node)
        node.pre_execution_snapshot = lambda: fresh_stationary_snapshot(node)
        write_json(root/"ready.json", {"status": "ready"})
        print("[ROS会话] 采集与执行节点已就绪，后续复用同一进程。", flush=True)
        sequence = 0
        cup_observer = None
        while rclpy.ok() and not node.stopping:
            rclpy.spin_once(node, timeout_sec=.02)
            path = root/f"request_{sequence:03d}.json"
            if not path.exists():
                continue
            result = {"status": "failed", "execution_attempted": False}
            try:
                request = json.loads(path.read_text())
                if request["operation"] == "capture":
                    after = node.get_clock().now().nanoseconds
                    deadline = time.monotonic()+15
                    pair[0] = None
                    while not node.stopping and time.monotonic() < deadline:
                        rclpy.spin_once(node, timeout_sec=.02)
                        if pair[0] is None:
                            continue
                        rgb, joints = pair[0]
                        pair[0] = None
                        try:
                            meta = observation(rgb, joints, node.get_clock().now().nanoseconds)
                        except ValueError:
                            continue
                        if min(meta["image"]["stamp_ns"], meta["joints_stamp_ns"]) < after:
                            continue
                        meta.update(clock="ros_sim_time", instruction=request["instruction"])
                        destination = Path(request["output_dir"]).resolve()
                        if not destination.is_relative_to(root.parent):
                            raise ValueError("采集输出越出本次会话")
                        saved = save_observation(destination, bridge.imgmsg_to_cv2(rgb, "bgr8"), meta)
                        result.update(status="completed", observation_directory=str(saved))
                        break
                    if result["status"] != "completed":
                        raise RuntimeError("等待新鲜 RGB/关节观测超时或已取消")
                elif request["operation"] == "measure_cup":
                    from wheel_arm_manipulation.cup_perception import CupObserver
                    if cup_observer is None:
                        perception_options = SimpleNamespace(rgb_topic="/camera/image", depth_topic="/camera/depth_image",
                            info_topic="/camera/camera_info", camera_frame="camera_optical_frame", wait=5.,
                            output_dir=str(root.parent/"cup_evidence"))
                        cup_observer = CupObserver(node, perception_options)
                    try:
                        cup = cup_observer.capture(require_support=request["require_support"], wait=5., label=request["label"])
                        measurement = {"status": "detected", "center_base_m": cup.center.tolist(),
                                       "top_base_m": float(cup.top), "radius_m": float(cup.radius)}
                    except RuntimeError as error:
                        measurement = {"status": "unavailable", "reason": str(error)}
                    result.update(status="completed", measurement=measurement)
                elif request["operation"] == "motion":
                    result = execute_request(node, arm, gripper, request, result)
                else:
                    raise ValueError("未知会话操作")
            except TimeoutError as error:
                if str(error) == "trial_deadline_reached":
                    result.update(status="completed", motion_skipped=True, skipped_reason="trial_deadline_reached")
                else:
                    result["error"] = f"{type(error).__name__}: {error}"
            except (Exception, KeyboardInterrupt) as error:
                result["error"] = f"{type(error).__name__}: {error}"
                # Preserve observed tracking error for model/contact diagnosis;
                # these diagnostics never turn a failed action into success.
                if node.joints is not None:
                    try:
                        feedback=extract_state(node.joints)
                        result['actual_at_failure']=feedback
                        result['feedback_ros_stamp_ns_at_failure']=(node.joints.header.stamp.sec*10**9
                                                                    +node.joints.header.stamp.nanosec)
                        if 'bounded_target' in result:
                            result['target_minus_actual_at_failure']=[a-b for a,b in zip(result['bounded_target'],feedback)]
                    except (ValueError,KeyError,IndexError):
                        pass
                print(f"[短时执行停止] {result['error']}", flush=True)
            if hasattr(node, "base_feedback_timing"):
                result["base_feedback_timing"] = dict(node.base_feedback_timing)
            if hasattr(node, "snapshot_clock_timing"):
                result["snapshot_clock_timing"] = dict(node.snapshot_clock_timing)
            write_json(root/f"result_{sequence:03d}.json", result)
            sequence += 1
            if result["status"] != "completed":
                return 1
        return 0
    except (Exception, KeyboardInterrupt) as error:
        if not (root/"ready.json").exists():
            write_json(root/"ready.json", {"status": "failed", "error": str(error)})
        print(f"[ROS会话停止] {type(error).__name__}: {error}", flush=True)
        return 1
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
