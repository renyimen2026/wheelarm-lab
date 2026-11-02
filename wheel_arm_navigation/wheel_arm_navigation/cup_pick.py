"""Trusted pickup phase, entered only after navigation and positive image confirmation."""

import signal
from pathlib import Path
from uuid import uuid4


def check_ready(observer, required=False):
    import rclpy
    from moveit_msgs.msg import PlanningSceneComponents
    from moveit_msgs.srv import GetPlanningScene

    client = observer.create_client(GetPlanningScene, "/get_planning_scene")
    try:
        if not client.service_is_ready():
            if not required:
                return
            if not client.wait_for_service(timeout_sec=observer.options.nav_wait):
                raise RuntimeError("抓取需要 MoveIt，请启动 living_room_nav.launch.py")
        future = client.call_async(GetPlanningScene.Request(components=PlanningSceneComponents(
            components=PlanningSceneComponents.ROBOT_STATE_ATTACHED_OBJECTS)))
        rclpy.spin_until_future_complete(observer, future, timeout_sec=observer.options.nav_wait)
        if not future.done() or future.result() is None:
            raise RuntimeError("无法确认上一轮夹持状态，禁止开始导航")
        if future.result().scene.robot_state.attached_collision_objects:
            raise RuntimeError("机械臂仍有附着物，请处理夹持物后再启动任务；未发送导航目标")
    finally:
        observer.destroy_client(client)


def execute(options, observer, on_stage):
    from wheel_arm_manipulation.cup_grasp import CupGraspClient, parser, run
    from ament_index_python.packages import get_package_share_directory
    from moveit_msgs.srv import ApplyPlanningScene
    from rclpy.time import Time
    from scipy.spatial.transform import Rotation
    import numpy as np
    from wheel_arm_manipulation.house_navigation import dock_reference
    from wheel_arm_manipulation.living_room_fixture import fixture_scene

    if observer.stopping:
        raise KeyboardInterrupt()
    arguments = parser().parse_args(["--execute", "--finish", "hold", "--rgb-topic", options.rgb_topic,
                                     "--odom-topic", options.odom_topic, "--output-dir",
                                     str(Path(options.observations_dir) / ("grasp-"+uuid4().hex[:12]))])
    node = CupGraspClient(arguments)
    handlers = {}
    try:
        def interrupted(*_):
            node.stopping = observer.stopping = True
            if not node.execution_pending:
                raise KeyboardInterrupt()
        for signum in (signal.SIGINT, signal.SIGTERM):
            handlers[signum] = signal.signal(signum, interrupted)
        if on_stage is not None:
            on_stage("grasping", {})
        transform = observer.buffer.lookup_transform("map", "base_footprint", Time())
        map_base = np.eye(4)
        p, q = transform.transform.translation, transform.transform.rotation
        map_base[:3, 3] = [p.x, p.y, p.z]
        map_base[:3, :3] = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
        map_world = dock_reference(options.house_layout, options.house_reference,
                                   get_package_share_directory("wheel_arm_gazebo"))[1]
        scene = fixture_scene(options.house_layout, get_package_share_directory("wheel_arm_gazebo"),
                              np.linalg.inv(map_world) @ map_base)
        response = node.call(ApplyPlanningScene, "/apply_planning_scene", ApplyPlanningScene.Request(scene=scene))
        if response is None or not response.success:
            raise RuntimeError("家具碰撞场景刷新失败，禁止抓取")
        print("[场景] 已按当前地图定位刷新已配置家具碰撞代理；杯子和桌面再由 RGB-D 实测。", flush=True)
        # The entire navigation/vision/pick task was already confirmed in the terminal.
        code = run(node, arguments, confirm=lambda: not node.stopping)
        if code != 0 or not node.closed or not node.holding_verified:
            raise RuntimeError("未通过抓取抬升验证；请检查机械臂和夹持状态")
        return {"status": "holding", "verification": "RGB-D cup-rim rise", "lift_height": arguments.lift_height}
    finally:
        for signum, handler in handlers.items():
            signal.signal(signum, handler)
        node.destroy_node()
