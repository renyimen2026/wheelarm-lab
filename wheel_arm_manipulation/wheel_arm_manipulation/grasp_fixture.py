"""Known static fixture geometry and cup ground-truth logging; never commands motion."""

import math
from pathlib import Path
import time
import xml.etree.ElementTree as ET

from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Pose, PoseStamped
from moveit_msgs.msg import CollisionObject, PlanningScene
from moveit_msgs.srv import ApplyPlanningScene
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from shape_msgs.msg import SolidPrimitive


def fixture_scene():
    world = ET.parse(Path(get_package_share_directory("wheel_arm_gazebo")) /
                     "worlds/grasp_test.sdf").getroot().find("world")
    table = world.find("model[@name='cup_table']")
    position = [float(v) for v in table.findtext("pose").split()]
    size = [float(v) for v in table.findtext("link/collision/geometry/box/size").split()]
    scene = PlanningScene(is_diff=True)
    scene.robot_state.is_diff = True
    for name, center, dimensions in (("table", position[:3], size),
                                      ("floor", [0.0, 0.0, -0.02], [20.0, 20.0, 0.02])):
        obj = CollisionObject(id="grasp_fixture_" + name, operation=CollisionObject.ADD)
        obj.header.frame_id = "base_footprint"
        obj.pose.position.x, obj.pose.position.y, obj.pose.position.z = center
        obj.pose.orientation.w = 1.0
        obj.primitives = [SolidPrimitive(type=SolidPrimitive.BOX, dimensions=dimensions)]
        pose = Pose()
        pose.orientation.w = 1.0
        obj.primitive_poses = [pose]
        scene.world.collision_objects.append(obj)
    return scene


class GraspFixture(Node):
    def __init__(self):
        super().__init__("grasp_fixture")
        self.client = self.create_client(ApplyPlanningScene, "/apply_planning_scene")
        self.future, self.ready = None, False
        self.sent_at = 0.0
        self.last_report = 0.0
        self.create_timer(0.5, self.install_scene)
        self.create_subscription(PoseStamped, "/grasp_test/cup_pose", self.cup_pose, 10)
        # Keep camera subscriptions active from startup, also without RViz.
        self.camera_keepalive = [self.create_subscription(Image, topic, lambda _: None, 10)
                                 for topic in ("/camera/image", "/camera/depth_image")]

    def install_scene(self):
        if self.ready:
            return
        if self.future is not None:
            if not self.future.done():
                if time.monotonic() - self.sent_at < 5:
                    return
                self.future.cancel()
                self.future = None
            else:
                result = self.future.result()
                self.ready = result is not None and result.success
                self.future = None
            if self.ready:
                self.get_logger().info("[场景就绪] 已加入测试台和地面；请保持底盘不动。杯子接触不由 MoveIt 判定。")
                return
        if self.client.service_is_ready():
            self.future = self.client.call_async(ApplyPlanningScene.Request(scene=fixture_scene()))
            self.sent_at = time.monotonic()

    def cup_pose(self, message):
        if time.monotonic() - self.last_report < 3:
            return
        p = message.pose.position
        if not all(math.isfinite(v) for v in (p.x, p.y, p.z)):
            return
        self.last_report = time.monotonic()
        self.get_logger().info(f"[杯子真值] 世界坐标 x={p.x:.3f}, y={p.y:.3f}, z={p.z:.3f} m；"
                               f"相对桌面初始中心升高 {p.z - 0.5987:.3f} m（不是视觉估计）")


def main(args=None):
    rclpy.init(args=args)
    node = GraspFixture()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
