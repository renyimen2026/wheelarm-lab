"""Prior collision proxies and a fixed dock marker for the controlled house demo."""

from pathlib import Path
import time

from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Pose
from moveit_msgs.msg import CollisionObject, PlanningScene
from moveit_msgs.srv import ApplyPlanningScene
import numpy as np
import rclpy
from rclpy.node import Node
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import Image
from shape_msgs.msg import SolidPrimitive
from visualization_msgs.msg import Marker, MarkerArray

from wheel_arm_manipulation.living_room import layout


def fixture_scene(config, gazebo_share, current_world_base=None):
    settings, _, poses, bounds, _, base = layout(config, gazebo_share)
    if current_world_base is not None:
        base = current_world_base
    inverse = np.linalg.inv(base)
    scene = PlanningScene(is_diff=True)
    scene.robot_state.is_diff = True
    entries = []
    for name, limits in bounds.items():
        center = np.eye(4)
        center[:3, 3] = limits.mean(axis=0)
        identifier = "grasp_fixture_table" if name == settings["table_model"] else "house_fixture_"+name
        entries.append((identifier, inverse @ poses[name] @ center, limits[1]-limits[0]))
    floor = np.eye(4)
    floor[:3, 3] = [0, 0, -.02]
    entries.append(("grasp_fixture_floor", floor, [20., 20., .02]))
    for name, transform, size in entries:
        obj = CollisionObject(id=name, operation=CollisionObject.ADD)
        obj.header.frame_id = "base_footprint"
        obj.pose.position.x, obj.pose.position.y, obj.pose.position.z = map(float, transform[:3, 3])
        q = Rotation.from_matrix(transform[:3, :3]).as_quat()
        obj.pose.orientation.x, obj.pose.orientation.y, obj.pose.orientation.z, obj.pose.orientation.w = map(float, q)
        obj.primitives = [SolidPrimitive(type=SolidPrimitive.BOX, dimensions=[float(v) for v in size])]
        obj.primitive_poses = [Pose()]
        obj.primitive_poses[0].orientation.w = 1.
        scene.world.collision_objects.append(obj)
    return scene


class LivingRoomFixture(Node):
    def __init__(self):
        super().__init__("living_room_fixture")
        self.declare_parameter("layout", str(Path(get_package_share_directory(
            "wheel_arm_manipulation")) / "config/living_room.yaml"))
        config = self.get_parameter("layout").value
        gazebo = get_package_share_directory("wheel_arm_gazebo")
        self.scene = fixture_scene(config, gazebo)
        self.label = layout(config, gazebo)[0]["name"]
        self.client = self.create_client(ApplyPlanningScene, "/apply_planning_scene")
        self.future, self.ready, self.sent_at = None, False, 0.
        self.markers = self.create_publisher(MarkerArray, "/cup_grasp/markers", 1)
        self.keepalive = [self.create_subscription(Image, topic, lambda _: None, 10)
                          for topic in ("/camera/image", "/camera/depth_image")]
        self.create_timer(.5, self.update)

    def update(self):
        marker = Marker(ns="living_room_dock", id=0, type=Marker.TEXT_VIEW_FACING, action=Marker.ADD)
        marker.header.frame_id = "odom"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.pose.position.z = 1.15
        marker.pose.orientation.w = 1.
        marker.scale.z = .12
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = .1, .8, 1., 1.
        marker.text = self.label
        self.markers.publish(MarkerArray(markers=[marker]))
        if self.ready:
            return
        if self.future is not None:
            if self.future.done():
                response = self.future.result()
                self.ready = response is not None and response.success
                self.future = None
                if self.ready:
                    self.get_logger().info("[场景就绪] 已加载客厅局部家具碰撞代理；请保持小车在标记点，不要遥控移动。")
                    return
            elif time.monotonic()-self.sent_at < 5:
                return
            else:
                self.future.cancel()
                self.future = None
        if self.client.service_is_ready():
            self.future = self.client.call_async(ApplyPlanningScene.Request(scene=self.scene))
            self.sent_at = time.monotonic()


def main(args=None):
    rclpy.init(args=args)
    node = LivingRoomFixture()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
