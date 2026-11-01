"""Own only the measured cup/table collision objects and their contact allowance."""

import copy
from uuid import uuid4

from geometry_msgs.msg import Pose
from moveit_msgs.msg import (AllowedCollisionEntry, AttachedCollisionObject, CollisionObject,
                             PlanningScene, PlanningSceneComponents)
from moveit_msgs.srv import ApplyPlanningScene, GetPlanningScene
import numpy as np
from scipy.spatial.transform import Rotation
from shape_msgs.msg import SolidPrimitive


FINGERS = ("left_finger_link", "right_finger_link")


def cup_object(cup, name, frame):
    obj = CollisionObject(id=name, operation=CollisionObject.ADD)
    obj.header.frame_id = frame
    obj.pose.position.x, obj.pose.position.y, obj.pose.position.z = map(float, cup.center)
    obj.pose.orientation.w = 1.0
    obj.primitives = [SolidPrimitive(type=SolidPrimitive.CYLINDER,
                                    dimensions=[cup.height+.006, cup.radius+.003])]
    obj.primitive_poses = [Pose()]
    obj.primitive_poses[0].orientation.w = 1.0
    return obj


def contact_matrix(matrix, name, remove=False, support=None, fingers=False):
    matrix = copy.deepcopy(matrix)
    if remove:
        if name in matrix.entry_names:
            index = matrix.entry_names.index(name)
            del matrix.entry_names[index]
            del matrix.entry_values[index]
            for row in matrix.entry_values:
                del row.enabled[index]
        return matrix
    if not all(finger in matrix.entry_names for finger in FINGERS):
        raise RuntimeError("MoveIt 碰撞矩阵缺少夹爪，拒绝放宽未知碰撞项")
    if name in matrix.entry_names:
        raise RuntimeError("视觉碰撞对象 ID 已存在")
    if support is not None and support not in matrix.entry_names:
        for row in matrix.entry_values:
            row.enabled.append(False)
        matrix.entry_names.append(support)
        matrix.entry_values.append(AllowedCollisionEntry(enabled=[False]*len(matrix.entry_names)))
    for row in matrix.entry_values:
        row.enabled.append(False)
    matrix.entry_names.append(name)
    matrix.entry_values.append(AllowedCollisionEntry(enabled=[False]*len(matrix.entry_names)))
    index = len(matrix.entry_names)-1
    for finger in (*(FINGERS if fingers else ()), *((support,) if support is not None else ())):
        j = matrix.entry_names.index(finger)
        matrix.entry_values[index].enabled[j] = matrix.entry_values[j].enabled[index] = True
    return matrix


class GraspScene:
    def __init__(self, node, observation):
        self.node, self.cup = node, observation
        self.identifier = "rgbd_cup_" + uuid4().hex[:10]
        self.table_id = self.identifier + "_table"
        self.installed = False
        self.possibly_installed = False

    def get(self, ignore_stop=False):
        components = (PlanningSceneComponents.ALLOWED_COLLISION_MATRIX
                      | PlanningSceneComponents.ROBOT_STATE_ATTACHED_OBJECTS
                      | PlanningSceneComponents.WORLD_OBJECT_NAMES)
        return self.node.call(GetPlanningScene, "/get_planning_scene", GetPlanningScene.Request(
            components=PlanningSceneComponents(components=components)), ignore_stop=ignore_stop).scene

    def apply(self, scene, ignore_stop=False):
        response = self.node.call(ApplyPlanningScene, "/apply_planning_scene", ApplyPlanningScene.Request(scene=scene),
                                  ignore_stop=ignore_stop)
        if response is None or not response.success:
            raise RuntimeError("MoveIt 未确认视觉碰撞场景更新")

    def install(self):
        current = self.get()
        if any(a.object.id.startswith("rgbd_cup_") for a in current.robot_state.attached_collision_objects):
            raise RuntimeError("上一轮仍有规划附着杯子，请检查实际夹持状态后重新启动测试场景")
        scene = PlanningScene(is_diff=True)
        scene.robot_state.is_diff = True
        scene.allowed_collision_matrix = contact_matrix(current.allowed_collision_matrix, self.identifier,
                                                         support=self.table_id)
        bounds = self.cup.table_bounds
        table = CollisionObject(id=self.table_id, operation=CollisionObject.ADD)
        table.header.frame_id = self.node.model.frame
        table.pose.position.x, table.pose.position.y = map(float, (bounds[0, :2]+bounds[1, :2])/2)
        table.pose.position.z = float(bounds[1, 2]-.015)
        table.pose.orientation.w = 1.0
        table.primitives = [SolidPrimitive(type=SolidPrimitive.BOX,
                            dimensions=[float(v) for v in (*((bounds[1, :2]-bounds[0, :2])+.04), .03)])]
        table.primitive_poses = [Pose()]
        table.primitive_poses[0].orientation.w = 1.0
        scene.world.collision_objects = [cup_object(self.cup, self.identifier, self.node.model.frame), table]
        # The legacy fixture was inserted in base_footprint at spawn. Refresh it
        # from the observed table after driving; never use its stale position.
        if any(obj.id == "grasp_fixture_table" for obj in current.world.collision_objects):
            fixture = copy.deepcopy(table)
            fixture.id = "grasp_fixture_table"
            top = float(bounds[1, 2])
            fixture.pose.position.z = top/2
            fixture.primitives[0].dimensions[2] = top
            scene.world.collision_objects.append(fixture)
            names = scene.allowed_collision_matrix.entry_names
            if fixture.id not in names:
                for row in scene.allowed_collision_matrix.entry_values:
                    row.enabled.append(False)
                names.append(fixture.id)
                scene.allowed_collision_matrix.entry_values.append(
                    AllowedCollisionEntry(enabled=[False]*len(names)))
            a, b = names.index(self.identifier), names.index(fixture.id)
            scene.allowed_collision_matrix.entry_values[a].enabled[b] = True
            scene.allowed_collision_matrix.entry_values[b].enabled[a] = True
            print("[场景] 已用本次深度测量刷新测试桌碰撞体；不再使用出生点的桌面相对位置。", flush=True)
        self.possibly_installed = True
        self.apply(scene)
        self.installed = True

    def finger_contacts(self, enabled):
        current = self.get()
        matrix = current.allowed_collision_matrix
        index = matrix.entry_names.index(self.identifier)
        for finger in FINGERS:
            other = matrix.entry_names.index(finger)
            matrix.entry_values[index].enabled[other] = enabled
            matrix.entry_values[other].enabled[index] = enabled
        scene = PlanningScene(is_diff=True)
        scene.robot_state.is_diff = True
        scene.allowed_collision_matrix = matrix
        self.apply(scene)

    def attach(self, tcp_transform):
        attached = AttachedCollisionObject(link_name="gripper_tcp", touch_links=list(FINGERS))
        obj = cup_object(self.cup, self.identifier, "gripper_tcp")
        local = np.linalg.inv(tcp_transform)
        position = local[:3, :3] @ self.cup.center + local[:3, 3]
        obj.pose.position.x, obj.pose.position.y, obj.pose.position.z = map(float, position)
        q = Rotation.from_matrix(local[:3, :3]).as_quat()
        obj.pose.orientation.x, obj.pose.orientation.y, obj.pose.orientation.z, obj.pose.orientation.w = map(float, q)
        attached.object = obj
        scene = PlanningScene(is_diff=True)
        scene.robot_state.is_diff = True
        scene.robot_state.attached_collision_objects = [attached]
        # MoveIt transfers the same ID out of the world when attaching it.
        self.apply(scene)
        self.node.attachment = attached

    def detach(self, ignore_stop=False):
        scene = PlanningScene(is_diff=True)
        scene.robot_state.is_diff = True
        scene.robot_state.attached_collision_objects = [AttachedCollisionObject(link_name="gripper_tcp",
            object=CollisionObject(id=self.identifier, operation=CollisionObject.REMOVE))]
        scene.world.collision_objects = [cup_object(self.cup, self.identifier, self.node.model.frame)]
        self.apply(scene, ignore_stop=ignore_stop)
        self.node.attachment = None

    def cleanup(self):
        if not self.possibly_installed:
            return
        current = self.get(ignore_stop=True)
        if any(a.object.id == self.identifier for a in current.robot_state.attached_collision_objects):
            self.detach(ignore_stop=True)
            current = self.get(ignore_stop=True)
        scene = PlanningScene(is_diff=True)
        scene.robot_state.is_diff = True
        world_names = {obj.id for obj in current.world.collision_objects}
        scene.world.collision_objects = [CollisionObject(id=name, operation=CollisionObject.REMOVE)
                                          for name in (self.identifier, self.table_id) if name in world_names]
        scene.allowed_collision_matrix = contact_matrix(current.allowed_collision_matrix,
                                                         self.identifier, remove=True)
        scene.allowed_collision_matrix = contact_matrix(scene.allowed_collision_matrix, self.table_id, remove=True)
        self.apply(scene, ignore_stop=True)
        self.node.attachment = None
