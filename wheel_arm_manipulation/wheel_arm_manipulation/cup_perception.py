"""Fresh synchronized RGB-D observations and timestamped TF for cup grasping."""

from collections import deque
import copy
from pathlib import Path
import time

from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import PoseStamped
import message_filters
import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

from wheel_arm_manipulation.cup_geometry import detect_cup, require_stable
from wheel_arm_manipulation.camera_check import depth_statistics


def stamp_ns(message):
    return message.header.stamp.sec*10**9 + message.header.stamp.nanosec


def validate_sample(rgb, depth, info, now, after, camera_frame):
    stamps = [stamp_ns(m) for m in (rgb, depth, info)]
    if now <= 0 or min(stamps) <= 0 or min(stamps) < after or any(not 0 <= now-s <= 500000000 for s in stamps):
        raise ValueError("RGB-D 早于采集请求、过期或与 /clock 不一致")
    if max(stamps)-min(stamps) > 25000000:
        raise ValueError("RGB、深度、内参不同步")
    if any(m.header.frame_id != camera_frame for m in (rgb, depth, info)):
        raise ValueError("RGB-D/内参必须位于同一个预期光学坐标系，不能仅改 frame_id 来配准")
    if (rgb.width != depth.width or rgb.height != depth.height or info.width != rgb.width
            or info.height != rgb.height or not np.isfinite(info.k).all() or info.k[0] <= 0 or info.k[4] <= 0):
        raise ValueError("RGB-D 未配准或相机内参、图像尺寸不匹配")
    if (not np.isfinite(info.d).all() or any(abs(v) > 1e-9 for v in info.d)
            or info.binning_x > 1 or info.binning_y > 1 or info.roi.x_offset or info.roi.y_offset):
        raise ValueError("当前入口要求无畸变、无裁剪缩放的配准 RGB-D；请先完成图像校正")


def transform_matrix(message):
    t, q = message.transform.translation, message.transform.rotation
    values = np.array([t.x, t.y, t.z, q.x, q.y, q.z, q.w])
    if not np.isfinite(values).all() or abs(np.linalg.norm(values[3:])-1) > .01:
        raise ValueError("相机 TF 不是合法的刚体变换")
    result = np.eye(4)
    result[:3, 3] = values[:3]
    result[:3, :3] = Rotation.from_quat(values[3:]).as_matrix()
    return result


class CupObserver:
    def __init__(self, node, options):
        self.node, self.options = node, options
        self.bridge = CvBridge()
        self.buffer = Buffer()
        self.listener = TransformListener(self.buffer, node)
        self.sample = None
        self.subscriptions = [message_filters.Subscriber(node, kind, topic, qos_profile=qos_profile_sensor_data)
                              for kind, topic in ((Image, options.rgb_topic), (Image, options.depth_topic),
                                                  (CameraInfo, options.info_topic))]
        self.sync = message_filters.ApproximateTimeSynchronizer(self.subscriptions, queue_size=5, slop=.025)
        self.sync.registerCallback(self.received)
        self.markers = node.create_publisher(MarkerArray, "/cup_grasp/markers", 1)
        self.image = node.create_publisher(Image, "/cup_grasp/image", 1)
        self.pose = node.create_publisher(PoseStamped, "/cup_grasp/pose", 1)

    def received(self, rgb, depth, info):
        self.sample = (rgb, depth, info, time.monotonic())

    def clear(self):
        self.markers.publish(MarkerArray(markers=[Marker(action=Marker.DELETEALL)]))

    def capture(self, require_support=True, wait=None):
        after, started = self.node.get_clock().now().nanoseconds, time.monotonic()
        deadline = started + (self.options.wait if wait is None else wait)
        observations, last_stamp = deque(maxlen=3), -1
        reason = "等待同步 RGB-D、相机内参和拍摄时刻 TF"
        saved = False
        while rclpy.ok() and time.monotonic() < deadline:
            if self.node.stopping:
                raise KeyboardInterrupt()
            rclpy.spin_once(self.node, timeout_sec=.02)
            if self.sample is None:
                continue
            rgb, depth, info, receipt = self.sample
            if receipt <= started or stamp_ns(rgb) <= last_stamp:
                continue
            last_stamp = stamp_ns(rgb)
            try:
                validate_sample(rgb, depth, info, self.node.get_clock().now().nanoseconds, after, self.options.camera_frame)
                transform = self.buffer.lookup_transform(self.node.model.frame, self.options.camera_frame,
                                                         Time.from_msg(rgb.header.stamp))
                bgr = self.bridge.imgmsg_to_cv2(rgb, "bgr8")
                array = self.bridge.imgmsg_to_cv2(depth, "passthrough")
                if depth.encoding == "16UC1":
                    array = array.astype(np.float32)/1000
                elif depth.encoding != "32FC1":
                    raise ValueError("深度只支持 32FC1 米或 16UC1 毫米")
                matrix = transform_matrix(transform)
                if self.options.output_dir and not saved:
                    import cv2
                    path = Path(self.options.output_dir).expanduser() / f"capture-{time.time_ns()}"
                    path.mkdir(parents=True, exist_ok=False)
                    if not cv2.imwrite(str(path / "rgb.png"), bgr):
                        raise OSError("相机证据图片写入失败")
                    np.savez_compressed(path / "geometry.npz", depth=array, k=np.array(info.k), transform=matrix,
                                        stamp_ns=stamp_ns(rgb), frame=self.node.model.frame)
                    saved = True
                if not np.any(np.isfinite(array) & (array > 0)):
                    stats = depth_statistics(array)
                    raise ValueError(f"源深度无有效值：+Inf={stats['posinf']:.0%}，"
                                     f"-Inf={stats['neginf']:.0%}，NaN={stats['nan']:.0%}；"
                                     "请运行 camera_check 区分渲染/裁剪问题，不会使用黑图继续抓取")
                observed = detect_cup(bgr, array, info.k, matrix, require_support)
                if self.node.get_clock().now().nanoseconds-stamp_ns(rgb) > 500000000:
                    raise ValueError("图像处理后观测已过期")
                observations.append(observed)
                stable = require_stable(observations)
            except (ValueError, CvBridgeError, TransformException) as error:
                reason = str(error)
                # An invalid intervening frame cannot be hidden by older good detections.
                if len(observations) >= 3 or "至少需要" not in reason:
                    observations.clear()
                continue
            self.publish(stable, rgb, bgr)
            return stable
        self.clear()
        raise RuntimeError("视觉定位超时：" + reason)

    def publish(self, cup, rgb, bgr):
        import cv2

        pose = PoseStamped()
        pose.header = copy.deepcopy(rgb.header)
        pose.header.frame_id = self.node.model.frame
        pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = map(float, cup.center)
        pose.pose.orientation.w = 1.0
        self.pose.publish(pose)
        marker = Marker(header=pose.header, ns="cup_geometry", id=0, type=Marker.CYLINDER, action=Marker.ADD)
        marker.pose = pose.pose
        marker.scale.x = marker.scale.y = 2*cup.radius
        marker.scale.z = cup.height
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = .1, .9, .3, .55
        marker.lifetime = Duration(seconds=30).to_msg()
        self.markers.publish(MarkerArray(markers=[marker]))
        output = bgr.copy()
        contours, _ = cv2.findContours(cup.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(output, contours, -1, (0, 230, 80), 2)
        cv2.putText(output, f"cup D={2*cup.radius*1000:.1f} H={cup.height*1000:.1f} mm",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 180, 0), 2)
        image = self.bridge.cv2_to_imgmsg(output, "bgr8")
        image.header = copy.deepcopy(rgb.header)
        self.image.publish(image)
