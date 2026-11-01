"""Tabletop upright-cup geometry from registered RGB-D; no simulator model data."""

from dataclasses import dataclass
import math

import cv2
import numpy as np
from scipy.optimize import least_squares


@dataclass
class CupObservation:
    center: np.ndarray
    radius: float
    height: float
    top: float
    plane: np.ndarray
    table_bounds: np.ndarray
    mask: np.ndarray
    residual: float
    coverage: float


def unproject(depth, intrinsic, transform):
    depth = np.asarray(depth)
    k, tf = np.asarray(intrinsic, dtype=float).reshape(3, 3), np.asarray(transform, dtype=float)
    if (depth.ndim != 2 or min(depth.shape) == 0 or tf.shape != (4, 4)
            or not np.isfinite(k).all() or not np.isfinite(tf).all()
            or k[0, 0] <= 0 or k[1, 1] <= 0 or not np.allclose(k[2], [0, 0, 1])
            or not np.allclose(tf[3], [0, 0, 0, 1])
            or not np.allclose(tf[:3, :3].T @ tf[:3, :3], np.eye(3), atol=1e-5)
            or np.linalg.det(tf[:3, :3]) < .999):
        raise ValueError("深度、相机内参或坐标变换不合法")
    v, u = np.indices(depth.shape)
    z = np.where(np.isfinite(depth) & (depth > .1) & (depth < 2.0), depth, np.nan)
    camera = np.stack(((u-k[0, 2])*z/k[0, 0], (v-k[1, 2])*z/k[1, 1], z), axis=-1)
    return camera @ tf[:3, :3].T + tf[:3, 3]


def table_plane(cloud):
    small = cloud[::3, ::3]
    horizontal = np.cross(small[1:-1, 2:] - small[1:-1, :-2],
                          small[2:, 1:-1] - small[:-2, 1:-1])
    norm = np.linalg.norm(horizontal, axis=2)
    points = small[1:-1, 1:-1]
    usable = np.isfinite(points).all(axis=2) & (norm > 1e-9)
    usable &= np.abs(horizontal[:, :, 2]) > .98 * norm
    # A manipulation workspace, not a known table location or height.
    usable &= ((points[:, :, 0] > .15) & (points[:, :, 0] < 1.1)
               & (np.abs(points[:, :, 1]) < .6) & (points[:, :, 2] > .2) & (points[:, :, 2] < 1.1))
    samples = points[usable]
    if len(samples) < 150:
        raise ValueError("可见水平面不足，不能可靠识别桌面；请使用高位相机")
    samples = samples[::max(1, len(samples)//4000)]
    rng = np.random.default_rng(4)
    best = np.zeros(len(samples), dtype=bool)
    for _ in range(100):
        triangle = samples[rng.choice(len(samples), 3, replace=False)]
        normal = np.cross(triangle[1]-triangle[0], triangle[2]-triangle[0])
        length = np.linalg.norm(normal)
        if length < 1e-8 or abs(normal[2]) < .996 * length:
            continue
        normal /= length
        inliers = np.abs((samples-triangle[0]) @ normal) < .003
        if inliers.sum() > best.sum():
            best = inliers
        if best.sum() > .9 * len(samples):
            break
    if best.sum() < 150:
        raise ValueError("未找到可信的近水平桌面")
    fitted = samples[best]
    coefficients = np.linalg.lstsq(np.c_[fitted[:, :2], np.ones(len(fitted))], fitted[:, 2], rcond=None)[0]
    if np.linalg.norm(coefficients[:2]) > .09:
        raise ValueError("桌面倾斜超过当前直立杯抓取能力")
    distance = cloud[:, :, 2] - (cloud[:, :, :2] @ coefficients[:2] + coefficients[2])
    support = cloud[np.isfinite(cloud).all(axis=2) & (np.abs(distance) < .003)
                    & (cloud[:, :, 0] > .15) & (cloud[:, :, 0] < 1.1)
                    & (np.abs(cloud[:, :, 1]) < .6)]
    if len(support) < 500:
        raise ValueError("桌面有效点数不足")
    bounds = np.percentile(support, [.5, 99.5], axis=0)
    if min(bounds[1, :2]-bounds[0, :2]) < .04:
        raise ValueError("桌面可见范围太小")
    return coefficients, bounds, distance


def fit_cylinder(points, plane, require_support=True):
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 60 or not np.isfinite(points).all():
        raise ValueError("杯身有效三维点不足")
    low, top = np.percentile(points[:, 2], [2, 99])
    middle = points[(points[:, 2] > low + .15*(top-low)) & (points[:, 2] < low + .75*(top-low))]
    if len(middle) < 35:
        raise ValueError("可见杯身侧面不足")
    xy = middle[::max(1, len(middle)//2500), :2]
    origin = np.mean(xy, axis=0)
    centered = xy-origin
    design = np.c_[2*centered, np.ones(len(xy))]
    solution, _, rank, singular = np.linalg.lstsq(design, (centered**2).sum(axis=1), rcond=None)
    if rank < 3 or singular[-1] / singular[0] < 1e-4:
        raise ValueError("杯身侧面退化，无法拟合圆柱")
    radius = math.sqrt(max(1e-8, solution[2] + sum(solution[:2]**2)))
    if not .009 < radius < .06:
        raise ValueError("候选物体不是可抓取的小型圆柱杯")
    fit = least_squares(lambda x: np.linalg.norm(xy-x[:2], axis=1)-x[2],
                        [*(origin+solution[:2]), np.clip(radius, .0101, .0399)],
                        bounds=([origin[0]-.1, origin[1]-.1, .01], [origin[0]+.1, origin[1]+.1, .04]),
                        loss="soft_l1", f_scale=.001, max_nfev=80)
    center_xy, radius = fit.x[:2], float(fit.x[2])
    errors = np.abs(np.linalg.norm(xy-center_xy, axis=1)-radius)
    residual = float(np.sqrt(np.mean(errors**2)))
    angles = np.sort(np.arctan2(xy[:, 1]-center_xy[1], xy[:, 0]-center_xy[0]))
    coverage = float(2*np.pi - np.diff(np.r_[angles, angles[0]+2*np.pi]).max())
    floor = float(center_xy @ plane[:2] + plane[2])
    bottom = floor if require_support else max(floor, low)
    height = float(top-bottom)
    if (not fit.success or residual > .0025 or np.mean(errors < .003) < .85
            or coverage < 1.2 or not .02 < 2*radius < .074 or not .04 < height < .22
            or not 1.0 < height/(2*radius) < 4.5):
        raise ValueError("候选物体的圆柱拟合、尺寸或可见弧度不可靠")
    if require_support and (low-floor > .016 or top-low < .65*height):
        raise ValueError("杯底被遮挡或物体不在桌面上，拒绝猜测完整杯子高度")
    return np.array([*center_xy, bottom+height/2]), radius, height, float(top), residual, coverage


def detect_cup(bgr, depth, intrinsic, transform, require_support=True):
    if (bgr.dtype != np.uint8 or bgr.ndim != 3 or bgr.shape[2] != 3
            or depth.shape != bgr.shape[:2]):
        raise ValueError("需要配准、同尺寸的 BGR 彩色图和米单位深度图")
    cloud = unproject(depth, intrinsic, transform)
    plane, bounds, elevation = table_plane(cloud)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    # Neutral ceramic appearance is only an auxiliary filter; geometry decides.
    mask = ((hsv[:, :, 1] < 65) & (hsv[:, :, 2] > 45) & (elevation > .006) & (elevation < .25)
            & np.isfinite(cloud).all(axis=2)
            & (cloud[:, :, 0] >= bounds[0, 0]-.01) & (cloud[:, :, 0] <= bounds[1, 0]+.01)
            & (cloud[:, :, 1] >= bounds[0, 1]-.01) & (cloud[:, :, 1] <= bounds[1, 1]+.01))
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    candidates = []
    for label in range(1, count):
        x, y, w, h, area = stats[label]
        if area < 60 or area > .08*bgr.shape[0]*bgr.shape[1] or min(x, y) <= 1 or x+w >= bgr.shape[1]-1 or y+h >= bgr.shape[0]-1:
            continue
        selected = labels == label
        try:
            center, radius, height, top, residual, coverage = fit_cylinder(cloud[selected], plane, require_support)
        except ValueError:
            continue
        candidates.append(CupObservation(center, radius, height, top, plane, bounds, selected, residual, coverage))
    if len(candidates) != 1:
        raise ValueError(f"可靠的直立无柄杯候选数量为 {len(candidates)}；必须唯一，不会自动猜测目标")
    return candidates[0]


def require_stable(observations, position_tolerance=.004):
    if len(observations) < 3:
        raise ValueError("至少需要三帧独立观测")
    centers = np.array([o.center for o in observations])
    if (np.max(np.linalg.norm(centers-centers.mean(axis=0), axis=1)) > position_tolerance
            or np.ptp([o.radius for o in observations]) > .0015
            or np.ptp([o.height for o in observations]) > .005):
        raise ValueError("多帧杯子位置或尺寸不稳定")
    return observations[-1]
