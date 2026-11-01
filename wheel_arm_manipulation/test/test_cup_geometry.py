import copy
from types import SimpleNamespace as NS

import numpy as np
import pytest
from sensor_msgs.msg import CameraInfo, Image

from wheel_arm_manipulation.cup_geometry import detect_cup, fit_cylinder, require_stable, unproject
from wheel_arm_manipulation.cup_perception import validate_sample


def render(cups=(), flat=False):
    height, width = 360, 480
    k = np.array([[400., 0, width/2], [0, 400., height/2], [0, 0, 1]])
    tf = np.eye(4)
    tf[:3, 3] = [0, .15, .85]
    z = np.array([.65, -.15, -.27])
    z /= np.linalg.norm(z)
    x = np.cross(z, [0, 0, 1])
    x /= np.linalg.norm(x)
    tf[:3, :3] = np.array([x, np.cross(z, x), z]).T
    v, u = np.indices((height, width))
    ray = np.stack(((u-k[0, 2])/400, (v-k[1, 2])/400, np.ones_like(u)), axis=-1) @ tf[:3, :3].T
    origin = tf[:3, 3]
    depth = (.55-origin[2])/ray[:, :, 2]
    points = origin+depth[:, :, None]*ray
    valid = (depth > 0) & (points[:, :, 0] > .45) & (points[:, :, 0] < .9) & (np.abs(points[:, :, 1]) < .25)
    depth = np.where(valid, depth, np.inf)
    rgb = np.full((height, width, 3), 100, dtype=np.uint8)
    for cx, cy, radius, cup_height in cups:
        delta = origin[:2]-[cx, cy]
        a = np.sum(ray[:, :, :2]**2, axis=2)
        b = 2*(ray[:, :, :2] @ delta)
        c = delta @ delta-radius*radius
        disc = b*b-4*a*c
        distance = (-b-np.sqrt(np.maximum(disc, 0)))/(2*a)
        z = origin[2]+distance*ray[:, :, 2]
        body = (disc > 0) & (distance > 0) & (z > .55) & (z < .55+cup_height) & (distance < depth)
        rgb[body] = 210
        depth[body] = distance[body]
    if flat:
        distance = (.65-origin[0])/ray[:, :, 0]
        points = origin+distance[:, :, None]*ray
        body = ((points[:, :, 2] > .55) & (points[:, :, 2] < .66)
                & (np.abs(points[:, :, 1]) < .03) & (distance > 0) & (distance < depth))
        rgb[body], depth[body] = 210, distance[body]
    return rgb, depth.astype(np.float32), k, tf


@pytest.mark.parametrize("center,radius,height", [((.62, -.07), .018, .075), ((.68, .06), .025, .10),
                                                ((.76, -.03), .032, .13)])
def test_recovers_changed_position_and_size_from_rgbd(center, radius, height):
    sample = render([(*center, radius, height)])
    cup = detect_cup(*sample)
    assert np.linalg.norm(cup.center-[*center, .55+height/2]) < .003
    assert abs(cup.radius-radius) < .001
    assert abs(cup.height-height) < .004
    assert cup.residual < .001 and cup.coverage > 1.2


@pytest.mark.parametrize("mode", ["empty", "ambiguous", "flat", "invalid_depth", "color_only"])
def test_rejects_unreliable_candidates(mode):
    sample = list(render([(.65, 0, .025, .1)] if mode == "invalid_depth" else
                         [(.61, -.08, .025, .1), (.73, .08, .025, .1)] if mode == "ambiguous" else [],
                         flat=mode == "flat"))
    if mode == "invalid_depth":
        sample[1][:] = np.nan
    elif mode == "color_only":
        sample[0][120:160, 220:250] = 230
    with pytest.raises(ValueError):
        detect_cup(*sample)


def test_partial_arc_fit_and_moving_observations_are_rejected():
    cup = detect_cup(*render([(.65, 0, .025, .1)]))
    assert require_stable([cup, copy.deepcopy(cup), copy.deepcopy(cup)]) is not None
    moved = copy.deepcopy(cup)
    moved.center[0] += .02
    with pytest.raises(ValueError, match="不稳定"):
        require_stable([cup, cup, moved])
    angle, height = np.meshgrid(np.linspace(0, .3, 30), np.linspace(.56, .65, 30))
    points = np.c_[.65+.025*np.cos(angle.ravel()), .025*np.sin(angle.ravel()), height.ravel()]
    with pytest.raises(ValueError):
        fit_cylinder(points, np.array([0, 0, .55]))


@pytest.mark.parametrize("mode", ["negative_focal", "bad_transform", "reflection"])
def test_bad_projection_is_rejected(mode):
    _, depth, k, tf = render()
    if mode == "negative_focal":
        k[0, 0] = -1
    elif mode == "bad_transform":
        tf[0, 0] = 10
    else:
        tf[:3, 0] *= -1
    with pytest.raises(ValueError):
        unproject(depth, k, tf)


@pytest.mark.parametrize("mode", ["valid", "stale", "before_request", "future", "frame", "sync", "dimensions", "intrinsics", "distortion"])
def test_sample_headers_calibration_and_time(mode):
    rgb, depth = Image(width=640, height=480), Image(width=640, height=480)
    info = CameraInfo(width=640, height=480, k=[500., 0., 320., 0., 500., 240., 0., 0., 1.])
    for message in (rgb, depth, info):
        message.header.stamp.sec = 10
        message.header.frame_id = "camera_optical_frame"
    now, after = 10_100_000_000, 9_900_000_000
    if mode == "stale":
        now += 1_000_000_000
    elif mode == "before_request":
        after = 10_050_000_000
    elif mode == "future":
        now = 9_990_000_000
    elif mode == "frame":
        depth.header.frame_id = "camera_link"
    elif mode == "sync":
        depth.header.stamp.nanosec = 40_000_000
    elif mode == "dimensions":
        info.width = 320
    elif mode == "intrinsics":
        info.k[0] = 0.
    elif mode == "distortion":
        info.d = [.1]
    if mode == "valid":
        validate_sample(rgb, depth, info, now, after, "camera_optical_frame")
    else:
        with pytest.raises(ValueError):
            validate_sample(rgb, depth, info, now, after, "camera_optical_frame")
