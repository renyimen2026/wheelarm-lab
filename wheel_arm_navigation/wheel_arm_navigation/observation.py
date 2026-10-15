"""Validate and store aligned RGB-D samples without ROS or navigation actions."""

from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
from uuid import uuid4

import cv2
import numpy as np


def depth_meters(array, encoding):
    if encoding == "32FC1" and array.dtype.kind == "f":
        return array.astype(np.float32)
    if encoding == "16UC1" and array.dtype == np.uint16:
        return array.astype(np.float32) / 1000.0
    raise ValueError("深度只支持 32FC1（米）或 16UC1（毫米）")


def validate_sample(rgb, depth, metadata):
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("彩色图像必须是 uint8 三通道 BGR")
    if depth.dtype != np.float32 or depth.ndim != 2 or depth.shape != rgb.shape[:2]:
        raise ValueError("深度必须是与 RGB 同尺寸的 float32 米单位图像")
    if min(depth.shape) <= 0:
        raise ValueError("图像尺寸不能为空")
    info = metadata["camera_info"]
    if (info["width"] != rgb.shape[1] or info["height"] != rgb.shape[0]
            or len(info["k"]) != 9 or info["k"][0] <= 0 or info["k"][4] <= 0):
        raise ValueError("相机内参未标定或尺寸不匹配")
    if not np.isfinite(info["k"]).all():
        raise ValueError("相机内参包含非有限数值")
    json.dumps(metadata, allow_nan=False)


def depth_preview(depth):
    valid = np.isfinite(depth) & (depth > 0)
    values = depth[valid]
    lower = float(values.min()) if values.size else None
    upper = float(values.max()) if values.size else None
    gray = np.zeros(depth.shape, dtype=np.uint8)
    if values.size:
        gray[valid] = ((values - lower) / max(upper - lower, 1e-6) * 255).astype(np.uint8)
    preview = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
    preview[~valid] = 0
    return preview, {"valid_fraction": float(valid.mean()), "min_m": lower,
                     "max_m": upper, "median_m": float(np.median(values)) if values.size else None}


def save_observation(directory, rgb, depth, metadata):
    validate_sample(rgb, depth, metadata)
    preview, stats = depth_preview(depth)
    root = Path(directory).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    identifier = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:12]
    destination = root / identifier
    record = {**metadata, "schema_version": 1, "observation_id": identifier,
              "depth_unit": "meter", "depth_dtype": "float32", "depth_statistics": stats,
              "files": {"rgb": "rgb.png", "depth": "depth.npy", "depth_preview": "depth_preview.png"},
              "depth_preview_range_m": [stats["min_m"], stats["max_m"]]}
    # Publish a complete directory only after all three images and metadata exist.
    with tempfile.TemporaryDirectory(dir=root, prefix=".observation-") as temporary:
        staging = Path(temporary)
        if not cv2.imwrite(str(staging / "rgb.png"), rgb):
            raise OSError("RGB 图片写入失败")
        np.save(staging / "depth.npy", depth, allow_pickle=False)
        if not cv2.imwrite(str(staging / "depth_preview.png"), preview):
            raise OSError("深度预览写入失败")
        (staging / "metadata.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8")
        staging.rename(destination)
    return destination


def save_rgb_capture(directory, image, metadata):
    root = Path(directory).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    identifier = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:12]
    destination = root / identifier
    record = {**metadata, "schema_version": 1, "observation_id": identifier,
              "files": {"rgb": "rgb.png"}}
    serialized = json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    with tempfile.TemporaryDirectory(dir=root, prefix=".rgb-") as temporary:
        staging = Path(temporary)
        (staging / "rgb.png").write_bytes(image)
        (staging / "metadata.json").write_text(serialized, encoding="utf-8")
        staging.rename(destination)
    return destination, record
