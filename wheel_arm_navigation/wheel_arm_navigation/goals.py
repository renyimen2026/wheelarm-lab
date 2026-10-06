"""Validated, atomic storage of map-frame named poses."""

import math
from pathlib import Path
import tempfile

import yaml


def validate(document):
    if not isinstance(document, dict) or document.get("frame_id") != "map":
        raise ValueError("目标点文件必须包含 frame_id: map")
    goals = document.get("goals")
    if not isinstance(goals, dict):
        raise ValueError("goals 必须是名称到位姿的映射")
    for name, pose in goals.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("目标点名称必须是非空字符串")
        if not isinstance(pose, dict):
            raise ValueError(f"{name}: 位姿必须包含 x、y、yaw")
        for key in ("x", "y", "yaw"):
            value = pose.get(key)
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value)):
                raise ValueError(f"{name}: {key} 必须是有限数值")
    return document


def load(path):
    with Path(path).expanduser().open(encoding="utf-8") as stream:
        try:
            return validate(yaml.safe_load(stream))
        except yaml.YAMLError as error:
            raise ValueError(f"目标点 YAML 格式错误：{error}") from error


def save_pose(path, name, x, y, yaw, overwrite=False):
    path = Path(path).expanduser()
    document = load(path) if path.exists() else {"frame_id": "map", "goals": {}}
    if name in document["goals"] and not overwrite:
        raise ValueError(f"目标点 {name} 已存在；覆盖时添加 --overwrite")
    document["goals"][name] = {"x": x, "y": y, "yaw": yaw}
    validate(document)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=path.parent,
                prefix=".goals-", suffix=".yaml", delete=False) as stream:
            temporary = Path(stream.name)
            yaml.safe_dump(document, stream, allow_unicode=True, sort_keys=False)
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
