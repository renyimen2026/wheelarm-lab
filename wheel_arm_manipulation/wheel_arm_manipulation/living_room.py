"""Controlled house layout; world placement is never a cup-perception input."""

from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation
import yaml


def pose_matrix(values):
    values = np.asarray(values, dtype=float)
    if values.shape != (6,) or not np.isfinite(values).all():
        raise ValueError("Expected a finite xyz/rpy pose")
    result = np.eye(4)
    result[:3, 3] = values[:3]
    result[:3, :3] = Rotation.from_euler("xyz", values[3:]).as_matrix()
    return result


def layout(config, gazebo_share):
    with Path(config).open(encoding="utf-8") as stream:
        settings = yaml.safe_load(stream)
    world = ET.parse(Path(gazebo_share) / "worlds/indoor_house.sdf")
    root = world.getroot().find("world")
    models = {model.get("name"): model for model in root.findall("model")}
    poses = {name: pose_matrix([float(v) for v in models[name].findtext("pose").split()])
             for name in settings["collision_bounds"]}
    bounds = {name: np.array(value, dtype=float) for name, value in settings["collision_bounds"].items()}
    if any(value.shape != (2, 3) or not np.isfinite(value).all() or np.any(value[1] <= value[0])
           for value in bounds.values()):
        raise ValueError("Invalid fixture collision bounds")
    name = settings["table_model"]
    table = poses[name]
    cup_local = np.array([*settings["cup_offset"], bounds[name][1, 2]+settings["cup_height"]/2+.0002])
    if not np.isfinite(cup_local).all() or not .4 <= settings["stand_off"] <= .8:
        raise ValueError("Invalid cup placement or stand-off")
    if np.any(cup_local[:2] < bounds[name][0, :2]+.05) or np.any(cup_local[:2] > bounds[name][1, :2]-.05):
        raise ValueError("Cup must be safely inside the tabletop")
    cup = table[:3, :3] @ cup_local + table[:3, 3]
    yaw = Rotation.from_matrix(table[:3, :3]).as_euler("xyz")[2]-np.pi/2
    dock = cup[:2]-settings["stand_off"]*np.array([np.cos(yaw), np.sin(yaw)])
    base = pose_matrix([*dock, 0., 0., 0., yaw])
    # Move the cup independently of the established robot docking pose.
    offset = float(settings.get("cup_world_x_offset", 0.))
    if not np.isfinite(offset):
        raise ValueError("Cup world X offset must be finite")
    cup[0] += offset
    return settings, world, poses, bounds, cup, base


def write_world(config, gazebo_share, destination):
    settings, world, poses, bounds, cup, base = layout(config, gazebo_share)
    root = world.getroot().find("world")
    include = ET.SubElement(root, "include")
    ET.SubElement(include, "uri").text = "model://ceramic_cup"
    ET.SubElement(include, "name").text = "test_cup"
    ET.SubElement(include, "pose").text = " ".join(map(str, [*cup, 0, 0, 0]))
    marker = ET.SubElement(root, "model", name="living_room_dock")
    ET.SubElement(marker, "static").text = "true"
    yaw = Rotation.from_matrix(base[:3, :3]).as_euler("xyz")[2]
    ET.SubElement(marker, "pose").text = " ".join(map(str, [*base[:2, 3], .004, 0, 0, yaw]))
    link = ET.SubElement(marker, "link", name="marker")
    for index, (position, size) in enumerate([
            ((-.34, 0, 0), (.01, .56, .003)), ((.34, 0, 0), (.01, .56, .003)),
            ((0, -.28, 0), (.68, .01, .003)), ((0, .28, 0), (.68, .01, .003)),
            ((.46, 0, 0), (.20, .025, .003))]):
        visual = ET.SubElement(link, "visual", name=f"dock_{index}")
        ET.SubElement(visual, "pose").text = " ".join(map(str, [*position, 0, 0, 0]))
        ET.SubElement(ET.SubElement(ET.SubElement(visual, "geometry"), "box"), "size").text = " ".join(map(str, size))
        material = ET.SubElement(visual, "material")
        ET.SubElement(material, "ambient").text = "0.1 0.65 0.8 1"
        ET.SubElement(material, "diffuse").text = "0.1 0.65 0.8 1"
    world.write(destination, encoding="utf-8", xml_declaration=True)
    return settings, cup, base
