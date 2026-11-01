#!/usr/bin/env python3
"""Generate matching handled and handleless ceramic cup models in meters."""

from collections import defaultdict
import math
from pathlib import Path
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parent
TAU = 2 * math.pi


def add(parent, tag, text=None, **attributes):
    element = ET.SubElement(parent, tag, attributes)
    if text is not None:
        element.text = str(text)
    return element


def write_xml(root, path):
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def body_mesh():
    profile = [(0, -.05), (.0185, -.05), (.022, -.05), (.024, -.0495),
               (.0248, -.048), (.025, -.045), (.025, -.035), (.025, .035), (.025, .0485)]
    profile.extend((.0235 + .0015 * math.cos(t), .0485 + .0015 * math.sin(t))
                   for t in [math.pi * i / 12 for i in range(1, 13)])
    profile.extend([(.022, .043), (.022, -.039), (.0216, -.043), (.0205, -.045),
                    (.018, -.045), (0, -.045)])
    vertices, normals, rings = [], [], []
    segments = 96
    for index, (radius, z) in enumerate(profile):
        before, after = profile[max(0, index - 1)], profile[min(len(profile) - 1, index + 1)]
        dr, dz = after[0] - before[0], after[1] - before[1]
        length = math.hypot(dr, dz)
        ring = []
        for j in range(segments if radius else 1):
            angle = TAU * j / segments
            ring.append(len(vertices))
            vertices.append((radius * math.cos(angle), radius * math.sin(angle), z))
            normals.append((dz / length * math.cos(angle), dz / length * math.sin(angle), -dr / length))
        rings.append(ring)
    faces = defaultdict(list)
    for index, (lower, upper) in enumerate(zip(rings, rings[1:])):
        material = "outside" if index < 14 else "inside"
        for j in range(segments):
            k = (j + 1) % segments
            if len(lower) == 1:
                faces[material].append((lower[0], upper[k], upper[j]))
            elif len(upper) == 1:
                faces[material].append((lower[j], lower[k], upper[0]))
            else:
                faces[material].extend([(lower[j], lower[k], upper[k]),
                                        (lower[j], upper[k], upper[j])])
    return vertices, normals, faces


def handle_center(t):
    return (0, .024 + .028 * math.cos(t), .032 * math.sin(t))


def handle_mesh():
    vertices, normals, faces = [], [], defaultdict(list)
    steps, sides, radius = 64, 16, .004
    for i in range(steps + 1):
        t = -math.pi / 2 + math.pi * i / steps
        _, y, z = handle_center(t)
        ny, nz = .032 * math.cos(t), .028 * math.sin(t)
        length = math.hypot(ny, nz)
        for j in range(sides):
            v = TAU * j / sides
            normal = (math.cos(v), math.sin(v) * ny / length, math.sin(v) * nz / length)
            vertices.append((radius * normal[0], y + radius * normal[1], z + radius * normal[2]))
            normals.append(normal)
    for i in range(steps):
        for j in range(sides):
            a, b = i * sides + j, i * sides + (j + 1) % sides
            faces["outside"].extend([(a, b, b + sides), (a, b + sides, a + sides)])
    for i, sign in [(0, -1), (steps, 1)]:
        t = -math.pi / 2 + math.pi * i / steps
        center = len(vertices)
        vertices.append(handle_center(t))
        tangent = (0, -.028 * math.sin(t), .032 * math.cos(t))
        length = math.hypot(tangent[1], tangent[2])
        normals.append(tuple(sign * value / length for value in tangent))
        for j in range(sides):
            a, b = i * sides + j, i * sides + (j + 1) % sides
            faces["outside"].append((center, b, a) if sign < 0 else (center, a, b))
    return vertices, normals, faces


def make_mesh(path, with_handle=True):
    dae = ET.Element("COLLADA", xmlns="http://www.collada.org/2005/11/COLLADASchema", version="1.4.1")
    asset = add(dae, "asset")
    add(add(asset, "contributor"), "author", "renyimen")
    add(asset, "created", "2026-10-31T00:00:00Z")
    add(asset, "modified", "2026-10-31T00:00:00Z")
    add(asset, "unit", name="meter", meter="1")
    add(asset, "up_axis", "Z_UP")
    effects, materials = add(dae, "library_effects"), add(dae, "library_materials")
    for name, color in [("outside", "0.78 0.78 0.78 1"), ("inside", "0.94 0.95 0.95 1")]:
        phong = add(add(add(effects, "effect", id=name + "-fx"), "profile_COMMON"), "technique", sid="common")
        phong = add(phong, "phong")
        add(add(phong, "ambient"), "color", color)
        add(add(phong, "diffuse"), "color", color)
        add(add(phong, "specular"), "color", "0.7 0.75 0.75 1")
        add(add(phong, "shininess"), "float", "90")
        add(add(materials, "material", id=name, name=name), "instance_effect", url="#" + name + "-fx")
    geometries = add(dae, "library_geometries")
    scene = add(add(dae, "library_visual_scenes"), "visual_scene", id="mug-scene")
    meshes = [("body", body_mesh())]
    if with_handle:
        meshes.append(("handle", handle_mesh()))
    for name, (vertices, normals, faces) in meshes:
        mesh = add(add(geometries, "geometry", id=name, name=name), "mesh")
        for field, values in [("positions", vertices), ("normals", normals)]:
            source = add(mesh, "source", id=name + "-" + field)
            identifier = name + "-" + field + "-array"
            add(source, "float_array", " ".join(f"{v:.9g}" for row in values for v in row),
                id=identifier, count=str(len(values) * 3))
            accessor = add(add(source, "technique_common"), "accessor", source="#" + identifier,
                           count=str(len(values)), stride="3")
            for axis in "XYZ":
                add(accessor, "param", name=axis, type="float")
        add(add(mesh, "vertices", id=name + "-vertices"), "input", semantic="POSITION",
            source="#" + name + "-positions")
        for material, triangles in faces.items():
            primitive = add(mesh, "triangles", material=material, count=str(len(triangles)))
            add(primitive, "input", semantic="VERTEX", source="#" + name + "-vertices", offset="0")
            add(primitive, "input", semantic="NORMAL", source="#" + name + "-normals", offset="1")
            add(primitive, "p", " ".join(str(i) for face in triangles for index in face for i in (index, index)))
        instance = add(add(scene, "node", id=name + "-node"), "instance_geometry", url="#" + name)
        binding = add(add(instance, "bind_material"), "technique_common")
        for material in faces:
            add(binding, "instance_material", symbol=material, target="#" + material)
    add(add(dae, "scene"), "instance_visual_scene", url="#mug-scene")
    write_xml(dae, path)


def make_model(path, with_handle=True):
    name = "ceramic_mug" if with_handle else "ceramic_cup"
    filename = "mug.dae" if with_handle else "cup.dae"
    sdf = ET.Element("sdf", version="1.9")
    model = add(sdf, "model", name=name)
    add(model, "static", "false")
    link = add(model, "link", name="body")
    visual = add(link, "visual", name="glazed_ceramic")
    mesh = add(add(visual, "geometry"), "mesh")
    add(mesh, "uri", f"model://{name}/meshes/{filename}")
    add(visual, "cast_shadows", "true")
    parts = []

    def collision(name, center, dimensions, angle=0, cylinder=False):
        item = add(link, "collision", name=name)
        add(item, "pose", " ".join(f"{v:.9g}" for v in (*center, angle if name.startswith("handle") else 0,
                                                       0, 0 if name.startswith("handle") else angle)))
        shape = add(add(item, "geometry"), "cylinder" if cylinder else "box")
        if cylinder:
            radius, height = dimensions
            add(shape, "radius", radius)
            add(shape, "length", height)
            volume = math.pi * radius ** 2 * height
            inertia = ((3 * radius ** 2 + height ** 2) / 12,) * 2 + (radius ** 2 / 2,)
        else:
            add(shape, "size", " ".join(f"{v:.9g}" for v in dimensions))
            x, y, z = dimensions
            volume = x * y * z
            inertia = ((y * y + z * z) / 12, (x * x + z * z) / 12, (x * x + y * y) / 12)
        # Rotate each local diagonal inertia into the model frame before summing.
        c, s = math.cos(angle), math.sin(angle)
        if name.startswith("handle"):
            ix, iy, iz = inertia
            matrix = [[ix, 0, 0], [0, c*c*iy + s*s*iz, c*s*(iy-iz)],
                      [0, c*s*(iy-iz), s*s*iy + c*c*iz]]
        else:
            ix, iy, iz = inertia
            matrix = [[c*c*ix+s*s*iy, c*s*(ix-iy), 0],
                      [c*s*(ix-iy), s*s*ix+c*c*iy, 0], [0, 0, iz]]
        parts.append((center, volume, matrix))
        ode = add(add(add(item, "surface"), "friction"), "ode")
        add(ode, "mu", "1.5")
        add(ode, "mu2", "1.5")

    collision("base", (0, 0, -.0475), (.023, .005), cylinder=True)
    for i in range(32):
        angle = TAU * i / 32
        collision(f"wall_{i:02}", (.0235 * math.cos(angle), .0235 * math.sin(angle), .002),
                  (.003, 2 * .025 * math.tan(math.pi / 32), .094), angle)
    for i in range(16 if with_handle else 0):
        a, b = (handle_center(-math.pi / 2 + math.pi * j / 16) for j in (i, i + 1))
        dy, dz = b[1] - a[1], b[2] - a[2]
        collision(f"handle_{i:02}", tuple((u+v)/2 for u, v in zip(a, b)),
                  (.008, .008, math.hypot(dy, dz) + .001), math.atan2(-dy, dz))
    total = sum(volume for _, volume, _ in parts)
    mass = .12 if with_handle else .10
    center = [sum(p[k] * volume for p, volume, _ in parts) / total for k in range(3)]
    tensor = [[0.0] * 3 for _ in range(3)]
    for p, volume, matrix in parts:
        weight = mass * volume / total
        d = [p[k] - center[k] for k in range(3)]
        for i in range(3):
            for j in range(3):
                tensor[i][j] += weight * (matrix[i][j] +
                    (sum(v*v for v in d) if i == j else 0) - d[i]*d[j])
    inertial = add(link, "inertial")
    add(inertial, "pose", " ".join(f"{v:.9g}" for v in center) + " 0 0 0")
    add(inertial, "mass", mass)
    inertia = add(inertial, "inertia")
    for key, i, j in [("ixx", 0, 0), ("iyy", 1, 1), ("izz", 2, 2),
                       ("ixy", 0, 1), ("ixz", 0, 2), ("iyz", 1, 2)]:
        add(inertia, key, f"{tensor[i][j]:.9g}")
    plugin = add(model, "plugin", filename="gz-sim-pose-publisher-system", name="gz::sim::systems::PosePublisher")
    for tag, value in [("publish_model_pose", "true"), ("publish_nested_model_pose", "true"),
                       ("publish_link_pose", "false"), ("publish_visual_pose", "false"),
                       ("publish_collision_pose", "false"), ("publish_sensor_pose", "false"),
                       ("use_pose_vector_msg", "false"), ("update_frequency", "30")]:
        add(plugin, tag, value)
    write_xml(sdf, path)


def main():
    (ROOT / "meshes").mkdir(exist_ok=True)
    make_mesh(ROOT / "meshes/mug.dae")
    make_model(ROOT / "model.sdf")
    world = ET.parse(ROOT.parents[1] / "worlds/grasp_test.sdf")
    world.find("world/include/uri").text = "model://ceramic_mug"
    # Keep the handle beyond the fingertips during the existing body-grasp test.
    world.find("world/include/pose").text = "0.6680 0 0.5988 0 0 -1.57079632679"
    write_xml(world.getroot(), ROOT.parents[1] / "worlds/grasp_mug.sdf")
    cup = ROOT.parent / "ceramic_cup"
    (cup / "meshes").mkdir(parents=True, exist_ok=True)
    make_mesh(cup / "meshes/cup.dae", with_handle=False)
    make_model(cup / "model.sdf", with_handle=False)
    world = ET.parse(ROOT.parents[1] / "worlds/grasp_test.sdf")
    world.find("world/include/uri").text = "model://ceramic_cup"
    write_xml(world.getroot(), ROOT.parents[1] / "worlds/grasp_cup.sdf")


if __name__ == "__main__":
    main()
