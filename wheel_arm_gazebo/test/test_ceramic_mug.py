from collections import Counter
import importlib.util
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models/ceramic_mug"
spec = importlib.util.spec_from_file_location("mug_generator", MODEL / "generate_model.py")
generator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(generator)


@pytest.mark.parametrize("builder", [generator.body_mesh, generator.handle_mesh])
def test_mesh_is_closed_finite_and_outward_facing(builder):
    vertices, normals, groups = builder()
    vertices, normals = np.array(vertices), np.array(normals)
    triangles = np.array([face for group in groups.values() for face in group])
    assert np.isfinite(vertices).all() and np.isfinite(normals).all()
    assert np.allclose(np.linalg.norm(normals, axis=1), 1)
    edges = Counter(tuple(sorted((int(a), int(b)))) for face in triangles
                    for a, b in zip(face, np.roll(face, -1)))
    assert set(edges.values()) == {2}
    a, b, c = vertices[triangles].transpose(1, 0, 2)
    cross = np.cross(b - a, c - a)
    assert np.all(np.linalg.norm(cross, axis=1) > 1e-10)
    assert np.all(np.einsum("ij,ij->i", cross, normals[triangles].mean(axis=1)) > 0)
    assert np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6 > 0


@pytest.mark.parametrize("with_handle", [True, False])
def test_model_has_hollow_collision_walls_and_positive_inertia(with_handle):
    name = "ceramic_mug" if with_handle else "ceramic_cup"
    link = ET.parse(ROOT / "models" / name / "model.sdf").find("model/link")
    collisions = link.findall("collision")
    assert len([c for c in collisions if c.get("name").startswith("wall_")]) == 32
    assert len([c for c in collisions if c.get("name").startswith("handle_")]) == (16 if with_handle else 0)
    cylinders = [c for c in collisions if c.find("geometry/cylinder") is not None]
    assert len(cylinders) == 1
    assert float(cylinders[0].findtext("geometry/cylinder/length")) == .005
    values = link.find("inertial/inertia")
    xx, yy, zz, xy, xz, yz = [float(values.findtext(key)) for key in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")]
    assert np.linalg.eigvalsh([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]]).min() > 0
    assert float(link.findtext("inertial/mass")) == (.12 if with_handle else .10)
    filename = "mug.dae" if with_handle else "cup.dae"
    assert link.findtext("visual/geometry/mesh/uri") == f"model://{name}/meshes/{filename}"
    if not with_handle:
        pose = [float(v) for v in link.findtext("inertial/pose").split()]
        assert np.allclose(pose[:2] + pose[3:], 0, atol=1e-12)
        assert -.05 < pose[2] < 0  # The solid bottom lowers the center of mass.
        assert xx == pytest.approx(yy)
        assert np.allclose([xy, xz, yz], 0, atol=1e-12)


@pytest.mark.parametrize("with_handle", [True, False])
def test_generated_assets_are_reproducible(tmp_path, with_handle):
    model = MODEL if with_handle else ROOT / "models/ceramic_cup"
    filename = "mug.dae" if with_handle else "cup.dae"
    for function, name in [(generator.make_mesh, "meshes/" + filename), (generator.make_model, "model.sdf")]:
        generated = tmp_path / Path(name).name
        function(generated, with_handle=with_handle)
        assert generated.read_bytes() == (model / name).read_bytes()


@pytest.mark.parametrize("with_handle", [True, False])
def test_exterior_glaze_is_neutral_light_gray(with_handle):
    namespace = {"c": "http://www.collada.org/2005/11/COLLADASchema"}
    mesh = ET.parse(MODEL / "meshes/mug.dae" if with_handle else ROOT / "models/ceramic_cup/meshes/cup.dae")
    color = mesh.find(".//c:effect[@id='outside-fx']//c:diffuse/c:color", namespace)
    red, green, blue, alpha = map(float, color.text.split())
    assert red == green == blue and .65 <= red <= .95 and alpha == 1


def test_handleless_mesh_keeps_body_without_hidden_handle():
    namespace = {"c": "http://www.collada.org/2005/11/COLLADASchema"}
    mug = ET.parse(MODEL / "meshes/mug.dae")
    cup = ET.parse(ROOT / "models/ceramic_cup/meshes/cup.dae")
    geometries = cup.findall("c:library_geometries/c:geometry", namespace)
    assert [g.get("id") for g in geometries] == ["body"]
    body = mug.find("c:library_geometries/c:geometry[@id='body']", namespace)
    body.tail = geometries[0].tail = None
    assert ET.tostring(body) == ET.tostring(geometries[0])


@pytest.mark.parametrize("with_handle", [True, False])
def test_new_world_only_replaces_cup_asset(with_handle):
    old = ET.parse(ROOT / "worlds/grasp_test.sdf")
    new = ET.parse(ROOT / ("worlds/grasp_mug.sdf" if with_handle else "worlds/grasp_cup.sdf"))
    assert new.findtext("world/include/uri") == ("model://ceramic_mug" if with_handle else "model://ceramic_cup")
    new.find("world/include/uri").text = "model://test_cup"
    if with_handle:
        assert new.findtext("world/include/pose").endswith("-1.57079632679")
    else:
        assert new.findtext("world/include/pose") == old.findtext("world/include/pose")
    new.find("world/include/pose").text = old.findtext("world/include/pose")
    for tree in (old, new):
        ET.indent(tree, space="  ")
    assert ET.tostring(old.getroot()) == ET.tostring(new.getroot())


def test_launch_selects_mug_without_changing_legacy_default():
    from launch import LaunchContext
    from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
    from launch.utilities import normalize_to_list_of_substitutions, perform_substitutions

    path = ROOT.parent / "wheel_arm_manipulation/launch/cup_test.launch.py"
    spec = importlib.util.spec_from_file_location("cup_launch", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    launch = module.generate_launch_description()
    argument = next(a for a in launch.entities if isinstance(a, DeclareLaunchArgument) and a.name == "cup_model")
    context = LaunchContext()
    assert perform_substitutions(context, argument.default_value) == "test_cup"
    include = next(a for a in launch.entities if isinstance(a, IncludeLaunchDescription))
    world = dict(include.launch_arguments)["world"]
    for name, expected in [("test_cup", "grasp_test.sdf"), ("ceramic_mug", "grasp_mug.sdf"),
                            ("ceramic_cup", "grasp_cup.sdf")]:
        context.launch_configurations["cup_model"] = name
        assert perform_substitutions(context, normalize_to_list_of_substitutions(world)).endswith(expected)
