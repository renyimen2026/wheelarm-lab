"""Synthetic local LeRobot integration checks; never use a real demonstration."""

import json
from importlib.util import find_spec
import os
from pathlib import Path
import sys
import tempfile
import unittest

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"wheel_arm_manipulation"))

import numpy as np
from PIL import Image

from wheel_arm_manipulation.action_alignment import STATE_NAMES, STATE_UNITS
from wheel_arm_manipulation.controller_reference import ARM_JOINTS, FINGER_JOINTS
from wheel_arm_manipulation.lerobot_conversion import convert_episode, prepare_episode
from wheel_arm_manipulation.time_grid import AVAILABLE_GRID_FILE, resample_episode
if find_spec("lerobot") is not None:
    from wheel_arm_manipulation.smolvla_interface import check_interface, validate_stats
    from lerobot.configs.types import PolicyFeature, FeatureType


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False)+"\n" for row in rows), encoding="utf-8")


def synthetic_episode(root, missing_index=None, task_id="synthetic", frame_count=25):
    (root/"images").mkdir(parents=True)
    frames, references, events = [], [], []
    for index in range(frame_count):
        stamp = 1_000_000_000+index*100_000_000
        arm = [index*.01, -.2, .4, -.2]
        for group, names, units, values in (
                ("arm", ARM_JOINTS, ["rad"]*4, arm),
                ("gripper", FINGER_JOINTS, ["m"]*2, [.02, .02])):
            references.append({"kind": "controller_reference", "controller": group,
                               "stamp_ns": stamp, "clock": "ros_sim_time",
                               "semantics": "current_interpolated_position_reference",
                               "names": list(names), "units": units, "positions": values})
        if index == missing_index:
            continue
        image_file = f"images/{index:06d}.png"
        pixels = np.zeros((12, 16, 3), dtype=np.uint8)
        pixels[:, :, 0] = (index*7) % 256
        pixels[:, :, 1] = np.arange(16, dtype=np.uint8)[None, :]*11
        pixels[:, :, 2] = np.arange(12, dtype=np.uint8)[:, None]*13
        Image.fromarray(pixels).save(root/image_file)
        frames.append({"schema_version": 1, "kind": "observation_only", "clock": "ros_sim_time",
                       "frame_index": len(frames), "image": {"file": image_file, "stamp_ns": stamp},
                       "joints_stamp_ns": stamp, "timestamp_difference_ms": 0.,
                       "image_age_ms": 20., "joints_age_ms": 20., "instruction": "拿起桌上的杯子并保持夹持",
                       "state": {"names": list(STATE_NAMES), "units": list(STATE_UNITS), "values": arm+[.04]}})
    write_jsonl(root/"frames.jsonl", frames)
    write_jsonl(root/"controller_references.jsonl", references)
    (root/"summary.json").write_text(json.dumps({"collection_status": "finished", "frames": len(frames),
        "actual_hz": 10., "controller_references": {"file": "controller_references.jsonl",
        "both_controllers_recorded": True, "saved": {"arm": frame_count, "gripper": frame_count}}}))

    def event(stamp, kind, **details):
        events.append({"task_id": task_id, "event_index": len(events), "stamp_ns": stamp,
                       "event": kind, **details})

    lift_end = 1.+(frame_count-1)*.1-.4
    for phase, start, end in [("open", 1.2, 1.4), ("pregrasp", 1.4, 1.6),
                              ("approach", 1.6, 1.8), ("close", 1.8, 2.), ("lift", 2., lift_end)]:
        event(round(start*1e9), "phase_started", phase=phase)
        if phase == "lift":
            event(round((lift_end-.5)*1e9), "lift_verified")
        event(round(end*1e9), "phase_finished", phase=phase)
    event(round((lift_end+.1)*1e9), "task_finished", execution_status="finished", task_success=True)
    write_jsonl(root/"task_events.jsonl", events)
    (root/"task_metadata.json").write_text(json.dumps({"task_id": task_id, "source": "cup_grasp",
        "execution_status": "finished", "task_success": True, "lift_verified": True,
        "clock": "ros_sim_time", "finish": "hold"}))
    resample_episode(root, image_policy="available-past")


@unittest.skipUnless(find_spec("lerobot") is not None, "Use the isolated .venv-lerobot environment")
class ConversionTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="wheelarm-lerobot-test-")
        self.base = Path(self.temporary.name)
        self.source = self.base/"source"
        synthetic_episode(self.source)

    def tearDown(self):
        self.temporary.cleanup()

    def test_real_package_roundtrip_preserves_source_and_single_step_semantics(self):
        before = {str(p.relative_to(self.source)): p.read_bytes() for p in self.source.rglob("*") if p.is_file()}
        output = self.base/"dataset"
        result = convert_episode(self.source, output)
        self.assertEqual(result["status"], "verified")
        self.assertEqual(result["frames"], 15)
        self.assertEqual(result["checks"]["images_pixel_checked"], 15)
        self.assertEqual(result["checks"]["complete_chunk_starts_checked"], 12)
        self.assertEqual(result["checks"]["padded_tail_starts_excluded"], 3)
        self.assertEqual(result["checks"]["single_action_shape"], [5])
        self.assertEqual(result["checks"]["loaded_chunk_shape"], [4, 5])
        info = json.loads((output/"meta/info.json").read_text())
        self.assertEqual(info["features"]["action"]["shape"], [5])
        mapping = [json.loads(line) for line in (output/"source_mapping.jsonl").read_text().splitlines()]
        self.assertEqual(mapping[0]["dataset_timestamp_seconds"], 0.)
        self.assertEqual(mapping[0]["source_grid_row"]["grid_index"], 1)
        self.assertEqual(mapping[0]["source_grid_row"]["stamp_ns"], 1_100_000_000)
        self.assertEqual(mapping[0]["source_grid_row"]["action_stamp_ns"], 1_200_000_000)
        self.assertEqual(before, {str(p.relative_to(self.source)): p.read_bytes()
                                  for p in self.source.rglob("*") if p.is_file()})
        self.assertTrue((output/"conversion_report.json").is_file())

    def test_existing_output_is_never_overwritten(self):
        output = self.base/"existing"
        output.mkdir()
        with self.assertRaises(FileExistsError):
            convert_episode(self.source, output)

    def test_output_inside_source_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "源记录目录之外"):
            convert_episode(self.source, self.source/"dataset")

    def test_stale_or_modified_grid_cannot_be_exported(self):
        path = self.source/AVAILABLE_GRID_FILE
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        next(row for row in rows if row["status"] == "accepted")["action"][0] += .01
        write_jsonl(path, rows)
        with self.assertRaisesRegex(ValueError, "时间网格文件"):
            prepare_episode(self.source)

    def test_disconnected_runs_are_not_concatenated(self):
        source = self.base/"gapped"
        synthetic_episode(source, missing_index=6)
        with self.assertRaisesRegex(ValueError, "不拼接缺口"):
            prepare_episode(source)

    def test_smolvla_official_config_loader_and_numeric_processors_without_weights(self):
        output = self.base/"interface"
        convert_episode(self.source, output)
        report = check_interface(output)
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["complete_chunk_starts_checked"], 12)
        self.assertEqual(report["policy_features"]["observation.images.mast"]["shape"], [3, 12, 16])
        self.assertEqual(report["first_batch_shapes"]["observation.images.mast"], [8, 1, 3, 12, 16])
        self.assertEqual(report["first_batch_shapes"]["observation.state"], [8, 1, 5])
        self.assertEqual(report["first_batch_shapes"]["action"], [8, 4, 5])
        self.assertEqual(report["loader_delta_timestamps"]["action"], [0., .1, .2, .3])
        self.assertEqual(report["physical_action_offsets_seconds"], [.1, .2, .3, .4])
        self.assertEqual(report["padded_tail_starts_checked_and_excluded"], 3)
        self.assertLessEqual(report["max_normalization_roundtrip_error"]["action"], 1e-6)
        self.assertFalse(report["model_instantiated"])
        self.assertFalse(report["model_forward_checked"])
        self.assertFalse(report["language_tokenization_checked"])

    def test_missing_or_invalid_normalization_stats_never_silently_use_identity(self):
        features = {"observation.state": PolicyFeature(type=FeatureType.STATE, shape=(5,)),
                    "action": PolicyFeature(type=FeatureType.ACTION, shape=(5,))}
        good = {key: {"mean": [0.]*5, "std": [1.]*5} for key in features}
        for bad in ({}, {"action": good["action"]},
                    {**good, "action": {"mean": [0.]*5}},
                    {**good, "action": {"mean": [0.]*5, "std": [-1.]*5}},
                    {**good, "action": {"mean": [float("nan")]*5, "std": [1.]*5}}):
            with self.subTest(stats=bad), self.assertRaises(ValueError):
                validate_stats(bad, features)

    def test_smolvla_interface_rejects_changed_action_provenance(self):
        output = self.base/"changed_mapping"
        convert_episode(self.source, output)
        path = output/"source_mapping.jsonl"
        mapping = [json.loads(line) for line in path.read_text().splitlines()]
        mapping[0]["source_grid_row"]["future_absolute_references"][0][0] += .1
        write_jsonl(path, mapping)
        with self.assertRaisesRegex(ValueError, "官方加载器动作块"):
            check_interface(output)

    def test_smolvla_interface_rejects_padding_start_in_full_start_list(self):
        output = self.base/"bad_starts"
        convert_episode(self.source, output)
        path = output/"conversion_manifest.json"
        manifest = json.loads(path.read_text())
        manifest["full_chunk_start_indices"].append(manifest["rows"]-1)
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "完整动作块起点"):
            check_interface(output)


if __name__ == "__main__":
    unittest.main(verbosity=2)
