"""Synthetic real-LeRobot split/run tests, without a model or real recordings."""

import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_lerobot_conversion import synthetic_episode, write_jsonl
from wheel_arm_manipulation.lerobot_batch_conversion import convert_batch, prepare_batch,previous_training_statistics
from wheel_arm_manipulation.time_grid import resample_episode


class BatchConversionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="wheelarm-batch-test-")
        self.base = Path(self.temporary.name)
        self.raw = self.base/"raw"
        self.entries = []
        for name, split, gap in (("train1", "train", None), ("train2", "train", 6),
                                 ("val1", "validation", None)):
            root = self.raw/name
            synthetic_episode(root, missing_index=gap, task_id=name, frame_count=80 if name == "train1" else 25)
            self.entries.append({"split": split, "episode_dir": str(root), "task_id": name})
        # Ensure validation stats differ, to detect accidental mixing or reuse.
        val = self.raw/"val1"
        frames = [json.loads(s) for s in (val/"frames.jsonl").read_text().splitlines()]
        refs = [json.loads(s) for s in (val/"controller_references.jsonl").read_text().splitlines()]
        for row in frames:
            row["state"]["values"][0] += .5
        for row in refs:
            if row["controller"] == "arm":
                row["positions"][0] += .5
        write_jsonl(val/"frames.jsonl", frames)
        write_jsonl(val/"controller_references.jsonl", refs)
        resample_episode(val, image_policy="available-past")
        self.split_file = self.raw/"collection_split.json"
        self.save_split()

    def tearDown(self):
        self.temporary.cleanup()

    def save_split(self):
        self.split_file.write_text(json.dumps({
            "split_unit": "original_recording_all_derived_segments_inherit_split", "episodes": self.entries}))

    def test_roundtrip_preserves_gaps_splits_images_and_training_only_statistics(self):
        before = {str(p.relative_to(self.raw)): p.read_bytes() for p in self.raw.rglob("*") if p.is_file()}
        output = self.base/"dataset"
        report = convert_batch(self.split_file, output)
        self.assertEqual(report["status"], "verified")
        self.assertEqual(report["splits"]["train"]["episodes"], 3)
        self.assertEqual(report["splits"]["validation"]["episodes"], 1)
        self.assertTrue(report["splits"]["train"]["checks"]["no_chunk_crosses_episode_boundary"])
        self.assertEqual(report["splits"]["train"]["checks"]["chunks"]["50"]["complete_starts_checked"], 21)
        self.assertGreater(report["splits"]["train"]["checks"]["chunks"]["50"]["padded_starts_checked_and_excluded"], 0)
        self.assertEqual((output/"training_stats.json").read_bytes(), (output/"train/meta/stats.json").read_bytes())
        self.assertNotEqual((output/"training_stats.json").read_bytes(), (output/"validation/meta/stats.json").read_bytes())
        train = json.loads((output/"train/conversion_manifest.json").read_text())
        val = json.loads((output/"validation/conversion_manifest.json").read_text())
        self.assertEqual({ep["source_task_id"] for ep in train["episodes"]}, {"train1", "train2"})
        self.assertEqual({ep["source_task_id"] for ep in val["episodes"]}, {"val1"})
        split_runs = [ep for ep in train["episodes"] if ep["source_task_id"] == "train2"]
        self.assertEqual(len(split_runs), 2)
        self.assertGreater(split_runs[1]["first_source_grid_index"], split_runs[0]["last_source_grid_index"]+1)
        self.assertTrue(all(ep["segment_task_success"] is None for ep in train["episodes"]))
        self.assertEqual(before, {str(p.relative_to(self.raw)): p.read_bytes() for p in self.raw.rglob("*") if p.is_file()})
        self.assertFalse(report["normalization_stats"]["validation_used"])

    def test_duplicate_source_across_splits_rejected_before_output(self):
        self.entries.append({**self.entries[0], "split": "validation"})
        self.save_split()
        output = self.base/"duplicate"
        with self.assertRaisesRegex(ValueError, "重复出现"):
            convert_batch(self.split_file, output)
        self.assertFalse(output.exists())

    def test_wrong_task_id_rejected(self):
        self.entries[1]["task_id"] = "incorrect"
        self.save_split()
        with self.assertRaisesRegex(ValueError, "任务 ID"):
            prepare_batch(self.split_file)

    def test_inherited_training_statistics_require_unchanged_training_sources(self):
        output=self.base/'dataset'
        convert_batch(self.split_file,output)
        prepared=prepare_batch(self.split_file)[2]
        raw,origin=previous_training_statistics(output,prepared)
        self.assertEqual(raw,(output/'training_stats.json').read_bytes())
        self.assertEqual(origin['source_training_recordings'],2)
        changed=[dict(p) for p in prepared]
        changed[0]['split']='validation'
        with self.assertRaisesRegex(ValueError,'当前训练集'):
            previous_training_statistics(output,changed)
        changed=[dict(p) for p in prepared]
        changed[0]['hashes']={}
        with self.assertRaisesRegex(ValueError,'原始文件未变化'):
            previous_training_statistics(output,changed)

    def test_existing_output_and_output_inside_raw_rejected(self):
        output = self.base/"existing"
        output.mkdir()
        with self.assertRaises(FileExistsError):
            convert_batch(self.split_file, output)
        with self.assertRaisesRegex(ValueError, "原始批次目录之外"):
            convert_batch(self.split_file, self.raw/"output")


if __name__ == "__main__":
    unittest.main(verbosity=2)
