"""CPU-only split/loader and training-loop tests; no pretrained model."""

from collections import Counter
from types import SimpleNamespace
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_smolvla_batch as pilot
from test_lerobot_conversion import synthetic_episode
from wheel_arm_manipulation.lerobot_batch_conversion import convert_batch
from wheel_arm_manipulation.smolvla_batch_data import (
    BalancedStarts, PhaseBalancedStarts, load_batch, training_phase_groups,active_training_groups)
from lerobot.optim.optimizers import AdamWConfig

torch = pilot.torch
torch.set_num_threads(1)


class TrackingConfig(AdamWConfig):
    def build(self, params):
        self.calls = getattr(self, "calls", 0)+1
        self.optimizer = super().build(params)
        return self.optimizer


class BatchPilotTests(unittest.TestCase):
    def test_active_sampling_excludes_waits_but_retains_terminal_holds_without_rewriting(self):
        rows=[]
        for i in range(30):
            action=[0.,-.6,.9,-.3,.06+(.0006*i if i<10 else 0)]
            rows.append({'source_grid_row':{'action':action,'state':[0.,-.6,.9,-.3,.06]}})
        data=SimpleNamespace(split='train',groups={'one':[0,10,20]},mapping=rows)
        original=json.dumps(rows)
        selected=active_training_groups(data,{0:'open',10:'approach',20:'lift'},10)
        self.assertEqual(selected,{'one':[0,20]})
        self.assertEqual(json.dumps(rows),original)
        data.split='validation'
        with self.assertRaises(ValueError):active_training_groups(data,{0:'open',10:'approach',20:'lift'},10)

    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="wheelarm-pilot-test-")
        cls.root = Path(cls.temporary.name)
        raw = cls.root/"raw"
        entries = []
        for name, split, gap in (("train1", "train", None), ("train2", "train", 6), ("val1", "validation", None)):
            source = raw/name
            synthetic_episode(source, missing_index=gap, task_id=name, frame_count=80)
            entries.append({"split": split, "task_id": name, "episode_dir": str(source)})
        split_file = raw/"collection_split.json"
        split_file.write_text(json.dumps({"split_unit": "original_recording_all_derived_segments_inherit_split",
                                         "episodes": entries}))
        cls.dataset = cls.root/"dataset"
        convert_batch(split_file, cls.dataset)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_loader_respects_episode_boundaries_shapes_and_source_actions(self):
        train, val, stats = load_batch(self.dataset)
        self.assertEqual(set(train.groups), {"train1", "train2"})
        self.assertEqual(set(val.groups), {"val1"})
        self.assertEqual(stats, json.loads((self.dataset/"train/meta/stats.json").read_text()))
        for data in (train, val):
            for indices in data.groups.values():
                for index in (indices[0], indices[-1]):
                    batch = data.batch(index)
                    self.assertEqual(tuple(batch["observation.images.mast"].shape), (1, 1, 3, 12, 16))
                    self.assertEqual(tuple(batch["observation.state"].shape), (1, 1, 5))
                    self.assertEqual(tuple(batch["action"].shape), (1, 50, 5))
                    self.assertFalse(batch["action_is_pad"].any())
            self.assertTrue(set(data.evaluation_indices()) <= data.start_set)
            excluded = next(i for i in range(len(data.mapping)) if i not in data.start_set)
            with self.assertRaisesRegex(ValueError, "完整 50 步"):
                data.batch(excluded)

    def test_sampler_uses_training_only_and_balances_original_tasks(self):
        train, _, _ = load_batch(self.dataset)
        sampler = BalancedStarts(train.groups, seed=0)
        counts = dict.fromkeys(train.groups, 0)
        for _ in range(1000):
            task, index = sampler.draw()
            self.assertIn(index, train.groups[task])
            counts[task] += 1
        self.assertTrue(all(400 < count < 600 for count in counts.values()))

    def test_phase_sampler_balances_unequal_buckets_without_inventing_missing_phases(self):
        groups = {"a": {"close": [1], "approach": list(range(2, 102))},
                  "b": {"lift": [102], "before_or_between_phases": [103]}}
        first, second = PhaseBalancedStarts(groups), PhaseBalancedStarts(groups)
        draws = [first.draw() for _ in range(12000)]
        self.assertEqual(draws, [second.draw() for _ in draws])
        counts = Counter()
        for task, index in draws:
            phase = next(p for p, rows in groups[task].items() if index in rows)
            counts[task, phase] += 1
        self.assertEqual(len(counts), 4)
        self.assertTrue(all(2700 < n < 3300 for n in counts.values()))
        for invalid in ({}, {"a": {}}, {"a": {"close": []}}):
            with self.assertRaises(ValueError):
                PhaseBalancedStarts(invalid)

    def test_training_phase_groups_pin_events_and_exclude_validation(self):
        train, val, _ = load_batch(self.dataset)
        groups, labels, hashes = training_phase_groups(self.dataset, train)
        self.assertEqual(set(groups), set(train.groups))
        self.assertEqual(set(labels), train.start_set)
        self.assertEqual(set(hashes), set(train.groups))
        for task, phases in groups.items():
            self.assertEqual(sorted(i for rows in phases.values() for i in rows), train.groups[task])
        with self.assertRaisesRegex(ValueError, "仅允许训练集"):
            training_phase_groups(self.dataset, val)
        source = json.loads((self.dataset/"source_manifest.json").read_text())["sources"][0]
        path = Path(source["source_episode"])/"task_events.jsonl"
        saved = path.read_bytes()
        try:
            path.write_bytes(saved+b"\n")
            with self.assertRaisesRegex(ValueError, "已变化"):
                training_phase_groups(self.dataset, train)
        finally:
            path.write_bytes(saved)

    def test_changed_normalization_statistics_rejected(self):
        path = self.dataset/"training_stats.json"
        saved = path.read_bytes()
        try:
            path.write_bytes(saved+b"\n")
            with self.assertRaisesRegex(ValueError, "归一化统计"):
                load_batch(self.dataset)
        finally:
            path.write_bytes(saved)

    def test_shared_source_task_in_validation_is_rejected(self):
        manifest = self.dataset/"validation/conversion_manifest.json"
        mapping = self.dataset/"validation/source_mapping.jsonl"
        saved_m, saved_rows = manifest.read_bytes(), mapping.read_bytes()
        try:
            m = json.loads(saved_m)
            for ep in m["episodes"]:
                ep["source_task_id"] = "train1"
            rows = [json.loads(s) for s in saved_rows.splitlines()]
            for row in rows:
                row["source_task_id"] = "train1"
            manifest.write_text(json.dumps(m))
            mapping.write_text("".join(json.dumps(r)+"\n" for r in rows))
            with self.assertRaisesRegex(ValueError, "数据泄漏"):
                load_batch(self.dataset)
        finally:
            manifest.write_bytes(saved_m)
            mapping.write_bytes(saved_rows)

    def test_update_loop_preserves_rng_and_keeps_evaluation_out_of_optimizer(self):
        model = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(1.)
        config, noises = TrackingConfig(lr=.01), []

        def next_batch():
            return "train1", 0, torch.ones(1, 1)

        def loss_function(policy, batch):
            noises.append(torch.rand(1).item())
            return policy(batch).square().mean(), {}

        def evaluate(step):
            self.assertFalse(model.training)
            self.assertFalse(torch.is_grad_enabled())
            torch.rand(100)
            summary = {"evaluated_starts": 1, "fixed_flow_loss": model.weight.item()**2,
                       "joint_mae_rad": [0.]*4, "gripper_mae_mm": 0.}
            return {"splits": {s: {"summary": summary} for s in ("train", "validation")}}

        torch.manual_seed(19)
        expected = torch.rand(4).tolist()
        torch.manual_seed(19)
        result = pilot.run_updates(model, next_batch, evaluate, config, 4, 2, "cpu", loss_function=loss_function)
        self.assertEqual(noises, expected)
        self.assertEqual(config.calls, 1)
        self.assertEqual(result["updates_per_original_recording"], {"train1": 4})
        self.assertEqual([e["step"] for e in result["evaluations"]], [0, 2, 4])
        self.assertEqual(config.optimizer.state[model.weight]["step"].item(), 4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
