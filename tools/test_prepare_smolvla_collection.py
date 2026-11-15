"""Collection split provenance and held-out exclusion checks; no ROS motion."""
import json
import hashlib
from pathlib import Path
import tempfile
import unittest

from prepare_smolvla_collection import audit, prepare, finalize


def write(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def fixture(root):
    entries, slots = [], []
    for index in range(1, 9):
        episode = root / f"episode_{index}"
        episode.mkdir()
        task = f"task_{index}"
        split = "train" if index <= 6 else "independent_test"
        entry = dict(slot=index, task_id=task, episode_dir=str(episode), split=split)
        entries.append(entry)
        slots.append(dict(slot=index, split=split))
        write(episode / "task_metadata.json", dict(task_id=task, task_success=True))
        events = []
        for phase, begin, end in [('open', 1, 5), ('pregrasp', 5, 10), ('approach', 10, 16),
                                  ('close', 16, 20), ('lift', 20, 60)]:
            events += [dict(task_id=task, event="phase_started", phase=phase, stamp_ns=begin*10**9),
                       dict(task_id=task, event="phase_finished", phase=phase, stamp_ns=end*10**9)]
        events += [dict(task_id=task, event="hold_started", stamp_ns=40*10**9),
                   dict(task_id=task, event="hold_verified", stamp_ns=50*10**9)]
        events.sort(key=lambda e: e['stamp_ns'])
        (episode / 'task_events.jsonl').write_text('\n'.join(json.dumps(e) for e in events))
        grid = [dict(grid_index=i, status='accepted', stamp_ns=10**9+i*10**8,
                     state=[0, 0, 0, 0, .05], action=[0, 0, 0, 0, .049]) for i in range(590)]
        (episode / 'grasp_hold_available_time_grid.jsonl').write_text('\n'.join(json.dumps(r) for r in grid))
        write(episode / 'grasp_hold_available_time_grid_report.json',
              {'accepted_image_age_ms': {'p95': 80}})
    write(root / 'collection_plan.json', {'slots': slots})
    write(root / 'collection_report.json', {'status': 'verified', 'episodes': entries})
    old = root / 'old.json'
    write(old, {'episodes': [dict(episode_dir='old_train', task_id='old_train', split='train'),
                            dict(episode_dir='old_validation', task_id='old_validation', split='validation')]})
    return entries, old


class CollectionTests(unittest.TestCase):
    def test_identical_stat_values_share_frozen_bytes_without_recalculation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / 'datasets'
            for label, stats in [('development', {'state': [1, 2], 'action': [3, 4]}),
                                 ('independent_test', {'action': [3, 4], 'state': [1, 2]})]:
                dest = data / label
                (dest / 'train/meta').mkdir(parents=True)
                write(dest / 'training_stats.json', stats)
                write(dest / 'train/meta/stats.json', stats)
                (dest / 'train/source_mapping.jsonl').write_text('same training rows\n')
                checksum = hashlib.sha256((dest / 'training_stats.json').read_bytes()).hexdigest()
                write(dest / 'batch_conversion_report.json', dict(status='verified', splits={},
                      normalization_stats={'sha256': checksum}))
            finalize(root, data)
            frozen = (data / 'development/training_stats.json').read_bytes()
            self.assertEqual(frozen, (data / 'independent_test/training_stats.json').read_bytes())
            self.assertEqual(frozen, (data / 'independent_test/train/meta/stats.json').read_bytes())
            self.assertTrue((data / 'independent_test/original_training_stats_serialization.json').exists())
            self.assertEqual(json.loads(frozen), {'state': [1, 2], 'action': [3, 4]})

    def test_independent_test_is_absent_from_development(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            entries, old = fixture(root)
            prepare(root, old)
            dev = json.loads((root / 'development_split.json').read_text())['episodes']
            heldout = json.loads((root / 'independent_test_split.json').read_text())['episodes']
            test_ids = {e['task_id'] for e in entries[6:]}
            self.assertFalse(test_ids & {e['task_id'] for e in dev})
            self.assertEqual(test_ids, {e['task_id'] for e in heldout if e['split'] == 'validation'})
            self.assertEqual({e['task_id'] for e in dev if e['split'] == 'train'},
                             {e['task_id'] for e in heldout if e['split'] == 'train'})

    def test_short_hold_cannot_be_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            entries, _ = fixture(Path(tmp))
            path = Path(entries[0]['episode_dir']) / 'task_events.jsonl'
            path.write_text(path.read_text().replace('50000000000', '49000000000'))
            with self.assertRaisesRegex(ValueError, '不足十'):
                audit(entries[0])

    def test_fragmented_transition_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            entries, _ = fixture(Path(tmp))
            path = Path(entries[0]['episode_dir']) / 'grasp_hold_available_time_grid.jsonl'
            rows = [json.loads(x) for x in path.read_text().splitlines()]
            for row in rows:
                if 140 < row['grid_index'] < 200:
                    row['status'] = 'rejected'
            path.write_text('\n'.join(json.dumps(r) for r in rows))
            with self.assertRaisesRegex(ValueError, '缺少连续'):
                audit(entries[0])


if __name__ == '__main__':
    unittest.main()
