"""Reference-based window selection and exact mixed sampling; no GPU or weights."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'wheel_arm_manipulation'))
from wheel_arm_manipulation.smolvla_batch_data import TransitionMixedStarts, training_transition_groups
from wheel_arm_manipulation.smolvla_coordination import coordination_plan


class HeightFK:
    def positions(self, values):
        values = np.asarray(values)
        return np.stack([values[..., 0]*0, values[..., 0]*0, .599+values[..., 0]], axis=-1)


class TransitionTests(unittest.TestCase):
    def test_exact_two_plus_two_and_uniform_recordings_in_both_branches(self):
        full = {'a': list(range(100)), 'b': [100, 101, 102]}
        targeted = {'a': [2, 3], 'b': [101]}
        sampler, second = TransitionMixedStarts(full, targeted, 7), TransitionMixedStarts(full, targeted, 7)
        counts = Counter()
        ordinary_nontransition = 0
        for _ in range(2000):
            batch = []
            for _ in range(4):
                task, index = sampler.draw()
                self.assertEqual((task, index), second.draw())
                self.assertEqual(sampler.last_pool, second.last_pool)
                pool = sampler.last_pool
                batch.append(pool)
                counts[pool, task] += 1
                self.assertIn(index, targeted[task] if pool == 'transition' else full[task])
                ordinary_nontransition += pool == 'all' and index not in targeted[task]
            self.assertEqual(Counter(batch), {'all': 2, 'transition': 2})
        for pool in ['all', 'transition']:
            self.assertEqual(sum(counts[pool, task] for task in full), 4000)
            self.assertTrue(all(1800 < counts[pool, task] < 2200 for task in full))
        self.assertGreater(ordinary_nontransition, 2000)

    def test_target_pool_must_be_a_nonempty_subset_of_every_training_recording(self):
        for full, targeted in [({}, {}), ({'a': [0]}, {}), ({'a': []}, {'a': [0]}),
                ({'a': [0]}, {'a': []}), ({'a': [0]}, {'a': [1]}),
                ({'a': [0]}, {'a': [0, 0]}), ({'a': [0]}, {'a': [0], 'test': [1]})]:
            with self.subTest(full=full, targeted=targeted), self.assertRaises(ValueError):
                TransitionMixedStarts(full, targeted)

    def test_validation_cannot_be_used_to_select_training_windows(self):
        with self.assertRaisesRegex(ValueError, '仅允许训练集'):
            training_transition_groups(None, SimpleNamespace(split='validation'), None)

    def test_all_eligible_windows_are_used_and_events_remain_pinned(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root/'source'
            source.mkdir()
            events = []
            for phase, begin, end in [('close', 0, 2_000_000_000), ('lift', 2_000_000_000, 7_000_000_000)]:
                events.extend([dict(task_id='t', phase=phase, event='phase_started', stamp_ns=begin),
                               dict(task_id='t', phase=phase, event='phase_finished', stamp_ns=end)])
            path = source/'task_events.jsonl'
            path.write_text('\n'.join(json.dumps(e) for e in events))
            (root/'source_manifest.json').write_text(json.dumps({'sources': [dict(split='train', task_id='t',
                source_episode=str(source), source_file_sha256={'task_events.jsonl': hashlib.sha256(path.read_bytes()).hexdigest()})]}))
            mapping = []
            for i in range(70):
                state = [0. if i < 20 else .08, 0, 0, 0, .08 if i < 8 else .049]
                action = [0. if i+1 < 20 else .08, 0, 0, 0, .08 if i+1 < 8 else .049]
                mapping.append(dict(source_task_id='t', source_grid_row=dict(stamp_ns=i*100_000_000,
                    action_stamp_ns=(i+1)*100_000_000, state=state, action=action)))
            data = SimpleNamespace(split='train', groups={'t': list(range(21))}, mapping=mapping)
            groups, plan = training_transition_groups(root, data, HeightFK())
            self.assertEqual(groups['t'], list(range(20)))
            self.assertEqual(len(plan['t']['indices']), 2)
            self.assertTrue(set(plan['t']['indices']) < set(groups['t']))
            self.assertEqual(plan['t']['eligible_indices'], groups['t'])
            path.write_text(path.read_text()+'\n')
            with self.assertRaisesRegex(ValueError, 'changed'):
                training_transition_groups(root, data, HeightFK())

    def test_missing_transition_does_not_silently_remove_a_training_recording(self):
        data = SimpleNamespace(split='train', groups={'a': [0], 'b': [1]})
        with patch('wheel_arm_manipulation.smolvla_coordination.coordination_plan',
                   return_value={'a': {'eligible_indices': [0]}, 'b': {'eligible_indices': []}}):
            with self.assertRaisesRegex(ValueError, '每条训练示范'):
                training_transition_groups(None, data, None)


if __name__ == '__main__':
    unittest.main(verbosity=2)
