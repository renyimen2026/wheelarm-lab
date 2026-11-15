import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vla_trial_summary import trial_limits, lift_evidence, hold_evidence


class TrialTests(unittest.TestCase):
    def test_explicit_trial_mode_required_for_long_runs(self):
        self.assertEqual(trial_limits(True, None, 60, 3), 3)
        self.assertEqual(trial_limits(True, 90, 60, 3), 60)
        self.assertEqual(trial_limits(True, 600, 80, 3), 80)
        for args in ((True, None, 60, 4), (False, 90, 60, 3), (True, math.nan, 60, 3),
                     (True, math.inf, 60, 3), (True, 601, 60, 3), (True, 90, 101, 3)):
            with self.assertRaises(ValueError):
                trial_limits(*args)

    def cup(self, height=.7, xy=(.66,0.), radius=.025):
        return {'status':'detected', 'top_base_m':height, 'center_base_m':[*xy,height-.05], 'radius_m':radius}

    def test_rgbd_lift_evidence_does_not_claim_full_task_success(self):
        result = lift_evidence(self.cup(), self.cup(.78))
        self.assertTrue(result['lift_verified'])
        self.assertIsNone(result['task_success'])
        self.assertAlmostEqual(result['cup_top_delta_m'], .08)

    def test_stationary_cup_and_mismatched_object_are_not_verified(self):
        for after in (self.cup(), self.cup(.78, xy=(.7,0)), self.cup(.78, radius=.03)):
            self.assertFalse(lift_evidence(self.cup(), after)['lift_verified'])

    def test_missing_measurement_remains_unknown(self):
        for after in (None, {'status':'unavailable'}):
            result=lift_evidence(self.cup(), after)
            self.assertIsNone(result['lift_verified'])
            self.assertIsNone(result['task_success'])

    def test_hold_requires_duration_height_and_closed_stable_gripper(self):
        samples=[dict(elapsed_seconds=t,measurement=self.cup(.78),state=[0,0,0,0,.05])
                 for t in (0,5,10)]
        self.assertTrue(hold_evidence(self.cup(),samples)['task_success'])
        self.assertFalse(hold_evidence(self.cup(),samples[:2])['task_success'])
        for key,value in [('elapsed_seconds',9.),('measurement',self.cup(.76)),
                          ('measurement',{'status':'unavailable'}),('state',[0,0,0,0,.08])]:
            changed=[dict(s) for s in samples]
            changed[-1][key]=value
            self.assertFalse(hold_evidence(self.cup(),changed)['task_success'])
        slowed=[dict(s,joints_stamp_ns=int(s['elapsed_seconds']*.5e9)) for s in samples]
        self.assertFalse(hold_evidence(self.cup(),slowed)['task_success'])


if __name__=='__main__':
    unittest.main(verbosity=2)
