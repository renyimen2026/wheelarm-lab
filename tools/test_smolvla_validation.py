"""CPU tests for phase attribution, physical metrics, and exported plots."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate_smolvla_validation as evaluation


def sample_record():
    np, torch = evaluation.np, evaluation.torch
    target = np.zeros((50, 5))
    prediction = np.ones((50, 5))*.001
    prediction[25:, 4] = .003
    state = np.ones(5)*.001
    physical = evaluation.single.action_errors(torch.tensor(prediction)[None], torch.tensor(target)[None])
    baseline = evaluation.single.action_errors(torch.tensor(state)[None, None].expand(1, 50, 5), torch.tensor(target)[None])
    return {"dataset_index": 0, "source_task_id": "task", "source_stamp_ns": 1_000_000_000,
            "start_phase": "approach", "action_phases": ["close"]*25+["lift"]*25,
            "target_physical": target.tolist(), "prediction_physical": prediction.tolist(),
            "state_physical": state.tolist(), "physical_errors": physical, "baseline": baseline,
            "fixed_flow_loss": .2}


class ValidationTests(unittest.TestCase):
    def test_external_validation_never_uses_external_normalization(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'batch_conversion_report.json').write_text(json.dumps({
                'status': 'verified', 'split_unit': 'original_recording'}))
            train = SimpleNamespace(groups={'train': [0]}, features={'state': 5},
                manifest={'episodes': [{'source_episode': str(root/'train_source')}], 'instruction': 'hold'})
            heldout = SimpleNamespace(groups={'test': [0]}, features={'state': 5},
                manifest={'episodes': [{'source_episode': str(root/'test_source')}], 'instruction': 'hold'})
            # No external training_stats.json is present: this must still work.
            with patch.object(evaluation, 'SplitData', return_value=heldout):
                self.assertIs(evaluation.external_validation(root, train), heldout)
                heldout.groups = {'train': [0]}
                with self.assertRaisesRegex(ValueError, '重叠'):
                    evaluation.external_validation(root, train)
                heldout.groups = {'test': [0]}
                heldout.manifest['episodes'] = train.manifest['episodes']
                with self.assertRaisesRegex(ValueError, '重叠'):
                    evaluation.external_validation(root, train)

    def test_phase_boundaries_use_half_open_intervals(self):
        events = [{"task_id": "t", "event": event, "phase": phase, "stamp_ns": stamp}
                  for event, phase, stamp in (("phase_started", "approach", 10), ("phase_finished", "approach", 20),
                                               ("phase_started", "close", 20), ("phase_finished", "close", 30))]
        phases = evaluation.parse_phases(events, "t")
        self.assertEqual(evaluation.phase_at(19, phases), "approach")
        self.assertEqual(evaluation.phase_at(20, phases), "close")
        self.assertEqual(evaluation.phase_at(30, phases), "before_or_between_phases")

    def test_incomplete_or_cross_task_phase_events_rejected(self):
        for events in ([{"task_id": "t", "event": "phase_started", "phase": "close", "stamp_ns": 1}],
                       [{"task_id": "other", "event": "task_finished"}]):
            with self.subTest(events=events), self.assertRaises(ValueError):
                evaluation.parse_phases(events, "t")

    def test_stage_metrics_attribute_future_steps_not_just_start_phase(self):
        report = evaluation.summarize([sample_record()])
        self.assertEqual(set(report["by_start_phase"]), {"approach"})
        self.assertEqual(set(report["by_action_phase"]), {"close", "lift"})
        self.assertEqual(report["all_actions"]["action_comparisons"], 50)
        self.assertAlmostEqual(report["all_actions"]["gripper_mae_mm"], 2.)
        self.assertAlmostEqual(report["all_actions"]["gripper_p95_abs_error_mm"], 3.)
        self.assertAlmostEqual(report["by_action_phase"]["close"]["gripper_mae_mm"], 1.)
        self.assertAlmostEqual(report["by_action_phase"]["lift"]["gripper_mae_mm"], 3.)
        self.assertAlmostEqual(report["by_action_phase"]["lift"]["hold_gripper_mae_mm"], 1.)

    def test_invalid_trajectory_and_nonfinite_metrics_rejected(self):
        record = sample_record()
        record["prediction_physical"].pop()
        with self.assertRaises(ValueError):
            evaluation.summarize([record])
        with self.assertRaises(ValueError):
            evaluation.physical_summary([[0, 0, 0, 0, float("nan")]], [[0]*5])

    def test_plotting_uses_saved_results_without_a_model(self):
        with tempfile.TemporaryDirectory(prefix="wheelarm-validation-plot-") as directory:
            root = Path(directory)
            record = sample_record()
            (root/"evaluation_report.json").write_text(json.dumps({"status": "completed", "summary": evaluation.summarize([record])}))
            (root/"predictions.jsonl").write_text(json.dumps(record)+"\n")
            result = subprocess.run(["/usr/bin/python3", "-I", str(Path(__file__).with_name("plot_smolvla_validation.py")), str(root)],
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            self.assertEqual(result.returncode, 0, result.stdout)
            for name in ("all_starts_0", "largest_closing_window", "worst_gripper_mae_window"):
                self.assertGreater((root/"plots"/(name+".png")).stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main(verbosity=2)
