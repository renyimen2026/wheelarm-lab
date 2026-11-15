"""Probe guard tests; no model construction, weights, or GPU operation."""

import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("forward_probe", Path(__file__).with_name("probe_smolvla_forward.py"))
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def mapping(count=52):
    return [{"dataset_frame_index": i, "source_grid_row": {
        "status": "accepted", "stamp_ns": 1_000_000_000+i*100_000_000,
        "action_stamp_ns": 1_100_000_000+i*100_000_000,
        "action": [i*.01, -.2, .4, -.2, .04],
    }} for i in range(count)]


class ForwardProbeTests(unittest.TestCase):
    def test_full_window_preserves_future_action_order(self):
        actions = probe.validate_window(mapping(), 2, 50)
        self.assertEqual(tuple(actions.shape), (50, 5))
        self.assertAlmostEqual(actions[0, 0].item(), .02)
        self.assertAlmostEqual(actions[-1, 0].item(), .51)

    def test_negative_and_incomplete_start_rejected(self):
        for index in (-1, 3):
            with self.subTest(index=index), self.assertRaises(ValueError):
                probe.validate_window(mapping(), index, 50)

    def test_discontinuity_and_wrong_action_offset_rejected(self):
        for field in ("stamp_ns", "action_stamp_ns"):
            rows = mapping()
            rows[20]["source_grid_row"][field] += 100_000_000
            with self.subTest(field=field), self.assertRaises(ValueError):
                probe.validate_window(rows, 0, 50)
        rows = mapping()
        rows[20]["source_grid_row"]["status"] = "rejected"
        with self.assertRaises(ValueError):
            probe.validate_window(rows, 0, 50)

    def test_invalid_prediction_or_loss_rejected(self):
        torch = probe.torch
        probe.require_tensor(torch.tensor(1.), (), "loss")
        for tensor, shape in ((torch.tensor(float("nan")), ()),
                              (torch.zeros(1, 4, 5), (1, 50, 5)),
                              (torch.full((1, 50, 5), float("inf")), (1, 50, 5))):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                probe.require_tensor(tensor, shape, "prediction")


if __name__ == "__main__":
    unittest.main()
