"""CPU tests of motion selection, physical metrics, and persistent AdamW."""

import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("overfit_probe", Path(__file__).with_name("probe_smolvla_overfit.py"))
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)
torch = probe.torch
from lerobot.optim.optimizers import AdamWConfig


class CountingConfig(AdamWConfig):
    def build(self, params):
        self.builds = getattr(self, "builds", 0)+1
        self.optimizer = super().build(params)
        return self.optimizer


class ToyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.action_out_proj = torch.nn.Linear(1, 1, bias=False)
        self.frozen = torch.nn.Parameter(torch.tensor(1.), requires_grad=False)
        with torch.no_grad():
            self.model.action_out_proj.weight.fill_(1.)

    def forward(self, batch):
        return self.model.action_out_proj(batch).square().mean(), {}


class OverfitTests(unittest.TestCase):
    def test_alternates_two_batches_with_one_optimizer_and_separate_evaluations(self):
        policy, config = ToyPolicy(), CountingConfig(lr=.01)
        first, second = torch.ones(1, 1), torch.full((1, 1), 2.)
        seen = []

        def loss_function(p, batch):
            seen.append(batch.item())
            return p.forward(batch)

        def evaluate():
            with torch.no_grad():
                return {"samples": {label: {
                    "fixed_flow_loss": policy.forward(batch)[0].item(),
                    "physical_errors": {"joint_mae_rad": [0.]*4, "gripper_mae_mm": 0.},
                } for label, batch in (("arm", first), ("gripper-close", second))}}

        report = probe.train_fixed_batch(policy, first, config, 6, 4, evaluate, "cpu",
                                          loss_function=loss_function, alternate_batch=second)
        self.assertEqual(seen, [1., 2., 1., 2., 1., 2.])
        self.assertEqual(config.builds, 1)
        self.assertEqual([r["sample_index"] for r in report["history"]], [0, 1, 0, 1, 0, 1])
        self.assertEqual([r["step"] for r in report["evaluations"]], [0, 4, 6])
        self.assertEqual(set(report["evaluations"][-1]["samples"]), {"arm", "gripper-close"})
        self.assertEqual(policy.frozen.item(), 1.)
        for state in config.optimizer.state.values():
            self.assertEqual(state["step"].item(), 6)

    def test_closing_selection_ignores_large_opening_and_static_arm(self):
        # First window is mostly opening; last complete window closes by 30 mm.
        widths = [.05]*20+[.08]*20+[.08-.0025*(i-40) for i in range(40, 65)]
        mapping = [{"dataset_frame_index": i, "source_grid_row": {
            "status": "accepted", "stamp_ns": i*100_000_000,
            "action_stamp_ns": (i+1)*100_000_000, "action": [0, 0, 0, 0, width],
        }} for i, width in enumerate(widths)]
        index, span = probe.choose_moving_start(mapping, selection="gripper-close")
        self.assertEqual(index, 15)
        self.assertGreater(widths[index]-widths[index+49], .005)
        self.assertGreater(span[4], .005)
        for mode in ("static", "opening"):
            for i, row in enumerate(mapping):
                row["source_grid_row"]["action"][4] = .04 if mode == "static" else .001*i
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, "净闭合"):
                probe.choose_moving_start(mapping, selection="gripper-close")

    def test_selects_moving_complete_window_and_rejects_static_data(self):
        mapping = [{"dataset_frame_index": i, "source_grid_row": {
            "status": "accepted", "stamp_ns": i*100_000_000,
            "action_stamp_ns": (i+1)*100_000_000, "action": [i*i*.001, 0, 0, 0, .04],
        }} for i in range(65)]
        index, span = probe.choose_moving_start(mapping)
        self.assertEqual(index, 15)
        self.assertGreater(span[0], .02)
        self.assertEqual(span[4], 0)
        for row in mapping:
            row["source_grid_row"]["action"] = [0, 0, 0, 0, .04]
        with self.assertRaisesRegex(ValueError, "0.02 rad"):
            probe.choose_moving_start(mapping)

    def test_physical_metrics_keep_radians_and_convert_gripper_to_mm(self):
        target = torch.zeros(1, 50, 5, dtype=torch.float64)
        prediction = target+torch.tensor([.1, .2, .3, .4, .002], dtype=torch.float64)
        result = probe.action_errors(prediction, target)
        for value, expected in zip(result["joint_mae_rad"], [.1, .2, .3, .4]):
            self.assertAlmostEqual(value, expected)
        self.assertAlmostEqual(result["gripper_mae_mm"], 2.)
        self.assertAlmostEqual(result["gripper_max_abs_error_mm"], 2.)

    def test_reuses_optimizer_and_evaluates_initial_intermediate_and_final(self):
        policy, config = ToyPolicy(), CountingConfig(lr=.01)
        batch = torch.ones(1, 1)

        def evaluate():
            with torch.no_grad():
                return {"fixed_flow_loss": policy.forward(batch)[0].item(),
                        "physical_errors": {"joint_mae_rad": [0.]*4, "gripper_mae_mm": 0.}}

        report = probe.train_fixed_batch(policy, batch, config, 5, 2, evaluate, "cpu",
                                          loss_function=lambda p, b: p.forward(b))
        self.assertEqual(config.builds, 1)
        self.assertEqual(report["optimizer_steps"], 5)
        self.assertEqual([r["step"] for r in report["evaluations"]], [0, 2, 4, 5])
        self.assertLess(report["evaluations"][-1]["fixed_flow_loss"], report["evaluations"][0]["fixed_flow_loss"])
        self.assertEqual(policy.frozen.item(), 1.)
        self.assertIsNone(policy.frozen.grad)
        for state in config.optimizer.state.values():
            self.assertEqual(state["step"].item(), 5)


if __name__ == "__main__":
    unittest.main()
