"""CPU toy-model tests of the optimizer checks; no SmolVLA weights or GPU."""

import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("train_probe", Path(__file__).with_name("probe_smolvla_train_step.py"))
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)
torch = probe.torch
from lerobot.optim.optimizers import AdamWConfig


class ToyPolicy(torch.nn.Module):
    def __init__(self, detached=False):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.action_out_proj = torch.nn.Linear(2, 1)
        self.backbone = torch.nn.Linear(2, 2)
        self.backbone.requires_grad_(False)
        self.detached = detached
        with torch.no_grad():
            for p in self.parameters():
                p.fill_(.25)

    def forward(self, batch):
        loss = self.model.action_out_proj(self.backbone(batch)).square().mean()
        return (loss.detach() if self.detached else loss), {"loss": loss.item()}


class TrackingConfig(AdamWConfig):
    def build(self, params):
        self.built = super().build(params)
        return self.built


class TrainingStepTests(unittest.TestCase):
    def test_one_real_adamw_step_changes_head_and_preserves_frozen_backbone(self):
        policy, config = ToyPolicy(), TrackingConfig()
        before_head = policy.model.action_out_proj.weight.detach().clone()
        before_frozen = {n: p.detach().clone() for n, p in policy.backbone.named_parameters()}
        report = probe.one_update(policy, torch.ones(1, 2), config, "cpu")
        self.assertEqual(report["optimizer_steps"], 1)
        self.assertGreater(report["gradient_norm_before_clip"], 0)
        self.assertFalse(torch.equal(before_head, policy.model.action_out_proj.weight))
        for name, p in policy.backbone.named_parameters():
            self.assertIsNone(p.grad)
            self.assertTrue(torch.equal(before_frozen[name], p))
        for state in config.built.state.values():
            self.assertEqual(state["step"].item(), 1)

    def test_nan_gradient_stops_before_optimizer_allocates_state(self):
        policy, config = ToyPolicy(), TrackingConfig()
        before = policy.model.action_out_proj.weight.detach().clone()
        policy.model.action_out_proj.weight.register_hook(lambda g: torch.full_like(g, float("nan")))
        with self.assertRaisesRegex(ValueError, "NaN/Inf"):
            probe.one_update(policy, torch.ones(1, 2), config, "cpu")
        self.assertFalse(config.built.state)
        self.assertTrue(torch.equal(before, policy.model.action_out_proj.weight))

    def test_detached_loss_rejected_without_update(self):
        config = TrackingConfig()
        with self.assertRaisesRegex(ValueError, "自动求导"):
            probe.one_update(ToyPolicy(detached=True), torch.ones(1, 2), config, "cpu")
        self.assertFalse(config.built.state)

    def test_disconnected_training_group_rejected_without_update(self):
        policy, config = ToyPolicy(), TrackingConfig()
        policy.model.state_proj = torch.nn.Linear(2, 2)
        with self.assertRaisesRegex(ValueError, "非零梯度"):
            probe.one_update(policy, torch.ones(1, 2), config, "cpu")
        self.assertFalse(config.built.state)


if __name__ == "__main__":
    unittest.main()
