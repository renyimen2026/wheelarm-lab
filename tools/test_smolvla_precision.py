"""Precision regressions: a real safetensors reload and sub-BF16 updates."""
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch
from safetensors.torch import save_file
from smolvla_precision import restore_trainable_precision, trainable_precision


class PrecisionTests(unittest.TestCase):
    def test_saved_fp32_is_not_rounded_by_bf16_constructor_on_reload(self):
        model = torch.nn.Linear(2, 1, bias=False).to(torch.bfloat16)
        target = torch.tensor([[.1001234, -.2008765]])
        with tempfile.TemporaryDirectory() as directory:
            save_file({'weight': target}, str(Path(directory)/'model.safetensors'))
            model.load_state_dict({'weight': target}, strict=True)
            self.assertFalse(torch.equal(model.weight.float(), target))
            self.assertEqual(restore_trainable_precision(model, directory), ['weight'])
        self.assertEqual(model.weight.dtype, torch.float32)
        self.assertTrue(torch.equal(model.weight, target))

    def test_fp32_optimizer_keeps_small_updates_and_frozen_backbone(self):
        model = torch.nn.Linear(2, 1).to(torch.bfloat16)
        model.bias.requires_grad_(False)
        frozen = model.bias.detach().clone()
        with torch.no_grad():
            model.weight.fill_(.125)
        before = model.weight.float().detach().clone()
        report = trainable_precision(model)
        self.assertEqual(report['optimizer_parameter_counts'], {'torch.float32': 2})
        optimizer = torch.optim.AdamW([model.weight], lr=1e-5, weight_decay=0)
        model.weight.grad = torch.ones_like(model.weight)
        optimizer.step()
        self.assertTrue((model.weight != before).all())
        self.assertTrue(torch.equal(model.weight.to(torch.bfloat16), before.to(torch.bfloat16)))
        self.assertEqual(model.bias.dtype, torch.bfloat16)
        self.assertTrue(torch.equal(model.bias, frozen))


if __name__ == '__main__':
    unittest.main()
