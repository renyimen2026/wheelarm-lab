"""CPU tests of real-dimension loss values, masking, and gradient routing."""

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"wheel_arm_manipulation"))
import torch
from wheel_arm_manipulation.smolvla_loss import reduce_robot_loss, robot_action_loss
from lerobot.utils.constants import OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK


class RobotLossTests(unittest.TestCase):
    def test_mean_excludes_extra_dimensions_and_padded_steps(self):
        raw = torch.full((2, 3, 32), 100.)
        raw[:, :, :5] = torch.arange(1., 6.)
        mask = torch.tensor([[False, False, True], [False, True, True]])
        raw[mask] = 999.
        loss, details = reduce_robot_loss(raw, mask)
        self.assertEqual(loss.item(), 3.)
        self.assertEqual(details["valid_action_steps"], 3)
        self.assertEqual(details["excluded_padded_steps"], 3)
        self.assertEqual(details["robot_5d_loss_per_dimension"], [1., 2., 3., 4., 5.])
        self.assertEqual(details["padding_27d_loss_diagnostic_only"], 100.)
        self.assertAlmostEqual(details["all_32d_loss_same_forward"], (5*3.+27*100.)/32)

    def test_gradient_only_reaches_valid_real_action_elements(self):
        raw = torch.ones(2, 3, 32, dtype=torch.float64, requires_grad=True)
        mask = torch.tensor([[False, False, True], [False, True, True]])
        reduce_robot_loss(raw, mask)[0].backward()
        expected = torch.zeros_like(raw)
        for b, t in ((0, 0), (0, 1), (1, 0)):
            expected[b, t, :5] = 1/15
        self.assertTrue(torch.equal(raw.grad, expected))

    def test_invalid_mask_empty_window_and_nonfinite_loss_rejected(self):
        raw = torch.ones(1, 2, 32)
        for mask in (torch.ones(1, 2, dtype=torch.bool), torch.zeros(1, 2),
                     torch.zeros(2, 1, dtype=torch.bool)):
            with self.subTest(mask=mask), self.assertRaises(ValueError):
                reduce_robot_loss(raw, mask)
        raw[0, 0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "NaN/Inf"):
            reduce_robot_loss(raw, torch.zeros(1, 2, dtype=torch.bool))

    def test_adapter_preserves_official_preparation_noise_and_time(self):
        raw = torch.ones(1, 2, 32, requires_grad=True)
        core = Mock()
        core.forward.return_value = raw
        images, image_masks = [object()], [object()]
        state, padded_action = object(), object()
        policy = SimpleNamespace(
            config=SimpleNamespace(type="smolvla", max_action_dim=32,
                                   action_feature=SimpleNamespace(shape=(5,)),
                                   adapt_to_pi_aloha=False, use_delta_joint_actions_aloha=False),
            prepare_images=Mock(return_value=(images, image_masks)),
            prepare_state=Mock(return_value=state), prepare_action=Mock(return_value=padded_action),
            model=core,
        )
        batch = {"action": torch.ones(1, 2, 5), "action_is_pad": torch.zeros(1, 2, dtype=torch.bool),
                 OBS_LANGUAGE_TOKENS: object(), OBS_LANGUAGE_ATTENTION_MASK: object()}
        noise, time = object(), object()
        loss, _ = robot_action_loss(policy, batch, noise, time)
        core.forward.assert_called_once_with(images, image_masks, batch[OBS_LANGUAGE_TOKENS],
                                             batch[OBS_LANGUAGE_ATTENTION_MASK], state, padded_action,
                                             noise, time)
        loss.backward()
        self.assertFalse(raw.grad[:, :, 5:].any())
        # No mask must fail before invoking the model again.
        del batch["action_is_pad"]
        with self.assertRaisesRegex(ValueError, "显式提供"):
            robot_action_loss(policy, batch)
        self.assertEqual(core.forward.call_count, 1)


if __name__ == "__main__":
    unittest.main()
