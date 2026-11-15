"""CPU checks against fused-batch SGD; no model weights, GPU or ROS execution."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_smolvla_batch as pilot

torch = pilot.torch
torch.set_num_threads(1)


class ToyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.))
        self.frozen = torch.nn.Parameter(torch.tensor(1.), requires_grad=False)

    def forward(self, x):
        return self.weight*(x*self.frozen)


class CountingSGD(torch.optim.SGD):
    def __init__(self, parameters):
        super().__init__(parameters, lr=.01)
        self.steps = 0
        self.gradients = []

    def step(self, closure=None):
        self.gradients.append([p.grad.detach().clone() for group in self.param_groups for p in group['params']])
        self.steps += 1
        return super().step(closure)


class Config:
    grad_clip_norm = 100.

    def build(self, parameters):
        self.optimizer = CountingSGD(parameters)
        return self.optimizer


def loss_function(model, batch):
    return (model(batch['x'])-batch['target']).square().mean(), {}


def evaluation(model, calls):
    def evaluate(step):
        assert not model.training and not torch.is_grad_enabled()
        calls.append(step)
        torch.rand(100)
        summary = dict(evaluated_starts=1, fixed_flow_loss=model.weight.item()**2,
                       joint_mae_rad=[0.]*4, gripper_mae_mm=0.)
        return {'splits': {'validation': {'summary': summary}}}
    return evaluate


class AccumulationTests(unittest.TestCase):
    def test_proprioception_augmentation_is_bounded_reproducible_and_does_not_mutate_state(self):
        state=torch.tensor([[[0.,-1.,2.,-1.,.08]]]).expand(10000,1,5).clone()
        original=state.clone()
        torch.manual_seed(9)
        first=pilot.augment_proprioception(state,.01)
        torch.manual_seed(9)
        second=pilot.augment_proprioception(state,.01)
        self.assertTrue(torch.equal(first,second))
        self.assertTrue(torch.equal(state,original))
        self.assertAlmostEqual((first-state)[:,:,1].std().item(),.01,delta=.0004)
        self.assertTrue((first[...,4]<=.08).all())
        self.assertTrue((first[...,4]>=0.).all())
        self.assertTrue(torch.equal(pilot.augment_proprioception(state,0.),state))

    def test_matches_true_four_sample_gradient_and_one_sgd_step(self):
        model, reference, config = ToyPolicy(), ToyPolicy(), Config()
        x, target = torch.tensor([1., 2., 3., 4.]), torch.tensor([0., 1., -1., 2.])
        reference_loss, _ = loss_function(reference, dict(x=x, target=target))
        reference_loss.backward()
        expected_gradient = reference.weight.grad.clone()
        torch.optim.SGD([reference.weight], lr=.01).step()
        position, draws, logs, evaluations = 0, [], [], []

        def next_batch():
            nonlocal position
            i = position
            position += 1
            draws.append(model.weight.item())
            return ('a' if i % 2 == 0 else 'b'), i, dict(x=x[i:i+1], target=target[i:i+1])

        result = pilot.run_updates(model, next_batch, evaluation(model, evaluations), config,
            1, 1, 'cpu', loss_function=loss_function, on_update=logs.append,
            gradient_accumulation_steps=4)
        torch.testing.assert_close(model.weight, reference.weight)
        torch.testing.assert_close(config.optimizer.gradients[0][0], expected_gradient)
        self.assertEqual(draws, [1.]*4)
        self.assertEqual(config.optimizer.steps, 1)
        self.assertEqual(evaluations, [0, 1])
        self.assertEqual(result['sample_draws_per_original_recording'], {'a': 2, 'b': 2})
        self.assertEqual(result['updates_per_original_recording'], {'a': 1, 'b': 1})
        self.assertEqual(result['total_sample_draws'], 4)
        self.assertEqual(logs[0]['sample_draws_so_far'], 4)
        self.assertEqual([s['dataset_index'] for s in logs[0]['samples']], [0, 1, 2, 3])
        self.assertAlmostEqual(logs[0]['random_training_loss'], reference_loss.item())
        self.assertNotIn('dataset_index', logs[0])
        self.assertEqual(model.frozen.item(), 1.)
        self.assertIsNone(model.frozen.grad)

    def test_two_updates_reset_gradients_and_preserve_training_rng(self):
        model, reference, config = ToyPolicy(), ToyPolicy(), Config()
        x, target = torch.arange(1., 9.), torch.zeros(8)
        # Compare TWO steps: this catches failure to clear the previous gradient.
        optimizer = torch.optim.SGD([reference.weight], lr=.01)
        for start in [0, 4]:
            optimizer.zero_grad(set_to_none=True)
            loss, _ = loss_function(reference, dict(x=x[start:start+4], target=target[start:start+4]))
            loss.backward()
            torch.nn.utils.clip_grad_norm_([reference.weight], 100.)
            optimizer.step()
        position, noises, logs, evaluations = 0, [], [], []

        def next_batch():
            nonlocal position
            i = position
            position += 1
            return 'train', i, dict(x=x[i:i+1], target=target[i:i+1])

        def random_loss(model, batch):
            noises.append(torch.rand(1).item())
            return loss_function(model, batch)

        torch.manual_seed(19)
        expected = torch.rand(8).tolist()
        torch.manual_seed(19)
        result = pilot.run_updates(model, next_batch, evaluation(model, evaluations), config,
            2, 1, 'cpu', loss_function=random_loss, on_update=logs.append, gradient_accumulation_steps=4)
        torch.testing.assert_close(model.weight, reference.weight)
        self.assertEqual(noises, expected)
        self.assertEqual(config.optimizer.steps, 2)
        self.assertEqual(result['total_sample_draws'], 8)
        self.assertEqual([r['sample_draws_so_far'] for r in logs], [4, 8])
        self.assertEqual(evaluations, [0, 1, 2])

    def test_nonfinite_third_microbatch_never_updates_parameters(self):
        model, config, draw = ToyPolicy(), Config(), 0

        def next_batch():
            nonlocal draw
            draw += 1
            if draw == 3:
                model.weight.register_hook(lambda g: torch.full_like(g, float('nan')))
            return 'train', draw, dict(x=torch.ones(1), target=torch.zeros(1))

        with self.assertRaisesRegex(ValueError, 'NaN/Inf'):
            pilot.run_updates(model, next_batch, evaluation(model, []), config, 1, 1, 'cpu',
                loss_function=loss_function, gradient_accumulation_steps=4)
        self.assertEqual(draw, 3)
        self.assertEqual(config.optimizer.steps, 0)
        self.assertEqual(model.weight.item(), 1.)

    def test_invalid_accumulation_rejected_before_optimizer_creation(self):
        for value in [0, -1, 17, True, 1.5]:
            config = Config()
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, '梯度累积'):
                pilot.run_updates(ToyPolicy(), None, None, config, 1, 1, 'cpu',
                                  gradient_accumulation_steps=value)
            self.assertFalse(hasattr(config, 'optimizer'))


if __name__ == '__main__':
    unittest.main(verbosity=2)
