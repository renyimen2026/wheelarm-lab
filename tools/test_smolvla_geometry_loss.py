import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'wheel_arm_manipulation'))
import torch
from wheel_arm_manipulation import smolvla_geometry_loss as geometry
from wheel_arm_manipulation.smolvla_loss import reduce_robot_loss


class GeometryTests(unittest.TestCase):
    def test_jacobian_matches_finite_difference_positions(self):
        torch.manual_seed(4)
        q=torch.randn(2,7,5,dtype=torch.float64)
        error=torch.randn_like(q)
        def positions(q):
            a=q[...,1:4].cumsum(-1)
            l=q.new_tensor([.32,.28,.23])
            r=(a.cos()*l).sum(-1)
            return torch.stack([r*q[...,0].cos(),r*q[...,0].sin(),.57-(a.sin()*l).sum(-1)],-1)
        eps=1e-6
        numerical=(positions(q+eps*error)-positions(q-eps*error))/(2*eps)
        self.assertTrue(torch.allclose(geometry.cartesian_error(q,error),numerical,atol=1e-8))

    def test_loss_preserves_mask_and_removes_hook_and_backpropagates(self):
        spec={'chain':[{'name':name,'kind':'revolute','axis':axis,'origin':[[1,0,0,x]]}
                     for name,axis,x in [('shoulder_pan_joint',[0.,0.,1.],0),
                      ('shoulder_lift_joint',[0.,1.,0.],0),('elbow_joint',[0.,1.,0.],.32),
                      ('wrist_pitch_joint',[0.,1.,0.],.28)]]+
                     [{'name':name,'kind':'fixed','origin':[[1,0,0,x]]}
                      for name,x in [('tool0_joint',.11),('gripper_tcp_joint',.12)]]}
        action=torch.zeros(1,4,5)
        padded=torch.zeros(1,4,32)
        errors=torch.nn.Parameter(torch.ones_like(padded))
        project=torch.nn.Identity()
        policy=SimpleNamespace(prepare_action=lambda b:padded,
            model=SimpleNamespace(action_out_proj=project))
        batch={'action':action,'action_is_pad':torch.tensor([[False,False,True,True]])}
        def flow(policy,batch,noise,time):
            output=project(errors)
            return reduce_robot_loss(output.square(),batch['action_is_pad'])
        objective=geometry.GeometryFlowLoss({'action':{'mean':[0]*5,'std':[.001,.07,.19,.16,.015]}},spec)
        with patch.object(geometry,'robot_action_loss',side_effect=flow):
            loss,details=objective(policy,batch,noise=padded,time=torch.ones(1))
            objective.gripper_weight=.5
            weighted,_=objective(policy,batch,noise=padded,time=torch.ones(1))
            self.assertAlmostEqual((weighted-loss).item(),.5,places=5)
        self.assertGreater(loss.item(),1.)
        self.assertGreater(details['geometry_loss'],0)
        self.assertEqual(len(project._forward_hooks),0)
        loss.backward()
        self.assertTrue(torch.isfinite(errors.grad).all())
        self.assertFalse(errors.grad[:,2:].any())
        self.assertFalse(errors.grad[... ,5:].any())
        self.assertTrue(errors.grad[:,:2,:5].any())


if __name__=='__main__':unittest.main(verbosity=2)
