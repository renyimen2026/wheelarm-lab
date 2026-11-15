"""Training-only Cartesian/orientation and temporal error metrics for wheelarm.

The policy still predicts five absolute joints. No geometry or stage labels are
added to its inputs, and no trajectory is replaced by an analytic controller.
"""
import torch
from .smolvla_loss import robot_action_loss


def cartesian_error(q, error, lengths=(.32,.28,.23)):
    """Apply the analytic TCP position Jacobian at target joints to joint error."""
    pan=q[...,0]
    angles=q[...,1:4].cumsum(-1)
    links=q.new_tensor(lengths)
    c,s=angles.cos()*links,angles.sin()*links
    radial=c.sum(-1)
    dr=-s.flip(-1).cumsum(-1).flip(-1)
    dz=-c.flip(-1).cumsum(-1).flip(-1)
    radial_error=(dr*error[...,1:4]).sum(-1)
    return torch.stack([-pan.sin()*radial*error[...,0]+pan.cos()*radial_error,
                        pan.cos()*radial*error[...,0]+pan.sin()*radial_error,
                        (dz*error[...,1:4]).sum(-1)],-1)


class GeometryFlowLoss:
    def __init__(self, stats, spec, geometry_weight=.2, smoothness_weight=.1,gripper_weight=0.):
        chain=spec['chain']
        revolute=[link for link in chain if link['kind']=='revolute']
        if [link['axis'] for link in revolute]!=[[0.,0.,1.]]+[[0.,1.,0.]]*3:
            raise ValueError('Geometry metric requires the wheelarm pan/three pitch axes')
        by_name={link['name']:link for link in chain}
        self.lengths=(by_name['elbow_joint']['origin'][0][3],
                      by_name['wrist_pitch_joint']['origin'][0][3],
                      by_name['tool0_joint']['origin'][0][3]+by_name['gripper_tcp_joint']['origin'][0][3])
        self.mean,self.std=stats['action']['mean'],stats['action']['std']
        self.geometry_weight,self.smoothness_weight=geometry_weight,smoothness_weight
        self.gripper_weight=gripper_weight

    def __call__(self,policy,batch,noise=None,time=None):
        action=policy.prepare_action(batch)
        if noise is None: noise=policy.model.sample_noise(action.shape,action.device)
        if time is None: time=policy.model.sample_time(action.shape[0],action.device)
        captured=[]
        handle=policy.model.action_out_proj.register_forward_hook(lambda module,args,value:captured.append(value))
        try:
            base,details=robot_action_loss(policy,batch,noise=noise,time=time)
        finally:
            handle.remove()
        if len(captured)!=1:
            raise ValueError('Expected one official flow velocity output')
        error=captured[0][...,:5].float()-(noise-action)[...,:5].float()
        valid=~batch['action_is_pad']
        std=error.new_tensor(self.std)
        target=batch['action'].float()*std+error.new_tensor(self.mean)
        physical_error=error*std
        position=cartesian_error(target,physical_error,self.lengths)
        pitch=physical_error[...,1:4].sum(-1)
        geometry=(position[valid].square().mean()/.01**2+pitch[valid].square().mean()/.02**2)
        pairs=valid[:,1:]&valid[:,:-1]
        smooth=(error[:,1:]-error[:,:-1])[pairs].square().mean() if pairs.any() else error.sum()*0
        gripper=error[...,4][valid].square().mean()
        total=base+self.geometry_weight*geometry+self.smoothness_weight*smooth+self.gripper_weight*gripper
        details.update(base_flow_loss=base.detach().item(),geometry_loss=geometry.detach().item(),
                       temporal_error_loss=smooth.detach().item(),gripper_flow_loss=gripper.detach().item())
        return total,details
