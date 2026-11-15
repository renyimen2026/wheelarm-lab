"""Robot-dimension flow-matching loss adapter for LeRobot 0.4.4 SmolVLA.

Keep the 32-dimensional pretrained architecture, targets/noise construction,
and model forward unchanged. Supervise only the five real action dimensions.
No installed LeRobot files are modified.
"""

import torch


def reduce_robot_loss(losses, action_is_pad):
    """Reduce B x T x 32 elementwise losses over valid steps and real dims."""
    if losses.ndim != 3 or losses.shape[-1] != 32 or not losses.is_floating_point():
        raise ValueError("逐元素 loss 必须是 [B,T,32] 浮点张量")
    if (action_is_pad.dtype != torch.bool or action_is_pad.shape != losses.shape[:2]
            or action_is_pad.device != losses.device):
        raise ValueError("action_is_pad 必须是与 loss 同设备的 [B,T] 布尔张量")
    valid = ~action_is_pad
    if not valid.any():
        raise ValueError("动作块没有有效时间步")
    if not torch.isfinite(losses).all():
        raise ValueError("逐元素 loss 含 NaN/Inf")
    selected = losses[valid]
    loss = selected[:, :5].mean()
    # All diagnostics below use precisely the same raw forward and time mask.
    # These are normalized flow-matching errors, not rad/m action errors.
    details = {
        "robot_5d_loss": loss.detach().item(),
        "all_32d_loss_same_forward": selected.detach().mean().item(),
        "padding_27d_loss_diagnostic_only": selected[:, 5:].detach().mean().item(),
        "robot_5d_loss_per_dimension": selected[:, :5].detach().mean(dim=0).tolist(),
        "valid_action_steps": int(valid.sum().item()),
        "excluded_padded_steps": int(action_is_pad.sum().item()),
        "supervised_action_dimensions": 5,
        "excluded_padding_dimensions": 27,
    }
    return loss, details


def robot_action_loss(policy, batch, noise=None, time=None):
    """Use official preparation/model code, then apply the robot-specific mean.

    Intended only for this project's absolute 5D, non-Aloha SmolVLA adapter.
    The pinned library's policy.forward already reduces over action dimensions,
    so the elementwise flow loss must be obtained from policy.model.forward.
    """
    from lerobot.utils.constants import OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK

    config = policy.config
    if (config.type != "smolvla" or config.max_action_dim != 32
            or tuple(config.action_feature.shape) != (5,)
            or config.adapt_to_pi_aloha or config.use_delta_joint_actions_aloha):
        raise ValueError("此 loss 适配仅支持内部 32 维、外部五维的绝对位置 SmolVLA")
    action = batch["action"]
    if action.ndim != 3 or action.shape[-1] != 5:
        raise ValueError("训练 action 必须是 [B,T,5]")
    mask = batch.get("action_is_pad")
    if mask is None:
        raise ValueError("必须显式提供 action_is_pad，禁止静默忽略时间 padding")
    # Validate the mask before constructing the expensive model graph.
    if mask.dtype != torch.bool or mask.shape != action.shape[:2] or mask.device != action.device:
        raise ValueError("训练 action_is_pad 类型、形状或设备不一致")
    if mask.all():
        raise ValueError("动作块没有有效时间步")
    images, image_masks = policy.prepare_images(batch)
    losses = policy.model.forward(
        images, image_masks, batch[OBS_LANGUAGE_TOKENS], batch[OBS_LANGUAGE_ATTENTION_MASK],
        policy.prepare_state(batch), policy.prepare_action(batch), noise, time,
    )
    if losses.shape[:2] != action.shape[:2]:
        raise ValueError("模型逐元素 loss 的批次或时间维与 action 不一致")
    return reduce_robot_loss(losses, mask)
