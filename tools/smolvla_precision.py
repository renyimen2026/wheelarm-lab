"""Preserve saved expert precision and keep optimizer-owned weights in fp32.

BF16 autocast is independent of parameter storage. LeRobot 0.4.4 constructs
some expert layers in BF16; a normal copy-based load would round an FP32
checkpoint back to that construction dtype before training or inference.
"""
from collections import Counter
from pathlib import Path

import torch
from safetensors import safe_open


def restore_trainable_precision(policy, checkpoint):
    """After a strict ordinary load, restore trainable FP32 tensors exactly."""
    restored = []
    with safe_open(str(Path(checkpoint)/'model.safetensors'), framework='pt', device='cpu') as saved:
        for name, parameter in policy.named_parameters():
            if not parameter.requires_grad:
                continue
            if saved.get_slice(name).get_dtype() == 'F32' and parameter.dtype != torch.float32:
                value = saved.get_tensor(name)
                if value.shape != parameter.shape:
                    raise ValueError(f'Checkpoint parameter shape changed: {name}')
                parameter.data = value.to(device=parameter.device)
                restored.append(name)
    return restored


def trainable_precision(policy, mode='fp32'):
    if mode not in ('fp32', 'checkpoint'):
        raise ValueError('Trainable precision must be fp32 or checkpoint')
    before, after = Counter(), Counter()
    for parameter in policy.parameters():
        if parameter.requires_grad:
            before[str(parameter.dtype)] += parameter.numel()
            if mode == 'fp32':
                parameter.data = parameter.data.float()
            after[str(parameter.dtype)] += parameter.numel()
    return {'mode': mode, 'before_parameter_counts': dict(before),
            'optimizer_parameter_counts': dict(after), 'forward_autocast': 'bfloat16'}
