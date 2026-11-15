"""Project delta queries before HF's image-decoding transform.

Local, per-instance workaround for LeRobot #4392 / #2895 on pinned 0.4.4.
Current observations still use the original reader, including RGB and text.
Only requested delta columns are read; ordering, episode clipping and masks
remain LeRobot's responsibility. No installed package or stored data changes.
"""
from types import MethodType

import torch


def projected_query(self, query_indices):
    self._ensure_hf_dataset_loaded()
    result = {}
    for key, indices in query_indices.items():
        if key in self.meta.video_keys:
            continue
        relative = indices if self._absolute_to_relative_idx is None else [
            self._absolute_to_relative_idx[index] for index in indices]
        if key not in self._wheelarm_delta_views:
            self._wheelarm_delta_views[key] = self.hf_dataset.select_columns([key])
        result[key] = torch.stack(self._wheelarm_delta_views[key][relative][key])
    return result


def project_delta_columns(dataset):
    if hasattr(dataset, '_wheelarm_original_delta_query'):
        raise ValueError('Delta projection already installed on this dataset')
    dataset._wheelarm_original_delta_query = dataset._query_hf_dataset
    dataset._wheelarm_delta_views = {}
    dataset._query_hf_dataset = MethodType(projected_query, dataset)
    return dataset
