"""Local multi-episode loader with explicit full-window and split checks."""

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import default_collate

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from wheel_arm_manipulation.action_alignment import STATE_NAMES, STATE_UNITS, read_jsonl
from wheel_arm_manipulation.lerobot_conversion import IMAGE_KEY, STATE_KEY, digest
from wheel_arm_manipulation.smolvla_interface import configure, validate_stats

HORIZON = 50


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


class SplitData:
    def __init__(self, root, split):
        self.root, self.split = Path(root).resolve(strict=True), split
        self.manifest = read_json(self.root/"conversion_manifest.json")
        report = read_json(self.root/"conversion_report.json")
        m = self.manifest
        if (report.get("status") != "verified" or m.get("split") != split
                or m.get("action_semantics") != "absolute_reference_at_grid_plus_0.1s"
                or m.get("names") != list(STATE_NAMES) or m.get("units") != list(STATE_UNITS)):
            raise ValueError("需要已通过验证、动作语义和集合标记一致的数据")
        self.mapping = read_jsonl(self.root/"source_mapping.jsonl")
        self.base = LeRobotDataset(repo_id=m["repo_id"], root=self.root, download_videos=False)
        if (len(self.base) != m["rows"] or len(self.mapping) != m["rows"] or self.base.fps != 10
                or self.base.num_episodes != len(m["episodes"])):
            raise ValueError("多片段数据集大小、频率或来源行数不一致")
        self.features = configure(self.base.meta)[1]
        self.groups, starts, cursor = {}, [], 0
        for ep_index, ep in enumerate(m["episodes"]):
            begin, end = ep["dataset_from_index"], ep["dataset_to_index"]
            if begin != cursor or end-begin != ep["rows"] or ep["dataset_episode_index"] != ep_index:
                raise ValueError("片段边界或编号不一致")
            for index in range(begin, end):
                row = self.mapping[index]
                grid = row["source_grid_row"]
                if (row["dataset_index"] != index or row["dataset_episode_index"] != ep_index
                        or row["dataset_frame_index"] != index-begin or row["split"] != split
                        or row["source_task_id"] != ep["source_task_id"]
                        or row["source_episode"] != ep["source_episode"]
                        or grid["status"] != "accepted" or grid["instruction"] != m["instruction"]
                        or grid["stamp_ns"] != ep["first_source_stamp_ns"]+(index-begin)*100_000_000
                        or grid["grid_index"] != ep["first_source_grid_index"]+index-begin
                        or grid["action_stamp_ns"] != grid["stamp_ns"]+100_000_000):
                    raise ValueError("来源映射的集合、片段边界、任务或物理时间不一致")
            candidates = list(range(begin, max(begin, end-HORIZON+1)))
            starts.extend(candidates)
            self.groups.setdefault(ep["source_task_id"], []).extend(candidates)
            cursor = end
        if cursor != len(self.base) or starts != m["full_chunk_start_indices"][str(HORIZON)]:
            raise ValueError("完整 50 步起点与片段边界不一致，禁止尾部 padding 起点")
        if not starts or any(not indices for indices in self.groups.values()):
            raise ValueError("每条原始示范都需要至少一个完整 50 步起点")
        self.starts, self.start_set = starts, set(starts)
        self.chunks = LeRobotDataset(repo_id=m["repo_id"], root=self.root, download_videos=False,
                                    delta_timestamps={"action": [i/10 for i in range(HORIZON)]})
        self.chunks._ensure_hf_dataset_loaded()
        self.chunks.hf_dataset = self.chunks.hf_dataset.select_columns([
            key for key in self.chunks.hf_dataset.column_names if key != IMAGE_KEY])

    def batch(self, index):
        if index not in self.start_set:
            raise ValueError("只允许取片段内部的完整 50 步起点")
        item = self.base[index]
        chunk = self.chunks[index]
        expected = torch.tensor([r["source_grid_row"]["action"]
                                 for r in self.mapping[index:index+HORIZON]], dtype=torch.float32)
        state = torch.tensor(self.mapping[index]["source_grid_row"]["state"], dtype=torch.float32)
        if (chunk["action_is_pad"].any() or not torch.equal(chunk["action"], expected)
                or not torch.equal(item[STATE_KEY], state) or item["task"] != self.manifest["instruction"]):
            raise ValueError("实际加载的状态、动作、任务或 padding 与来源不一致")
        item["action"] = chunk["action"]
        item["action_is_pad"] = chunk["action_is_pad"]
        item[IMAGE_KEY] = item[IMAGE_KEY].unsqueeze(0)
        item[STATE_KEY] = item[STATE_KEY].unsqueeze(0)
        item[IMAGE_KEY+"_is_pad"] = torch.zeros(1, dtype=torch.bool)
        item[STATE_KEY+"_is_pad"] = torch.zeros(1, dtype=torch.bool)
        if not all(torch.isfinite(item[key]).all() for key in (IMAGE_KEY, STATE_KEY, "action")):
            raise ValueError("训练样本含非有限值")
        return default_collate([item])

    def evaluation_indices(self,horizon=HORIZON):
        # Per original recording: six time-spread anchors and the two largest
        # arm/closing motions. Selection is fixed before any model is loaded.
        if horizon not in (10,50):
            raise ValueError('评估时域只能为10或50步')
        chosen = []
        for indices in self.groups.values():
            anchors = {indices[round(i*(len(indices)-1)/5)] for i in range(6)}
            arm_scores, closing_scores = {}, {}
            for index in indices:
                actions = np.asarray([r["source_grid_row"]["action"]
                                      for r in self.mapping[index:index+horizon]])
                arm_scores[index] = float(np.square(np.ptp(actions[:, :4], axis=0)).sum())
                closing_scores[index] = float(actions[0, 4]-actions[-1, 4])
            anchors.add(max(indices, key=arm_scores.__getitem__))
            anchors.add(max(indices, key=closing_scores.__getitem__))
            chosen.extend(sorted(anchors))
        return chosen


def load_batch(root):
    root = Path(root).expanduser().resolve(strict=True)
    report = read_json(root/"batch_conversion_report.json")
    normalization = report["normalization_stats"]
    if (report.get("status") != "verified" or report.get("split_unit") != "original_recording"
            or normalization.get("validation_used") is not False
            or normalization.get("file") != "training_stats.json"
            or normalization.get("source") != "train/meta/stats.json"
            or digest(root/"training_stats.json") != normalization["sha256"]
            or digest(root/"train/meta/stats.json") != normalization["sha256"]):
        raise ValueError("训练集归一化统计未冻结、已变化或混入验证集")
    train, validation = SplitData(root/"train", "train"), SplitData(root/"validation", "validation")
    train_sources = {str(Path(ep["source_episode"]).resolve()) for ep in train.manifest["episodes"]}
    validation_sources = {str(Path(ep["source_episode"]).resolve()) for ep in validation.manifest["episodes"]}
    if set(train.groups) & set(validation.groups) or train_sources & validation_sources:
        raise ValueError("训练和验证共享原始示范，发生数据泄漏")
    if train.features != validation.features or train.manifest["instruction"] != validation.manifest["instruction"]:
        raise ValueError("训练、验证的策略特征或指令不一致")
    stats = read_json(root/"training_stats.json")
    validate_stats(stats, train.features)
    return train, validation, stats


class BalancedStarts:
    """Uniform original-task choice, then uniform eligible start within it."""
    def __init__(self, groups, seed=0):
        self.groups = groups
        self.tasks = sorted(groups)
        if not self.tasks or any(not groups[t] for t in self.tasks):
            raise ValueError("均衡抽样需要非空训练示范组")
        self.rng = random.Random(seed)

    def draw(self):
        task = self.rng.choice(self.tasks)
        return task, self.rng.choice(self.groups[task])


def active_training_groups(data,labels,horizon):
    """Exclude pre-action waits as training starts; preserve lift/terminal holds.

    Original timestamps/chunks stay intact. Phase labels only select samples,
    and are never supplied to the policy. Validation is not filtered.
    """
    if data.split!='train' or horizon not in (10,50):
        raise ValueError('只筛选训练集的10/50步起点')
    selected={}
    for task,indices in data.groups.items():
        selected[task]=[]
        for index in indices:
            target=np.array([r['source_grid_row']['action'] for r in data.mapping[index:index+horizon]])
            state=np.array(data.mapping[index]['source_grid_row']['state'])
            span=np.ptp(target,axis=0)
            delta=abs(target[-1]-state)
            if (labels[index]=='lift' or max(span[:4])>.003 or span[4]>.001
                    or max(delta[:4])>.006 or delta[4]>.002):
                selected[task].append(index)
        if not selected[task]:
            raise ValueError('每条训练示范必须保留动作或末尾持杯样本')
    return selected


def training_phase_groups(root, data):
    """Verify pinned training events, then label each complete window by start time."""
    from wheel_arm_manipulation.task_phases import parse_phases, phase_at
    if data.split != "train":
        raise ValueError("阶段抽样仅允许训练集")
    intervals, hashes = {}, {}
    for source in read_json(Path(root)/"source_manifest.json")["sources"]:
        if source["split"] != "train":
            continue
        task = source["task_id"]
        path = Path(source["source_episode"])/"task_events.jsonl"
        checksum = digest(path)
        if task in intervals or checksum != source["source_file_sha256"]["task_events.jsonl"]:
            raise ValueError("原始训练任务阶段文件重复或已变化")
        intervals[task] = parse_phases(read_jsonl(path), task)
        hashes[task] = checksum
    if set(intervals) != set(data.groups):
        raise ValueError("训练示范的阶段信息不完整")
    groups, labels = {}, {}
    for task, indices in data.groups.items():
        groups[task] = {}
        for index in indices:
            phase = phase_at(data.mapping[index]["source_grid_row"]["stamp_ns"], intervals[task])
            groups[task].setdefault(phase, []).append(index)
            labels[index] = phase
    return groups, labels, hashes


class PhaseBalancedStarts:
    """Uniform task, then available start-phase bucket, then complete start.

    Between-phase starts remain an explicit bucket; no starts are removed.
    A window may cross future phase boundaries: its bucket is its start phase.
    """
    def __init__(self, groups, seed=0):
        if (not groups or any(not phases or any(not indices for indices in phases.values())
                              for phases in groups.values())):
            raise ValueError("阶段均衡抽样需要非空示范和阶段组")
        self.groups = groups
        self.tasks = sorted(groups)
        self.phases = {task: sorted(groups[task]) for task in self.tasks}
        self.rng = random.Random(seed)

    def draw(self):
        task = self.rng.choice(self.tasks)
        phase = self.rng.choice(self.phases[task])
        return task, self.rng.choice(self.groups[task][phase])


def training_transition_groups(root, data, fk):
    """Reference-only full closure-to-lift windows from TRAIN recordings.

    Keep all eligible indices, rather than the two fixed diagnostic anchors.
    Selection uses pinned phase events, real state and recorded references;
    it never inspects a model prediction or any validation recording.
    """
    from wheel_arm_manipulation.smolvla_coordination import coordination_plan
    if data.split != "train":
        raise ValueError("过渡窗口抽样仅允许训练集")
    plan = coordination_plan(root, data, fk)
    groups = {task: details['eligible_indices'] for task, details in plan.items()}
    if set(groups) != set(data.groups) or any(not group for group in groups.values()):
        raise ValueError("每条训练示范都需要完整闭合抬升过渡窗口，不能静默丢弃示范")
    return groups, plan


class TransitionMixedStarts:
    """Two ordinary and two targeted draws per shuffled four-draw cycle.

    Both branches choose a recording uniformly, then a start uniformly.
    The ordinary branch retains ALL complete starts, including transitions.
    Repeated indices/recordings across draws are allowed, as in BalancedStarts.
    """
    def __init__(self, groups, transitions, seed=0):
        if (not groups or set(groups) != set(transitions)
                or any(not indices for indices in groups.values())
                or any(not indices or len(set(indices)) != len(indices)
                       or not set(indices) <= set(groups[task]) for task, indices in transitions.items())):
            raise ValueError("混合抽样需要每条示范的非空完整起点及其过渡子集")
        self.groups, self.transitions = groups, transitions
        self.tasks = sorted(groups)
        self.rng = random.Random(seed)
        self.cycle = []
        self.last_pool = None

    def draw(self):
        if not self.cycle:
            self.cycle = ["all", "all", "transition", "transition"]
            self.rng.shuffle(self.cycle)
        self.last_pool = self.cycle.pop()
        task = self.rng.choice(self.tasks)
        pool = self.transitions if self.last_pool == "transition" else self.groups
        return task, self.rng.choice(pool[task])
