"""Offline SmolVLA configuration/numeric interface checks, without a model."""

import argparse
from importlib.metadata import version
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from lerobot.configs.types import FeatureType, NormalizationMode
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.processor_smolvla import SmolVLANewLineProcessor
from lerobot.processor import NormalizerProcessorStep, UnnormalizerProcessorStep, TransitionKey

from wheel_arm_manipulation.action_alignment import STATE_NAMES, STATE_UNITS, read_jsonl
from wheel_arm_manipulation.lerobot_conversion import IMAGE_KEY, STATE_KEY, REPO_ID, CHUNK_SIZE, write_json


def configure(metadata):
    features = dataset_to_policy_features(metadata.features)
    if set(features) != {IMAGE_KEY, STATE_KEY, "action"}:
        raise ValueError("此阶段要求单个 mast RGB、五维状态和五维动作")
    for key, kind in ((STATE_KEY, FeatureType.STATE), ("action", FeatureType.ACTION)):
        if features[key].type != kind or tuple(features[key].shape) != (5,):
            raise ValueError(f"{key} 的策略特征类型或维度不一致")
        if metadata.features[key]["names"] != list(STATE_NAMES):
            raise ValueError(f"{key} 关节顺序不一致")
    if features[IMAGE_KEY].type != FeatureType.VISUAL or features[IMAGE_KEY].shape[0] != 3:
        raise ValueError("图像策略特征必须是三通道 CHW")
    config = SmolVLAConfig(
        device="cpu", chunk_size=CHUNK_SIZE, n_action_steps=1, n_obs_steps=1,
        input_features={key: value for key, value in features.items() if value.type != FeatureType.ACTION},
        output_features={"action": features["action"]},
        adapt_to_pi_aloha=False, use_delta_joint_actions_aloha=False, compile_model=False,
    )
    config.validate_features()
    if config.normalization_mapping["STATE"] != NormalizationMode.MEAN_STD or config.normalization_mapping["ACTION"] != NormalizationMode.MEAN_STD:
        raise ValueError("SmolVLA 的状态/动作归一化规则与预期不一致")
    return config, features


def validate_stats(stats, features):
    result = {}
    for key in (STATE_KEY, "action"):
        if key not in stats:
            raise ValueError(f"缺少 {key} 归一化统计，禁止静默跳过归一化")
        values = {}
        for kind in ("mean", "std"):
            if kind not in stats[key]:
                raise ValueError(f"缺少 {key}.{kind} 归一化统计")
            tensor = torch.as_tensor(stats[key][kind], dtype=torch.float32)
            if tuple(tensor.shape) != tuple(features[key].shape) or not torch.isfinite(tensor).all():
                raise ValueError(f"{key}.{kind} 形状错误或含非有限值")
            if kind == "std" and (tensor < 0).any():
                raise ValueError(f"{key}.std 不能为负数")
            values[kind] = tensor.tolist()
        values["std_below_1e-6_dimensions"] = [name for name, std in zip(STATE_NAMES, values["std"]) if std < 1e-6]
        result[key] = values
    return result


def check_interface(directory):
    if version("lerobot") != "0.4.4":
        raise ValueError("接口检查固定使用 LeRobot 0.4.4")
    root = Path(directory).expanduser().resolve(strict=True)
    conversion = json.loads((root/"conversion_report.json").read_text(encoding="utf-8"))
    manifest = json.loads((root/"conversion_manifest.json").read_text(encoding="utf-8"))
    mapping = read_jsonl(root/"source_mapping.jsonl")
    if conversion.get("status") != "verified" or manifest.get("action_semantics") != "absolute_reference_at_grid_plus_0.1s":
        raise ValueError("必须使用已通过转换验证且动作语义明确的数据集")
    if manifest.get("names") != list(STATE_NAMES) or manifest.get("units") != list(STATE_UNITS):
        raise ValueError("来源清单的关节顺序或物理单位不一致")
    base = LeRobotDataset(repo_id=REPO_ID, root=root, download_videos=False)
    if base.num_episodes != 1 or base.fps != 10 or len(base) != len(mapping) or len(base) != manifest.get("rows"):
        raise ValueError("数据集帧数、episode 数量、FPS 或来源行数不一致")
    starts = manifest.get("full_chunk_start_indices")
    if starts != list(range(len(base)-CHUNK_SIZE+1)) or not starts:
        raise ValueError("完整动作块起点必须连续，不能包含尾部补齐起点")
    for index, provenance in enumerate(mapping):
        source = provenance["source_grid_row"]
        if (provenance["dataset_frame_index"] != index or source["action_stamp_ns"] != source["stamp_ns"]+100_000_000
                or (index and source["stamp_ns"]-mapping[index-1]["source_grid_row"]["stamp_ns"] != 100_000_000)):
            raise ValueError("来源映射的索引、连续性或动作时间偏移不一致")

    config, features = configure(base.meta)
    stats_report = validate_stats(base.meta.stats, features)
    deltas = resolve_delta_timestamps(config, base.meta)
    if deltas["action"] != [0., .1, .2, .3] or deltas[STATE_KEY] != [0] or deltas[IMAGE_KEY] != [0]:
        raise ValueError("官方数据加载器生成的时间偏移不符合预期")
    dataset = LeRobotDataset(repo_id=REPO_ID, root=root, download_videos=False, delta_timestamps=deltas)
    normalizer = NormalizerProcessorStep(features=features, norm_map=config.normalization_mapping,
                                         stats=base.meta.stats, device="cpu")
    unnormalizer = UnnormalizerProcessorStep(features=features, norm_map=config.normalization_mapping,
                                             stats=base.meta.stats, device="cpu")
    newline = SmolVLANewLineProcessor()
    loader = DataLoader(dataset, batch_size=8, sampler=starts, num_workers=0)
    max_state_error = max_action_error = max_state_normalized = max_action_normalized = 0.
    cursor, batches = 0, 0
    batch_shapes = None
    for batch in loader:
        count = batch["action"].shape[0]
        image, state, action = batch[IMAGE_KEY], batch[STATE_KEY], batch["action"]
        expected_image_shape = (count, 1, *tuple(features[IMAGE_KEY].shape))
        if tuple(image.shape) != expected_image_shape or tuple(state.shape) != (count, 1, 5) or tuple(action.shape) != (count, CHUNK_SIZE, 5):
            raise ValueError("带批次和时间维度的数据形状不符合配置")
        if any(not torch.isfinite(t).all() for t in (image, state, action)) or image.min() < 0 or image.max() > 1:
            raise ValueError("原始批次含非有限值或 RGB 超出 [0, 1]")
        if batch["action_is_pad"].any() or batch[STATE_KEY+"_is_pad"].any() or batch[IMAGE_KEY+"_is_pad"].any():
            raise ValueError("完整起点出现了时间 padding")
        expected = torch.tensor([mapping[index]["source_grid_row"]["future_absolute_references"]
                                 for index in starts[cursor:cursor+count]], dtype=torch.float32)
        if not torch.equal(action, expected):
            raise ValueError("官方加载器动作块与来源未来绝对动作不一致")
        expected_state = torch.tensor([mapping[index]["source_grid_row"]["state"]
                                       for index in starts[cursor:cursor+count]], dtype=torch.float32).unsqueeze(1)
        if not torch.equal(state, expected_state):
            raise ValueError("官方加载器实际状态与来源映射不一致")
        tasks = batch["task"]
        if any(task != manifest["instruction"] for task in tasks):
            raise ValueError("批次任务文本与来源不一致")
        if newline.complementary_data({"task": tasks})["task"] != [t if t.endswith("\n") else t+"\n" for t in tasks]:
            raise ValueError("SmolVLA 文本换行预处理不一致")
        transition = {TransitionKey.OBSERVATION: {IMAGE_KEY: image, STATE_KEY: state}, TransitionKey.ACTION: action}
        normalized = normalizer(transition)
        restored = unnormalizer(normalized)
        normalized_state = normalized[TransitionKey.OBSERVATION][STATE_KEY]
        normalized_action = normalized[TransitionKey.ACTION]
        if not torch.isfinite(normalized_state).all() or not torch.isfinite(normalized_action).all():
            raise ValueError("归一化产生非有限值")
        if not torch.equal(normalized[TransitionKey.OBSERVATION][IMAGE_KEY], image):
            raise ValueError("VISUAL IDENTITY 处理不应改变 [0,1] RGB")
        state_error = (restored[TransitionKey.OBSERVATION][STATE_KEY]-state).abs().max().item()
        action_error = (restored[TransitionKey.ACTION]-action).abs().max().item()
        max_state_error, max_action_error = max(max_state_error, state_error), max(max_action_error, action_error)
        max_state_normalized = max(max_state_normalized, normalized_state.abs().max().item())
        max_action_normalized = max(max_action_normalized, normalized_action.abs().max().item())
        if state_error > 1e-6 or action_error > 1e-6:
            raise ValueError("归一化/反归一化误差超过 1e-6 物理单位")
        if batch_shapes is None:
            batch_shapes = {key: list(batch[key].shape) for key in (IMAGE_KEY, STATE_KEY, "action", "action_is_pad")}
        cursor += count
        batches += 1
        if batches == 1 or batches % 10 == 0 or cursor == len(starts):
            print(f"[批次检查] 已核对 {cursor}/{len(starts)} 个完整起点；配置、来源动作和归一化通过。", flush=True)
    if cursor != len(starts):
        raise ValueError("未检查全部完整动作块起点")
    for index in range(len(starts), len(dataset)):
        expected = torch.tensor([index+i >= len(dataset) for i in range(CHUNK_SIZE)])
        if not torch.equal(dataset[index]["action_is_pad"], expected):
            raise ValueError("末尾不完整起点 padding 标记不一致")
    return {
        "schema_version": 1, "status": "passed", "scope": "smolvla_config_and_numeric_interface_only",
        "dataset": str(root), "lerobot_version": version("lerobot"),
        "dataset_frames": len(base), "complete_chunk_starts_checked": cursor, "batches_checked": batches,
        "policy_features": {key: {"type": value.type.value, "shape": list(value.shape)} for key, value in features.items()},
        "config_preview": {"chunk_size": config.chunk_size, "n_obs_steps": config.n_obs_steps,
                           "n_action_steps": config.n_action_steps, "n_action_steps_scope": "interface_test_placeholder_not_execution_setting",
                           "max_state_dim": config.max_state_dim, "max_action_dim": config.max_action_dim,
                           "resize_imgs_with_padding": list(config.resize_imgs_with_padding),
                           "adapt_to_pi_aloha": config.adapt_to_pi_aloha,
                           "use_delta_joint_actions_aloha": config.use_delta_joint_actions_aloha},
        "normalization_mapping": {str(key): value.value for key, value in config.normalization_mapping.items()},
        "first_batch_shapes": batch_shapes, "loader_delta_timestamps": deltas,
        "physical_action_offsets_seconds": [.1, .2, .3, .4],
        "normalization_stats": stats_report,
        "max_normalization_roundtrip_error": {"state": max_state_error, "action": max_action_error},
        "max_abs_normalized_value": {"state": max_state_normalized, "action": max_action_normalized},
        "padded_tail_starts_checked_and_excluded": CHUNK_SIZE-1,
        "language_text_newline_checked": True, "language_tokenization_checked": False,
        "model_image_resize_and_vector_padding_executed": False,
        "model_instantiated": False, "model_forward_checked": False,
        "weights_downloaded": False, "training_started": False,
    }


def main(args=None):
    parser = argparse.ArgumentParser(description="离线检查数据到 SmolVLA 配置与归一化接口；不加载模型或权重")
    parser.add_argument("dataset", help="已通过转换验证的本地 LeRobot 数据集目录")
    parser.add_argument("--output", required=True, help="新的 JSON 检查报告；不会覆盖已有文件")
    options = parser.parse_args(args)
    output = Path(options.output).expanduser()
    try:
        if output.exists():
            raise FileExistsError(f"报告已存在：{output}")
        report = check_interface(options.dataset)
        output.parent.mkdir(parents=True, exist_ok=True)
        write_json(output, report)
    except (KeyError, OSError, ValueError, RuntimeError, ImportError) as error:
        print(f"[接口检查失败] {error}", flush=True)
        return 1
    print(f"[接口检查通过] {report['dataset_frames']} 帧；"
          f"{report['complete_chunk_starts_checked']} 个完整四步起点，{report['batches_checked']} 个批次。", flush=True)
    print(f"[策略特征] {report['policy_features']}", flush=True)
    print(f"[首批形状] {report['first_batch_shapes']}", flush=True)
    print(f"[归一化] {report['normalization_mapping']}；"
          f"往返最大误差 {report['max_normalization_roundtrip_error']}", flush=True)
    print(f"[归一化数值] 最大绝对值 {report['max_abs_normalized_value']}", flush=True)
    print(f"[时间] 加载器 action 偏移 [0, 0.1, 0.2, 0.3] s；物理参考偏移 [0.1, 0.2, 0.3, 0.4] s。", flush=True)
    print(f"[保存] {output.resolve()}", flush=True)
    print("[范围] 已检查配置、数值归一化、动作块、padding 和文本换行；"
          "尚未检查语言分词、模型图像变换、内部补维或模型前向。未下载权重、未训练。", flush=True)
    return 0
