"""One local pretrained GPU batch: forward loss and action sampling, no training."""

import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys
import time

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"wheel_arm_manipulation"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors
from lerobot.utils.constants import OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK

from wheel_arm_manipulation.action_alignment import read_jsonl, STATE_NAMES, STATE_UNITS
from wheel_arm_manipulation.lerobot_conversion import IMAGE_KEY, STATE_KEY, REPO_ID, write_json
from wheel_arm_manipulation.smolvla_interface import configure, validate_stats


def load_policy(checkpoint, config):
    from smolvla_precision import restore_trainable_precision
    policy = SmolVLAPolicy.from_pretrained(checkpoint, config=config, local_files_only=True, strict=True)
    restore_trainable_precision(policy, checkpoint)
    return policy


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_assets(directory):
    root = Path(directory).expanduser().resolve(strict=True)
    manifest = read_json(root/"assets_manifest.json")
    plan = manifest["plan"]
    expected = {
        "policy_repo_id": "lerobot/smolvla_base",
        "policy_revision": "5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6",
        "vlm_repo_id": "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        "vlm_revision": "7b375e1b73b11138ff12fe22c8f2822d8fe03467",
    }
    if manifest.get("status") != "downloaded" or any(plan.get(k) != v for k, v in expected.items()):
        raise ValueError("素材来源/提交号与本次验证固定版本不一致")
    policy, vlm = root/"policy", root/"vlm_config_tokenizer"
    for path in (policy/"config.json", policy/"model.safetensors", vlm/"config.json", vlm/"tokenizer_config.json"):
        if not path.is_file() or not path.resolve().is_relative_to(root):
            raise ValueError(f"本地素材缺失或越出素材目录：{path}")
    return policy, vlm, expected


def adapt_config(config, metadata, vlm):
    # Keep the checkpoint's architecture/horizon. Only adapt external features
    # and local loading; the full checkpoint will replace ALL initial weights.
    numeric_config, features = configure(metadata)
    if (config.type != "smolvla" or config.chunk_size not in (10,50) or config.n_obs_steps != 1
            or config.max_state_dim != 32 or config.max_action_dim != 32):
        raise ValueError("预训练结构与本阶段 10/50 步、32 维内部接口不一致")
    if config.normalization_mapping != numeric_config.normalization_mapping:
        raise ValueError("预训练归一化规则与已验证的数据接口不一致")
    config.input_features = numeric_config.input_features
    config.output_features = numeric_config.output_features
    config.vlm_model_name = str(vlm)
    config.load_vlm_weights = False
    config.device = "cpu"  # strict-load once on CPU, then move to CUDA.
    config.push_to_hub = False
    config.compile_model = False
    config.validate_features()
    return config, features


def validate_window(mapping, index, horizon):
    if index < 0 or index+horizon > len(mapping):
        raise ValueError("起点不足以构成完整 50 步动作块，禁止尾部 padding")
    window = mapping[index:index+horizon]
    start = window[0]["source_grid_row"]["stamp_ns"]
    for step, row in enumerate(window):
        grid = row["source_grid_row"]
        if (row["dataset_frame_index"] != index+step or grid["status"] != "accepted"
                or grid["stamp_ns"] != start+step*100_000_000
                or grid["action_stamp_ns"] != grid["stamp_ns"]+100_000_000):
            raise ValueError("选定动作块的来源索引、有效性或物理时间不一致")
    return torch.tensor([row["source_grid_row"]["action"] for row in window], dtype=torch.float32)


def prepare_batch(directory, policy_dir, vlm_dir, index):
    root = Path(directory).expanduser().resolve(strict=True)
    manifest = read_json(root/"conversion_manifest.json")
    if (read_json(root/"conversion_report.json").get("status") != "verified"
            or manifest.get("action_semantics") != "absolute_reference_at_grid_plus_0.1s"
            or manifest.get("names") != list(STATE_NAMES) or manifest.get("units") != list(STATE_UNITS)):
        raise ValueError("需要已验证且关节顺序、单位和动作语义一致的转换数据")
    base = LeRobotDataset(repo_id=REPO_ID, root=root, download_videos=False)
    mapping = read_jsonl(root/"source_mapping.jsonl")
    if base.num_episodes != 1 or base.fps != 10 or len(base) != len(mapping) or len(base) != manifest["rows"]:
        raise ValueError("数据集数量、FPS 或来源行数不一致")
    config, features = adapt_config(PreTrainedConfig.from_pretrained(policy_dir, local_files_only=True), base.meta, vlm_dir)
    validate_stats(base.meta.stats, features)
    expected = validate_window(mapping, index, config.chunk_size)
    deltas = resolve_delta_timestamps(config, base.meta)
    if deltas["action"] != [i/10 for i in range(config.chunk_size)]:
        raise ValueError("加载器动作时间偏移不一致")
    dataset = LeRobotDataset(repo_id=REPO_ID, root=root, download_videos=False, delta_timestamps=deltas)
    batch = next(iter(DataLoader(Subset(dataset, [index]), batch_size=1, num_workers=0)))
    if not torch.equal(batch["action"], expected.unsqueeze(0)):
        raise ValueError("加载后的 50 步动作与来源单步动作不一致")
    for key in ("action", STATE_KEY, IMAGE_KEY):
        if batch[key+"_is_pad"].any() or not torch.isfinite(batch[key]).all():
            raise ValueError(f"{key} 含 padding 或非有限值")
    if batch["task"] != [manifest["instruction"]]:
        raise ValueError("任务文本与来源不一致")
    tokenizer = AutoTokenizer.from_pretrained(vlm_dir, local_files_only=True)
    text = manifest["instruction"].rstrip("\n")+"\n"
    token_count = len(tokenizer(text, truncation=False)["input_ids"])
    if token_count > config.tokenizer_max_length:
        raise ValueError(f"任务文本 {token_count} tokens 超过上限 {config.tokenizer_max_length}，禁止静默截断")
    return config, base.meta.stats, batch, {
        "dataset": str(root), "dataset_frames": len(base), "frame_index": index,
        "source_grid_index": mapping[index]["source_grid_row"]["grid_index"],
        "complete_50_step_starts": len(base)-config.chunk_size+1,
        "instruction": manifest["instruction"], "unpadded_language_tokens": token_count,
        "loader_action_offsets_seconds": deltas["action"],
        "physical_action_offsets_seconds": [(i+1)/10 for i in range(config.chunk_size)],
    }


def require_tensor(value, shape, label):
    if tuple(value.shape) != tuple(shape) or not torch.isfinite(value).all():
        raise ValueError(f"{label} 形状错误或含非有限值：{tuple(value.shape)}")


def probe(dataset, assets, index):
    if version("lerobot") != "0.4.4":
        raise ValueError("本阶段固定使用 LeRobot 0.4.4")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise ValueError("本阶段需要支持 BF16 的 CUDA GPU")
    policy_dir, vlm_dir, revisions = load_assets(assets)
    config, stats, batch, input_report = prepare_batch(dataset, policy_dir, vlm_dir, index)
    print(f"[输入] 起点 {index}；单批次 50 步，物理动作偏移 +0.1～+5.0 s；无时间 padding。", flush=True)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    print("[模型] 从本地完整权重严格加载；不另载 VLM 权重。", flush=True)
    policy = SmolVLAPolicy.from_pretrained(policy_dir, config=config, local_files_only=True, strict=True)
    policy.to("cuda").eval()
    config.device = "cuda"
    preprocessor, postprocessor = make_smolvla_pre_post_processors(config, dataset_stats=stats)
    processed = preprocessor(batch)
    require_tensor(processed["action"], (1, 50, 5), "归一化动作")
    require_tensor(processed[OBS_LANGUAGE_TOKENS], (1, 48), "语言 tokens")
    require_tensor(processed[OBS_LANGUAGE_ATTENTION_MASK], (1, 48), "语言 mask")
    versions = {name: parameter._version for name, parameter in policy.named_parameters()}
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        images, masks = policy.prepare_images(processed)
        if len(images) != 1 or not masks[0].all():
            raise ValueError("需要单张有效 mast 相机输入")
        require_tensor(images[0], (1, 3, 512, 512), "模型图像")
        require_tensor(policy.prepare_state(processed), (1, 32), "内部状态")
        require_tensor(policy.prepare_action(processed), (1, 50, 32), "内部动作")
        if images[0].min() < -1 or images[0].max() > 1:
            raise ValueError("模型图像超出 [-1,1]")
        del images, masks
        print("[前向] 计算单批次 flow-matching loss；不反向传播。", flush=True)
        started = time.perf_counter()
        loss, loss_details = policy.forward(processed)
        torch.cuda.synchronize()
        forward_seconds = time.perf_counter()-started
        require_tensor(loss, (), "loss")
        print(f"[loss] {loss.item():.6g}；开始采样 50 步动作。", flush=True)
        # Sampling receives observations/language only, never target actions.
        inference_batch = {k: v for k, v in processed.items() if k != "action" and not k.startswith("action_")}
        started = time.perf_counter()
        prediction = policy.predict_action_chunk(inference_batch)
        torch.cuda.synchronize()
        sample_seconds = time.perf_counter()-started
        require_tensor(prediction, (1, 50, 5), "归一化预测")
        physical = postprocessor(prediction)
        require_tensor(physical, (1, 50, 5), "反归一化预测")
    if any(p.grad is not None or p._version != versions[n] for n, p in policy.named_parameters()):
        raise ValueError("模型产生梯度或参数发生了原地变更")
    return {
        "schema_version": 1, "status": "passed", "scope": "one_pretrained_batch_forward_and_sampling_only",
        "assets": str(Path(assets).resolve()), "revisions": revisions, "input": input_report,
        "packages": {name: version(name) for name in ("lerobot", "torch", "transformers", "torchcodec")},
        "gpu": torch.cuda.get_device_name(), "strict_checkpoint_load": True,
        "autocast_dtype": "bfloat16", "parameter_dtype": str(next(policy.parameters()).dtype),
        "processed_shapes": {k: list(v.shape) for k, v in processed.items() if isinstance(v, torch.Tensor)},
        "model_image_shape": [1, 3, 512, 512], "internal_state_shape": [1, 32], "internal_action_shape": [1, 50, 32],
        "loss": loss.item(), "loss_details": loss_details,
        "loss_scope": "compatibility_only; official_0.4.4_loss_includes_32_padded_action_dimensions; not_grasp_accuracy",
        "prediction_shape": list(physical.shape), "prediction_physical_units": list(STATE_UNITS),
        "prediction_min_per_dimension": physical.amin(dim=(0, 1)).tolist(),
        "prediction_max_per_dimension": physical.amax(dim=(0, 1)).tolist(),
        "forward_seconds": forward_seconds, "sampling_seconds": sample_seconds,
        "timing_scope": "first_call_no_warmup_not_a_throughput_benchmark",
        "peak_allocated_mib": torch.cuda.max_memory_allocated()/1024**2,
        "peak_reserved_mib": torch.cuda.max_memory_reserved()/1024**2,
        "gradients_absent_and_parameter_versions_unchanged": True,
        "training_started": False, "robot_commands_sent": False,
    }


def main(args=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset")
    parser.add_argument("--assets-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--frame-index", type=int, default=0)
    options = parser.parse_args(args)
    output = Path(options.output).expanduser().resolve()
    try:
        if output.exists():
            raise FileExistsError(f"报告已存在：{output}")
        for directory in (options.dataset, options.assets_dir):
            if output.is_relative_to(Path(directory).expanduser().resolve()):
                raise ValueError("报告应写入独立诊断目录，不写入数据集或模型素材目录")
        report = probe(options.dataset, options.assets_dir, options.frame_index)
        output.parent.mkdir(parents=True, exist_ok=True)
        write_json(output, report)
    except (OSError, ValueError, RuntimeError, ImportError, KeyError) as error:
        print(f"[前向检查失败] {type(error).__name__}: {error}", flush=True)
        return 1
    print(f"[前向检查通过] loss={report['loss']:.6g}；预测形状 {report['prediction_shape']}。", flush=True)
    print(f"[耗时] loss 前向 {report['forward_seconds']:.2f} s；动作采样 {report['sampling_seconds']:.2f} s。", flush=True)
    print(f"[显存] 张量峰值 {report['peak_allocated_mib']:.1f} MiB；缓存峰值 {report['peak_reserved_mib']:.1f} MiB。", flush=True)
    print(f"[保存] {output}", flush=True)
    print("[范围] 仅接口兼容性验证；loss 包含内部补齐维度，不能当作抓杯能力指标。未训练、未发送机器人命令。", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
