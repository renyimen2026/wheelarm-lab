"""Convert one verified contiguous available-past segment; read back locally."""

import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path

import numpy as np
from PIL import Image

from wheel_arm_manipulation.action_alignment import STATE_NAMES, STATE_UNITS, load_episode, read_jsonl
from wheel_arm_manipulation.episode_segment import select_grasp_hold
from wheel_arm_manipulation.time_grid import (
    AVAILABLE_GRID_FILE, AVAILABLE_GRID_REPORT_FILE, PERIOD_NS, build_time_grid,
)

IMAGE_KEY = "observation.images.mast"
STATE_KEY = "observation.state"
REPO_ID = "local/wheelarm_grasp_hold_validation"
CHUNK_SIZE = 4


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2)+"\n")


def prepare_episode_runs(directory):
    """Recompute the diagnostic to reject stale or modified derived inputs."""
    episode = load_episode(directory)
    frames, segment = select_grasp_hold(episode)
    expected_rows, expected_report = build_time_grid(
        frames, episode["references"], segment, CHUNK_SIZE, "available-past")
    root = episode["root"]
    rows = read_jsonl(root/AVAILABLE_GRID_FILE)
    report = json.loads((root/AVAILABLE_GRID_REPORT_FILE).read_text(encoding="utf-8"))
    if rows != expected_rows or report != expected_report:
        raise ValueError("时间网格文件与原始记录/当前规则不一致；请重新运行 available-past 网格诊断")
    accepted = [row for row in rows if row["status"] == "accepted"]
    if not accepted:
        raise ValueError("没有有效网格点")
    runs = []
    for row in accepted:
        if (not runs or row["stamp_ns"]-runs[-1][-1]["stamp_ns"] != PERIOD_NS
                or row["grid_index"] != runs[-1][-1]["grid_index"]+1):
            runs.append([])
        runs[-1].append(row)
    if len(runs) != len(report["valid_contiguous_runs"]):
        raise ValueError("连续段数量与网格报告不一致")
    instruction = accepted[0]["instruction"]
    image_shape = None
    for row in accepted:
        if row["instruction"] != instruction or row["names"] != list(STATE_NAMES) or row["units"] != list(STATE_UNITS):
            raise ValueError("状态/动作名称、单位或任务指令不一致")
        for key in ("state", "action"):
            if np.asarray(row[key]).shape != (5,) or not np.isfinite(row[key]).all():
                raise ValueError("每行 state 和 action 必须是五维有限数值")
        if row["action_stamp_ns"] != row["stamp_ns"]+PERIOD_NS:
            raise ValueError("单步动作时间偏移必须为 +0.1 s")
        image = row["image"]
        if not image["stamp_ns"] <= image["pair_validation_stamp_ns"] <= row["stamp_ns"]:
            raise ValueError("图像未在网格时刻前通过配对检查")
        image_path = (root/image["file"]).resolve(strict=True)
        if not image_path.is_relative_to(root):
            raise ValueError("源图像路径必须位于源记录目录内")
        with Image.open(image_path) as rgb:
            shape = (rgb.height, rgb.width, 3)
            if rgb.mode != "RGB" or (image_shape is not None and shape != image_shape):
                raise ValueError("源图像必须都是相同尺寸的 RGB 图像")
            rgb.verify()
        image_shape = shape
    # Verify that adjacent single-step labels recover the original four-step labels.
    for run in runs:
        for index in range(len(run)-CHUNK_SIZE+1):
            action_chunk = np.asarray([r["action"] for r in run[index:index+CHUNK_SIZE]])
            if not np.allclose(action_chunk, run[index]["future_absolute_references"], rtol=0., atol=1e-9):
                raise ValueError("相邻单步 action 与原始未来四步标签不一致")
    return root, runs, report, image_shape


def prepare_episode(directory):
    """Preserve the original single-contiguous-episode converter contract."""
    root, runs, report, image_shape = prepare_episode_runs(directory)
    if len(runs) != 1 or len(runs[0]) < CHUNK_SIZE:
        raise ValueError("本阶段仅转换一段连续且至少包含四个槽位的示范，不拼接缺口")
    return root, runs[0], report, image_shape


def read_rgb(root, row):
    with Image.open(root/row["image"]["file"]) as image:
        return np.array(image, dtype=np.uint8)


def readback(output, source, rows, image_shape, dataset_class):
    """Check every single row/image and every complete four-step start."""
    dataset = dataset_class(repo_id=REPO_ID, root=output, download_videos=False)
    if dataset.num_episodes != 1 or len(dataset) != len(rows) or dataset.fps != 10:
        raise ValueError("读回的数据集 episode 数量、帧数或 FPS 不一致")
    max_timestamp_error = 0.
    for index, row in enumerate(rows):
        sample = dataset[index]
        for key, source_key in ((STATE_KEY, "state"), ("action", "action")):
            expected = np.asarray(row[source_key], dtype=np.float32)
            actual = sample[key].cpu().numpy()
            if actual.shape != (5,) or actual.dtype != np.float32 or not np.array_equal(actual, expected):
                raise ValueError(f"第 {index} 行 {key} 读回不一致")
        timestamp_error = abs(sample["timestamp"].item()-index/10)
        max_timestamp_error = max(max_timestamp_error, timestamp_error)
        if (timestamp_error > 1e-4 or sample["frame_index"].item() != index
                or sample["episode_index"].item() != 0 or sample["task"] != row["instruction"]):
            raise ValueError(f"第 {index} 行时间戳、索引或任务指令不一致")
        image = sample[IMAGE_KEY].cpu().numpy()
        if image.shape != (3, image_shape[0], image_shape[1]) or not np.isfinite(image).all():
            raise ValueError(f"第 {index} 行 RGB 张量形状或有限值检查失败")
        if image.min() < 0. or image.max() > 1.:
            raise ValueError(f"第 {index} 行 RGB 张量必须归一化到 [0, 1]")
        pixels = np.rint(image.transpose(1, 2, 0)*255).astype(np.uint8)
        if not np.array_equal(pixels, read_rgb(source, row)):
            raise ValueError(f"第 {index} 行 RGB 图像读回像素不一致")
    chunks = dataset_class(repo_id=REPO_ID, root=output, download_videos=False,
                           delta_timestamps={"action": [i/10 for i in range(CHUNK_SIZE)]})
    complete_starts = len(rows)-CHUNK_SIZE+1
    for index in range(complete_starts):
        sample = chunks[index]
        actual = sample["action"].cpu().numpy()
        expected = np.asarray(rows[index]["future_absolute_references"], dtype=np.float32)
        if actual.shape != (CHUNK_SIZE, 5) or not np.array_equal(actual, expected):
            raise ValueError(f"第 {index} 个完整动作块与原始未来动作不一致")
        if sample["action_is_pad"].any().item():
            raise ValueError("完整动作块起点出现了 padding")
    for index in range(complete_starts, len(rows)):
        mask = chunks[index]["action_is_pad"].cpu().numpy()
        expected = np.asarray([index+offset >= len(rows) for offset in range(CHUNK_SIZE)])
        if not np.array_equal(mask, expected):
            raise ValueError("末尾不完整动作块的 padding 标记不一致")
    return {"rows_checked": len(rows), "images_pixel_checked": len(rows),
            "single_action_shape": [5], "loaded_chunk_shape": [CHUNK_SIZE, 5],
            "complete_chunk_starts_checked": complete_starts,
            "padded_tail_starts_excluded": CHUNK_SIZE-1,
            "max_timestamp_float32_error_seconds": max_timestamp_error,
            "all_state_action_values_exact_after_float32_cast": True}


def convert_episode(directory, destination):
    if version("lerobot") != "0.4.4":
        raise ValueError("转换器固定使用 LeRobot 0.4.4，请先运行环境检查脚本")
    source = Path(directory).expanduser().resolve(strict=True)
    output = Path(destination).expanduser().resolve()
    if output.is_relative_to(source):
        raise ValueError("输出必须位于源记录目录之外")
    if output.exists():
        raise FileExistsError(f"输出目录已存在，不会覆盖：{output}")
    source, rows, grid_report, image_shape = prepare_episode(source)
    print(f"[输入核对] {len(rows)} 个连续槽位；原始网格索引 "
          f"{rows[0]['grid_index']}～{rows[-1]['grid_index']}；源记录和网格一致。", flush=True)
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    features = {
        IMAGE_KEY: {"dtype": "image", "shape": image_shape, "names": ["height", "width", "channels"]},
        STATE_KEY: {"dtype": "float32", "shape": (5,), "names": list(STATE_NAMES)},
        "action": {"dtype": "float32", "shape": (5,), "names": list(STATE_NAMES)},
    }
    source_files = ("summary.json", "frames.jsonl", "controller_references.jsonl", "task_metadata.json",
                    "task_events.jsonl", AVAILABLE_GRID_FILE, AVAILABLE_GRID_REPORT_FILE)
    hashes = {name: digest(source/name) for name in source_files}
    manifest = {"schema_version": 1, "source_episode": str(source),
                "source_task_id": grid_report["segment"]["source_task_id"],
                "source_file_sha256": hashes, "rows": len(rows), "fps": 10,
                "original_first_grid_index": rows[0]["grid_index"],
                "original_first_grid_stamp_ns": rows[0]["stamp_ns"],
                "original_last_grid_stamp_ns": rows[-1]["stamp_ns"],
                "names": list(STATE_NAMES), "units": list(STATE_UNITS),
                "instruction": rows[0]["instruction"], "action_semantics": "absolute_reference_at_grid_plus_0.1s",
                "image_semantics": "RGB_lossless_embedded_images; available-past; max_age_100ms",
                "state_semantics": "offline_linear_interpolation_of_actual_joint_state",
                "dataset_timestamps": "zero_origin_at_first_exported_grid_slot; source_ns_preserved_in_sidecar",
                "chunk_size_for_validation": CHUNK_SIZE,
                "full_chunk_start_indices": list(range(len(rows)-CHUNK_SIZE+1)),
                "tail_policy": "exclude_padded_starts_from_full_chunk_validation; no_lower_stage_actions",
                "packages": {name: version(name) for name in (
                    "lerobot", "torch", "torchvision", "torchcodec", "datasets", "pyarrow", "numpy", "Pillow")}}
    output.parent.mkdir(parents=True, exist_ok=True)
    dataset = LeRobotDataset.create(repo_id=REPO_ID, fps=10, features=features, root=output,
                                   robot_type="wheelarm_4dof_parallel_gripper", use_videos=False,
                                   image_writer_processes=0, image_writer_threads=0)
    try:
        write_json(output/"conversion_manifest.json", manifest)
        with (output/"source_mapping.jsonl").open("x", encoding="utf-8") as mapping:
            for index, row in enumerate(rows):
                dataset.add_frame({IMAGE_KEY: read_rgb(source, row),
                                   STATE_KEY: np.asarray(row["state"], dtype=np.float32),
                                   "action": np.asarray(row["action"], dtype=np.float32),
                                   "task": row["instruction"]})
                provenance = {"dataset_frame_index": index, "dataset_timestamp_seconds": index/10,
                              "source_grid_row": row, "source_image_sha256": digest(source/row["image"]["file"])}
                mapping.write(json.dumps(provenance, ensure_ascii=False, allow_nan=False)+"\n")
        dataset.save_episode()
    finally:
        dataset.finalize()
    print("[读回验证] 已保存并关闭写入器，逐行检查状态、动作、RGB 和四步动作块。", flush=True)
    checks = readback(output, source, rows, image_shape, LeRobotDataset)
    if any(digest(source/name) != value for name, value in hashes.items()):
        raise ValueError("转换期间源文件发生变化，不能确认结果")
    result = {"schema_version": 1, "status": "verified", "source_episode": str(source),
              "output_dataset": str(output), "repo_id": REPO_ID, "lerobot_version": version("lerobot"),
              "dataset_codebase_version": dataset.meta.info["codebase_version"],
              "episodes": 1, "frames": len(rows), "fps": 10, "image_shape_hwc": list(image_shape),
              "checks": checks, "uploaded": False, "model_loaded": False, "training_started": False}
    write_json(output/"conversion_report.json", result)
    return result


def main(args=None):
    parser = argparse.ArgumentParser(description="单条连续 grasp-hold 示范转本地 LeRobot 0.4.4，并逐帧读回验证")
    parser.add_argument("episode", help="包含 available-past 网格诊断的原始记录目录")
    parser.add_argument("--output", required=True, help="新的本地数据集目录；不允许覆盖或写入源记录内")
    options = parser.parse_args(args)
    try:
        result = convert_episode(options.episode, options.output)
    except (KeyError, OSError, ValueError, RuntimeError, ImportError) as error:
        print(f"[转换失败] {error}；如输出目录已创建，将保留以便排查，不会自动覆盖重试。", flush=True)
        return 1
    print(f"[转换验证通过] {result['episodes']} episode，{result['frames']} 帧，10 Hz；"
          f"单步 action={result['checks']['single_action_shape']}。", flush=True)
    print(f"[图像] {result['checks']['images_pixel_checked']} 张逐像素读回一致；"
          f"RGB HWC={result['image_shape_hwc']}，使用无损图片存储。", flush=True)
    print(f"[动作块] {result['checks']['complete_chunk_starts_checked']} 个完整四步起点读回一致；"
          "末尾 3 个不完整起点的 padding 标记已检查，未计入完整动作块。", flush=True)
    print(f"[保存] {result['output_dataset']}", flush=True)
    print("[报告] conversion_report.json；[来源] conversion_manifest.json / source_mapping.jsonl", flush=True)
    print("[说明] 仅本地单条数据转换验证；未上传、未加载模型、未训练。", flush=True)
    return 0
