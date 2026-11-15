"""Offline split-aware conversion: one LeRobot episode per contiguous run."""

import argparse
from importlib.metadata import version
import json
from pathlib import Path

import numpy as np

from wheel_arm_manipulation.action_alignment import STATE_NAMES, STATE_UNITS, read_jsonl
from wheel_arm_manipulation.lerobot_conversion import (
    IMAGE_KEY, STATE_KEY, digest, prepare_episode_runs, read_rgb, write_json,
)
from wheel_arm_manipulation.time_grid import AVAILABLE_GRID_FILE, AVAILABLE_GRID_REPORT_FILE

SOURCE_FILES = ("summary.json", "frames.jsonl", "controller_references.jsonl", "task_metadata.json",
                "task_events.jsonl", AVAILABLE_GRID_FILE, AVAILABLE_GRID_REPORT_FILE)
HORIZONS = (4, 50)


def prepare_batch(split_file):
    path = Path(split_file).expanduser().resolve(strict=True)
    split_hash = digest(path)
    specification = json.loads(path.read_text(encoding="utf-8"))
    if specification.get("split_unit") != "original_recording_all_derived_segments_inherit_split":
        raise ValueError("必须使用按原始示范固定的集合划分")
    entries = specification["episodes"]
    if not entries or {entry["split"] for entry in entries} != {"train", "validation"}:
        raise ValueError("必须显式提供非空 train 和 validation 集合")
    seen_paths, seen_tasks = set(), set()
    prepared, expected_shape, expected_instruction = [], None, None
    for entry in entries:
        source = Path(entry["episode_dir"]).expanduser().resolve(strict=True)
        metadata = json.loads((source/"task_metadata.json").read_text(encoding="utf-8"))
        if (not entry["task_id"] or metadata["task_id"] != entry["task_id"]
                or metadata.get("task_success") is not True or metadata.get("lift_verified") is not True
                or metadata.get("execution_status") != "finished" or metadata.get("finish") != "hold"):
            raise ValueError("集合划分的任务 ID 或成功抓起保持结果不一致")
        if source in seen_paths or entry["task_id"] in seen_tasks:
            raise ValueError("同一原始示范/任务重复出现，可能造成训练与验证泄漏")
        seen_paths.add(source)
        seen_tasks.add(entry["task_id"])
        hashes = {name: digest(source/name) for name in SOURCE_FILES}
        _, runs, grid_report, shape = prepare_episode_runs(source)
        instruction = runs[0][0]["instruction"]
        if expected_shape is not None and (shape != expected_shape or instruction != expected_instruction):
            raise ValueError("本批示范的图像尺寸或任务指令不一致")
        expected_shape, expected_instruction = shape, instruction
        image_hashes = {row["image"]["file"]: digest(source/row["image"]["file"])
                        for run in runs for row in run}
        prepared.append({"source": source, "split": entry["split"], "task_id": entry["task_id"],
                         "runs": runs, "hashes": hashes, "image_hashes": image_hashes,
                         "grid_report": grid_report})
    if digest(path) != split_hash:
        raise ValueError("准备期间集合划分文件发生变化")
    return path, split_hash, prepared, expected_shape


def make_segments(prepared, split):
    segments, offset = [], 0
    for parent in prepared:
        if parent["split"] != split:
            continue
        for run_index, rows in enumerate(parent["runs"]):
            segments.append({"parent": parent, "rows": rows, "run_index": run_index,
                             "episode_index": len(segments), "global_from": offset,
                             "global_to": offset+len(rows)})
            offset += len(rows)
    return segments


def verify_dataset(output, repo_id, segments, shape, dataset_class):
    if not (output/"meta/info.json").is_file():
        raise ValueError(f"待读回的数据集不存在或尚未写完：{output}")
    base = dataset_class(repo_id=repo_id, root=output, download_videos=False)
    count = sum(len(segment["rows"]) for segment in segments)
    if base.num_episodes != len(segments) or base.fps != 10 or len(base) != count:
        raise ValueError("数据集读回的 episode 数、帧数或频率不一致")
    for segment in segments:
        rows, source, ep = segment["rows"], segment["parent"]["source"], segment["episode_index"]
        for index, row in enumerate(rows):
            sample = base[segment["global_from"]+index]
            if (sample["episode_index"].item() != ep or sample["frame_index"].item() != index
                    or abs(sample["timestamp"].item()-index/10) > 1e-4 or sample["task"] != row["instruction"]):
                raise ValueError("读回的片段边界、索引、时间或任务文本不一致")
            for key, source_key in ((STATE_KEY, "state"), ("action", "action")):
                actual = sample[key].cpu().numpy()
                if actual.shape != (5,) or not np.array_equal(actual, np.asarray(row[source_key], dtype=np.float32)):
                    raise ValueError("状态/动作逐行读回不一致")
            image = sample[IMAGE_KEY].cpu().numpy()
            if image.shape != (3, shape[0], shape[1]) or not np.isfinite(image).all() or image.min() < 0 or image.max() > 1:
                raise ValueError("RGB 形状、范围或有限值不正确")
            pixels = np.rint(image.transpose(1, 2, 0)*255).astype(np.uint8)
            if not np.array_equal(pixels, read_rgb(source, row)):
                raise ValueError("RGB 像素逐张读回不一致")
        print(f"[逐帧读回] episode {ep+1}/{len(segments)} 完成；累计 "
              f"{segment['global_to']}/{count} 张图像逐像素一致。", flush=True)
    chunk_checks = {}
    for horizon in HORIZONS:
        loader = dataset_class(repo_id=repo_id, root=output, download_videos=False,
                               delta_timestamps={"action": [i/10 for i in range(horizon)]})
        # Images were checked above. Dataset column indexing can decode every
        # column while querying action windows, so exclude image storage here.
        # Keep the official LeRobot delta-index/padding implementation intact.
        loader._ensure_hf_dataset_loaded()
        loader.hf_dataset = loader.hf_dataset.select_columns([
            key for key in loader.hf_dataset.column_names if key != IMAGE_KEY])
        complete, tails = 0, 0
        for segment in segments:
            rows = segment["rows"]
            for index in range(len(rows)):
                sample = loader[segment["global_from"]+index]
                # LeRobot clamps to THIS episode's final row for padded steps.
                # Verify those values too, so no neighbor episode can leak in.
                expected = np.asarray([rows[min(index+i, len(rows)-1)]["action"]
                                       for i in range(horizon)], dtype=np.float32)
                mask = np.asarray([index+i >= len(rows) for i in range(horizon)])
                if (not np.array_equal(sample["action"].cpu().numpy(), expected)
                        or not np.array_equal(sample["action_is_pad"].cpu().numpy(), mask)):
                    raise ValueError(f"{horizon} 步动作块或尾部 padding 错误；可能跨片段读取")
                if mask.any(): tails += 1
                else: complete += 1
            print(f"[动作块读回 {horizon} 步] episode {segment['episode_index']+1}/{len(segments)} 完成。", flush=True)
        chunk_checks[str(horizon)] = {"complete_starts_checked": complete, "padded_starts_checked_and_excluded": tails}
    return {"rows_checked": count, "images_pixel_checked": count, "chunks": chunk_checks,
            "all_state_action_values_exact_after_float32_cast": True,
            "no_chunk_crosses_episode_boundary": True}


def convert_split(output, split, prepared, shape, dataset_class):
    repo_id = f"local/wheelarm_grasp_hold_batch_{split}"
    segments = make_segments(prepared, split)
    features = {
        IMAGE_KEY: {"dtype": "image", "shape": shape, "names": ["height", "width", "channels"]},
        STATE_KEY: {"dtype": "float32", "shape": (5,), "names": list(STATE_NAMES)},
        "action": {"dtype": "float32", "shape": (5,), "names": list(STATE_NAMES)},
    }
    episode_manifest = []
    for segment in segments:
        rows, parent = segment["rows"], segment["parent"]
        episode_manifest.append({"dataset_episode_index": segment["episode_index"],
            "dataset_from_index": segment["global_from"], "dataset_to_index": segment["global_to"],
            "source_episode": str(parent["source"]), "source_task_id": parent["task_id"],
            "source_run_index": segment["run_index"], "rows": len(rows),
            "first_source_grid_index": rows[0]["grid_index"], "last_source_grid_index": rows[-1]["grid_index"],
            "first_source_stamp_ns": rows[0]["stamp_ns"], "last_source_stamp_ns": rows[-1]["stamp_ns"],
            "source_task_success": True, "segment_task_success": None})
    manifest = {"schema_version": 1, "split": split, "repo_id": repo_id, "episodes": episode_manifest,
        "rows": sum(len(s["rows"]) for s in segments), "fps": 10,
        "names": list(STATE_NAMES), "units": list(STATE_UNITS), "instruction": segments[0]["rows"][0]["instruction"],
        "action_semantics": "absolute_reference_at_grid_plus_0.1s",
        "state_semantics": "offline_linear_interpolation_of_actual_joint_state",
        "image_semantics": "RGB_lossless_embedded_images; available-past; max_age_100ms",
        "time_semantics": "each_run_zero_origin; original_grid_index_and_ns_preserved; no_gap_stitching",
        "success_semantics": "verified_success_applies_to_original_task_only; runs_are_not_complete_grasps",
        "full_chunk_start_indices": {str(h): [i for s in segments for i in
              range(s["global_from"], max(s["global_from"], s["global_to"]-h+1))] for h in HORIZONS}}
    dataset = dataset_class.create(repo_id=repo_id, fps=10, features=features, root=output,
                                   robot_type="wheelarm_4dof_parallel_gripper", use_videos=False,
                                   image_writer_processes=0, image_writer_threads=0)
    try:
        write_json(output/"conversion_manifest.json", manifest)
        with (output/"source_mapping.jsonl").open("x", encoding="utf-8") as mapping:
            for segment in segments:
                source, rows = segment["parent"]["source"], segment["rows"]
                for index, row in enumerate(rows):
                    dataset.add_frame({IMAGE_KEY: read_rgb(source, row),
                        STATE_KEY: np.asarray(row["state"], dtype=np.float32),
                        "action": np.asarray(row["action"], dtype=np.float32), "task": row["instruction"]})
                    mapping.write(json.dumps({"dataset_index": segment["global_from"]+index,
                        "dataset_episode_index": segment["episode_index"], "dataset_frame_index": index,
                        "dataset_timestamp_seconds": index/10, "split": split,
                        "source_episode": str(source), "source_task_id": segment["parent"]["task_id"],
                        "source_run_index": segment["run_index"], "source_grid_row": row,
                        "source_image_sha256": segment["parent"]["image_hashes"][row["image"]["file"]]},
                        ensure_ascii=False, allow_nan=False)+"\n")
                dataset.save_episode()
    finally:
        dataset.finalize()
    print(f"[读回 {split}] {manifest['rows']} 帧、{len(segments)} 段；逐行、逐像素核对，并检查 4/50 步及尾部 padding。", flush=True)
    checks = verify_dataset(output, repo_id, segments, shape, dataset_class)
    result = {"schema_version": 1, "status": "verified", "split": split,
              "output_dataset": str(output), "repo_id": repo_id, "frames": manifest["rows"],
              "episodes": len(segments), "checks": checks}
    write_json(output/"conversion_report.json", result)
    return result


def previous_training_statistics(root, prepared):
    """Keep a fine-tuned model's normalizer while adding TRAIN data only."""
    root=Path(root).resolve(strict=True)
    report=json.loads((root/'batch_conversion_report.json').read_text())
    normalization=report['normalization_stats']
    stats=root/'training_stats.json'
    if (report.get('status')!='verified' or normalization.get('validation_used') is not False
            or digest(stats)!=normalization['sha256']
            or digest(root/'train/meta/stats.json')!=normalization['sha256']):
        raise ValueError('继承统计需要已验证且未混入验证集的冻结训练统计')
    current={p['task_id']:p for p in prepared if p['split']=='train'}
    sources=json.loads((root/'source_manifest.json').read_text())['sources']
    training=[s for s in sources if s['split']=='train']
    if not training:
        raise ValueError('继承统计缺少训练来源')
    for source in training:
        p=current.get(source['task_id'])
        if (p is None or str(p['source'])!=source['source_episode']
                or p['hashes']!=source['source_file_sha256']):
            raise ValueError('统计来源必须全部属于当前训练集，且原始文件未变化')
    return stats.read_bytes(),{'frozen_from':str(stats),
        'source_training_recordings':len(training),'new_training_statistics_recomputed_for_audit_only':True}


def convert_batch(split_file, destination, normalization_root=None):
    if version("lerobot") != "0.4.4":
        raise ValueError("转换器固定使用 LeRobot 0.4.4")
    path = Path(split_file).expanduser().resolve(strict=True)
    output = Path(destination).expanduser().resolve()
    if output.is_relative_to(path.parent):
        raise ValueError("输出目录必须位于原始批次目录之外")
    if output.exists():
        raise FileExistsError(f"输出目录已存在，不会覆盖：{output}")
    path, split_hash, prepared, shape = prepare_batch(path)
    if any(output.is_relative_to(parent["source"]) for parent in prepared):
        raise ValueError("输出目录必须位于所有原始示范之外")
    frozen,origin=previous_training_statistics(normalization_root,prepared) if normalization_root else (None,{})
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    print(f"[输入核对] {len(prepared)} 条成功原始示范；所有网格已按原始记录重新计算核对。", flush=True)
    output.mkdir(parents=True)
    write_json(output/"source_manifest.json", {"schema_version": 1, "split_file": str(path),
        "split_file_sha256": split_hash, "sources": [{"source_episode": str(p["source"]),
        "split": p["split"], "task_id": p["task_id"], "source_file_sha256": p["hashes"],
        "run_lengths": [len(run) for run in p["runs"]]} for p in prepared]})
    reports = {split: convert_split(output/split, split, prepared, shape, LeRobotDataset)
               for split in ("train", "validation")}
    if digest(path) != split_hash or any(digest(p["source"]/name) != expected
            for p in prepared for name, expected in {**p["hashes"], **p["image_hashes"]}.items()):
        raise ValueError("转换期间原始文件或集合划分发生变化，不能确认结果")
    # Freeze ONLY the training statistics as the future normalizer's source.
    # Validation's automatically produced statistics are never merged into it.
    if frozen is not None:
        computed=output/'train/meta/stats.json'
        (output/'recomputed_training_stats_audit.json').write_bytes(computed.read_bytes())
        computed.write_bytes(frozen)
    stats_path = output/"training_stats.json"
    with stats_path.open("xb") as stream:
        stream.write((output/"train/meta/stats.json").read_bytes())
    result = {"schema_version": 1, "status": "verified", "output": str(output), "splits": reports,
        "original_demonstrations": len(prepared), "split_unit": "original_recording",
        "normalization_stats": {"file": "training_stats.json", "sha256": digest(stats_path),
                                 "source": "train/meta/stats.json", "validation_used": False,**origin},
        "packages": {name: version(name) for name in ("lerobot", "torch", "datasets", "pyarrow", "numpy", "Pillow")},
        "uploaded": False, "model_loaded": False, "training_started": False}
    write_json(output/"batch_conversion_report.json", result)
    return result


def main(args=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("split_file", help="已固定的 collection_split.json")
    parser.add_argument("--output", required=True, help="新的本地输出根目录；不覆盖已有文件")
    parser.add_argument('--normalization-root',help='保留原训练集冻结统计；其全部统计来源必须仍属于本次训练集')
    options = parser.parse_args(args)
    try:
        result = convert_batch(options.split_file, options.output,options.normalization_root)
    except (KeyError, OSError, ValueError, RuntimeError, ImportError) as error:
        print(f"[批量转换失败] {error}；如已创建部分输出，将保留用于排查，不会自动覆盖重试。", flush=True)
        return 1
    for split, report in result["splits"].items():
        print(f"[转换验证通过 {split}] {report['episodes']} 个连续片段、{report['frames']} 帧；"
              f"完整 50 步起点 {report['checks']['chunks']['50']['complete_starts_checked']}。", flush=True)
    print(f"[保存] {result['output']}", flush=True)
    print("[统计] 已单独冻结训练集统计；验证集不参与归一化统计。", flush=True)
    print("[范围] 仅本地数据转换及读回验证；片段不是完整成功抓取。未上传、未加载模型、未训练。", flush=True)
    return 0
