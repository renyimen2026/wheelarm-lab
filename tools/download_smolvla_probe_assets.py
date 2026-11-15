"""Download pinned public assets for a later single-batch forward check.

Only the complete SmolVLA checkpoint is downloaded. The VLM repository provides
configuration/processor/tokenizer files, not a second copy of backbone weights.
No model is constructed and no dataset, token, or local file is uploaded.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re

from huggingface_hub import HfApi, snapshot_download


POLICY_ID = "lerobot/smolvla_base"
VLM_ID = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
POLICY_FILES = ["config.json", "model.safetensors"]
VLM_PATTERNS = ["*.json", "*.jinja", "*.txt", "*.model"]


def save_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2)+"\n")


def validate_plan(plan):
    if plan.get("policy_repo_id") != POLICY_ID or plan.get("vlm_repo_id") != VLM_ID:
        raise ValueError("下载计划的模型来源与本阶段固定来源不一致")
    for key in ("policy_revision", "vlm_revision"):
        if not isinstance(plan.get(key), str) or not re.fullmatch(r"[0-9a-f]{40}", plan[key]):
            raise ValueError(f"下载计划缺少有效提交号：{key}")


def download(directory):
    root = Path(directory).expanduser().resolve()
    plan_file, manifest_file = root/"download_plan.json", root/"assets_manifest.json"
    if root.exists() and not plan_file.exists() and any(root.iterdir()):
        raise ValueError("目标目录已有其他文件且没有本工具的下载计划；请使用新的素材目录")
    root.mkdir(parents=True, exist_ok=True)
    if plan_file.exists():
        plan = json.loads(plan_file.read_text(encoding="utf-8"))
        validate_plan(plan)
    else:
        api = HfApi()
        policy = api.model_info(POLICY_ID, files_metadata=True)
        vlm = api.model_info(VLM_ID)
        plan = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
                "policy_repo_id": POLICY_ID, "policy_revision": policy.sha,
                "vlm_repo_id": VLM_ID, "vlm_revision": vlm.sha,
                "policy_files": POLICY_FILES, "vlm_allow_patterns": VLM_PATTERNS,
                "policy_selected_size_bytes": sum(s.size or 0 for s in policy.siblings if s.rfilename in POLICY_FILES)}
        validate_plan(plan)
        # Pin before downloading so interruption/retry uses the same revisions.
        save_json(plan_file, plan)
    print(f"[固定提交] {POLICY_ID}@{plan['policy_revision']}", flush=True)
    print(f"[固定提交] {VLM_ID}@{plan['vlm_revision']}", flush=True)
    if plan.get("policy_selected_size_bytes"):
        print(f"[策略文件大小] 约 {plan['policy_selected_size_bytes']/1024**3:.2f} GiB。", flush=True)
    policy_dir, vlm_dir = root/"policy", root/"vlm_config_tokenizer"
    snapshot_download(repo_id=POLICY_ID, revision=plan["policy_revision"], local_dir=policy_dir,
                      allow_patterns=POLICY_FILES, max_workers=2)
    config = json.loads((policy_dir/"config.json").read_text(encoding="utf-8"))
    if config.get("type") != "smolvla" or config.get("vlm_model_name") != VLM_ID:
        raise ValueError("下载的策略配置不是预期的 SmolVLA 主干组合")
    if not (policy_dir/"model.safetensors").is_file():
        raise ValueError("缺少完整 SmolVLA 权重")
    snapshot_download(repo_id=VLM_ID, revision=plan["vlm_revision"], local_dir=vlm_dir,
                      allow_patterns=VLM_PATTERNS, max_workers=2)
    for name in ("config.json", "tokenizer_config.json"):
        if not (vlm_dir/name).is_file():
            raise ValueError(f"主干配置/分词器缺少文件：{name}")
    if list(vlm_dir.rglob("*.safetensors")) or list(vlm_dir.rglob("*.bin")):
        raise ValueError("主干目录存在本阶段不应下载的权重文件")
    manifest = {"schema_version": 1, "status": "downloaded", "plan": plan,
                "policy_dir": str(policy_dir), "vlm_config_tokenizer_dir": str(vlm_dir),
                "checkpoint_chunk_size": config.get("chunk_size"),
                "checkpoint_max_state_dim": config.get("max_state_dim"),
                "checkpoint_max_action_dim": config.get("max_action_dim"),
                "model_instantiated": False, "gpu_operation_run": False, "training_started": False}
    if manifest_file.exists():
        if json.loads(manifest_file.read_text(encoding="utf-8")) != manifest:
            raise ValueError("已有素材清单与本次下载不一致；保留原文件，请检查")
    else:
        save_json(manifest_file, manifest)
    print(f"[下载完成] {root}", flush=True)
    print(f"[配置] chunk_size={config.get('chunk_size')}，max_state_dim={config.get('max_state_dim')}，"
          f"max_action_dim={config.get('max_action_dim')}", flush=True)
    print(f"[清单] {manifest_file}", flush=True)
    print("[范围] 仅下载公开权重和配置；未上传文件、未加载模型、未运行前向或训练。", flush=True)
    return manifest


def main(args=None):
    parser = argparse.ArgumentParser(description="下载单批次前向所需的公开素材，并固定提交号；不加载模型")
    parser.add_argument("--assets-dir", required=True, help="独立素材目录；支持按已有下载计划重试")
    options = parser.parse_args(args)
    try:
        download(options.assets_dir)
    except (OSError, ValueError, RuntimeError) as error:
        print(f"[下载失败] {error}；下载计划和已完成文件保留，可用同一命令重试。", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
