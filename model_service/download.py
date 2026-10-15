"""Explicit downloads only; the inference process never fetches model files."""

import argparse
from pathlib import Path


def main():
    from huggingface_hub import snapshot_download

    parser = argparse.ArgumentParser()
    parser.add_argument("--models-root", type=Path, default=Path("~/Models/transformers"))
    parser.add_argument("--model", choices=["text", "vision", "all"], default="all")
    args = parser.parse_args()
    models = {"text": "Qwen3-4B", "vision": "Qwen3-VL-4B-Instruct"}
    for kind, name in models.items():
        if args.model in (kind, "all"):
            path = snapshot_download(
                "Qwen/" + name, local_dir=args.models_root.expanduser() / name,
                allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja", "*.model"],
                max_workers=2)
            print(kind, path, flush=True)


if __name__ == "__main__":
    main()
