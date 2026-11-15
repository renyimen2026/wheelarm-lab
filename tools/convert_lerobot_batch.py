"""Local batch conversion entry; use the isolated CPU data environment."""

import os
from pathlib import Path
import sys

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"wheel_arm_manipulation"))
from wheel_arm_manipulation.lerobot_batch_conversion import main


if __name__ == "__main__":
    # Per-frame image conversion consists of small CPU tensor operations;
    # large intra-op thread pools add overhead and oversubscribe the desktop.
    import torch
    torch.set_num_threads(1)
    raise SystemExit(main())
