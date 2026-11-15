"""Local-only interface validation using the existing CPU data environment."""

import os
from pathlib import Path
import sys

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"wheel_arm_manipulation"))

from wheel_arm_manipulation.smolvla_interface import main


if __name__ == "__main__":
    raise SystemExit(main())
