"""Run with .venv-lerobot/bin/python -I; does not require a sourced ROS environment."""

import os
from pathlib import Path
import sys

# The converter uses local files only; prevent library fallbacks to the Hub.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"wheel_arm_manipulation"))

from wheel_arm_manipulation.lerobot_conversion import main


if __name__ == "__main__":
    raise SystemExit(main())
