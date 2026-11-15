"""Check the isolated data-conversion environment without accessing the Hub."""

import argparse
from importlib.metadata import PackageNotFoundError, version
import inspect
import json
from pathlib import Path
import sys


EXPECTED = {"lerobot": "0.4.4", "torch": "2.7.1", "torchvision": "0.22.1", "torchcodec": "0.5.0"}


def main(args=None):
    parser = argparse.ArgumentParser(description="检查 LeRobot CPU 数据环境；不下载数据或模型，不启动 ROS")
    parser.add_argument("--output", help="可选：将环境检查结果保存到一个新的 JSON 文件")
    options = parser.parse_args(args)
    try:
        from packaging.version import Version

        installed = {name: version(name) for name in EXPECTED}
        # Normalize release spellings (0.5 == 0.5.0), while keeping prerelease
        # and postrelease distinctions. Local wheel tags such as +cpu are
        # checked separately through torch.version.cuda below.
        mismatches = {name: {"expected": expected, "installed": installed[name]}
                      for name, expected in EXPECTED.items()
                      if Version(installed[name].split("+", 1)[0]) != Version(expected)}
        if mismatches:
            raise ValueError("依赖版本与固定配置不一致："+json.dumps(mismatches, ensure_ascii=False))
        import torch
        import torchvision
        from torchcodec.decoders import VideoDecoder
        from lerobot.datasets.lerobot_dataset import CODEBASE_VERSION, LeRobotDataset

        # Imports also exercise the PyTorch/TorchVision and FFmpeg decoder linkage.
        if CODEBASE_VERSION != "v3.0":
            raise ValueError(f"不符合预期的数据格式版本：{CODEBASE_VERSION}")
        required = {"repo_id", "fps", "features", "root", "use_videos"}
        if not required.issubset(inspect.signature(LeRobotDataset.create).parameters):
            raise ValueError("LeRobotDataset.create 接口与预期不一致")
        for method in ("add_frame", "save_episode", "finalize", "__getitem__"):
            if not callable(getattr(LeRobotDataset, method, None)):
                raise ValueError(f"LeRobotDataset 缺少接口：{method}")
        if torch.version.cuda is not None:
            raise ValueError("本阶段预期为 CPU PyTorch；请使用 --torch-backend cpu 安装")
        packages = {name: version(name) for name in (*EXPECTED, "datasets", "pyarrow", "numpy", "Pillow")}
        report = {"python_executable": sys.executable, "python_version": sys.version.split()[0],
                  "packages": packages, "dataset_codebase_version": CODEBASE_VERSION,
                  "torch_backend": "cpu", "dataset_import": "passed",
                  "torchvision_import": torchvision.__version__, "decoder_import": VideoDecoder.__name__,
                  "scope": "imports_and_interface_only; no_dataset_conversion_or_model_loading"}
        if options.output:
            output = Path(options.output).expanduser()
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("x", encoding="utf-8") as stream:
                stream.write(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
        print("[环境检查通过] "+json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        return 0
    except (PackageNotFoundError, ImportError, OSError, RuntimeError, ValueError) as error:
        print(f"[环境检查失败] {error}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
