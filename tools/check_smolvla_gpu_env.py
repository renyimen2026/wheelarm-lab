"""Import checks and a tiny CUDA operation, without downloading/loading weights."""

import argparse
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import sys

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

EXPECTED = {"lerobot": "0.4.4", "torch": "2.7.1", "torchvision": "0.22.1",
            "torchcodec": "0.5.0", "transformers": "4.57.1", "num2words": "0.5.14"}


def main(args=None):
    parser = argparse.ArgumentParser(description="检查独立 SmolVLA GPU 环境；只做导入和小矩阵运算，不加载权重")
    parser.add_argument("--output", help="可选：保存到新的 JSON 报告，不覆盖已有文件")
    options = parser.parse_args(args)
    try:
        from packaging.version import Version
        installed = {name: version(name) for name in EXPECTED}
        mismatches = {name: {"expected": expected, "installed": installed[name]}
                      for name, expected in EXPECTED.items()
                      if Version(installed[name].split("+", 1)[0]) != Version(expected)}
        if mismatches:
            raise ValueError("依赖版本不一致："+json.dumps(mismatches, ensure_ascii=False))
        if Version(installed["torchcodec"]).local not in (None, "cpu"):
            raise ValueError("本阶段需要 CPU TorchCodec；请按 tools/torchcodec-cpu-requirements.txt 单独替换，保留 CUDA PyTorch")
        import torch
        import torchvision
        from torchcodec.decoders import VideoDecoder
        from lerobot.datasets.lerobot_dataset import CODEBASE_VERSION, LeRobotDataset
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors

        if torch.version.cuda != "11.8":
            raise ValueError(f"预期 cu118 PyTorch，实际 CUDA runtime={torch.version.cuda}")
        if not torch.cuda.is_available():
            raise RuntimeError("PyTorch 无法使用 CUDA；请在能正常运行 nvidia-smi 的同一终端检查")
        if CODEBASE_VERSION != "v3.0":
            raise ValueError(f"数据格式版本不一致：{CODEBASE_VERSION}")
        device = torch.device("cuda:0")
        properties = torch.cuda.get_device_properties(device)
        torch.cuda.reset_peak_memory_stats(device)
        matrix = torch.arange(1024, dtype=torch.float32, device=device).reshape(32, 32)/1024
        product = matrix @ matrix.T
        torch.cuda.synchronize(device)
        if not torch.isfinite(product).all().item():
            raise RuntimeError("CUDA 小矩阵运算产生非有限值")
        cpu_matrix = matrix.cpu()
        if not torch.allclose(product.cpu(), cpu_matrix @ cpu_matrix.T, rtol=1e-4, atol=1e-5):
            raise RuntimeError("CUDA 小矩阵结果与 CPU 对照不一致")
        report = {
            "python_executable": sys.executable, "python_version": sys.version.split()[0],
            "packages": installed, "torch_cuda_runtime": torch.version.cuda,
            "gpu_name": properties.name, "gpu_total_memory_mib": properties.total_memory/1024**2,
            "compute_capability": list(torch.cuda.get_device_capability(device)),
            "bf16_supported": torch.cuda.is_bf16_supported(),
            "cuda_matrix_check": "passed", "torchvision_import": torchvision.__version__,
            "decoder_import": VideoDecoder.__name__, "torchcodec_backend": "cpu",
            "dataset_import": LeRobotDataset.__name__,
            "dataset_codebase_version": CODEBASE_VERSION, "policy_class_import": SmolVLAPolicy.__name__,
            "processor_factory_import": make_smolvla_pre_post_processors.__name__,
            "peak_tensor_memory_mib": torch.cuda.max_memory_allocated(device)/1024**2,
            "scope": "imports_and_tiny_cuda_operation_only",
            "model_instantiated": False, "weights_downloaded": False, "training_started": False,
        }
        if options.output:
            output = Path(options.output).expanduser()
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("x", encoding="utf-8") as stream:
                stream.write(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
        print("[GPU环境检查通过] "+json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        return 0
    except (PackageNotFoundError, ImportError, OSError, RuntimeError, ValueError) as error:
        print(f"[GPU环境检查失败] {error}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
