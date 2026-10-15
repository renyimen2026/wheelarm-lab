"""Describe one live RGB frame, or replay a saved observation; never move."""

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
from uuid import uuid4

from wheel_arm_navigation.mission import utc_now
from wheel_arm_navigation import vision
from wheel_arm_navigation.observation import save_rgb_capture
from wheel_arm_navigation.scene_snapshot import positive


def parser():
    result = argparse.ArgumentParser(description="识别命令触发后的单帧相机画面，不移动机器人")
    result.add_argument("observation", nargs="?", help="可选：传入已有观测目录则离线识别")
    result.add_argument("--rgb-topic", default="/camera/image")
    result.add_argument("--wait", type=positive, default=15, help="相机采集等待秒数，与推理超时独立")
    result.add_argument("--max-age", type=positive, default=0.5)
    result.add_argument("--observations-dir", default="observations", help="实时采集图片的保存根目录")
    result.add_argument("--question", default=vision.DEFAULT_QUESTION)
    result.add_argument("--model", default="qwen3-vl:4b-instruct")
    result.add_argument("--endpoint", default="http://127.0.0.1:11434")
    result.add_argument("--timeout", type=float, default=180)
    result.add_argument("--output-dir", default=None, help="默认保存在观测目录的 descriptions/ 内")
    return result


def save_report(directory, report):
    root = Path(directory).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    destination = root / ("vision-" + uuid4().hex + ".json")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=root,
                                         prefix=".vision-", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        # Atomic publication without overwriting even if a destination already exists.
        os.link(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return destination.resolve()


def main(argv=None):
    from rclpy.utilities import remove_ros_args

    ros_argv = sys.argv if argv is None else ["scene_describe", *argv]
    options = parser().parse_args(remove_ros_args(ros_argv)[1:])
    try:
        if options.observation:
            root, metadata, image = vision.read_observation(options.observation)
            source_mode = "offline"
        else:
            from wheel_arm_navigation.live_image import capture

            image, metadata = capture(options, ros_argv)
            root, metadata = save_rgb_capture(options.observations_dir, image, metadata)
            source_mode = "live_once"
            print(f"已冻结单帧：{root / 'rgb.png'}；开始识别，后续画面不会替换此帧", flush=True)
        result, elapsed = vision.describe(image, options.question, options.model,
                                          options.endpoint, options.timeout)
        report = {"schema_version": 1, "created_at": utc_now(),
                  "observation_id": metadata["observation_id"], "image_sha256": vision.image_fingerprint(image),
                  "model": options.model, "endpoint": options.endpoint,
                  "question": options.question, "elapsed_sec": elapsed,
                  "input_modalities": ["rgb"], "source_mode": source_mode,
                  "reference_frame": metadata.get("reference_frame"),
                  "image_stamps": metadata.get("stamps"), "result": result,
                  "limitations": ["模型推测，不是已验证事实", "画面方位不等于地图方位",
                                   "未进行深度定位、导航或机械臂控制"]}
        path = save_report(options.output_dir or root / "descriptions", report)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print(f"耗时：{elapsed:.3f} 秒；结果：{path}")
        return 0
    except KeyboardInterrupt:
        print("视觉理解已中断；未发送任何移动指令", file=sys.stderr)
        return 130
    except (ValueError, OSError, RuntimeError) as error:
        print(f"视觉理解失败：{error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
