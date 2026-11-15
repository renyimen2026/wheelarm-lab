"""Plot saved numerical validation results using system Python/matplotlib."""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/wheelarm-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def save_figure(fig, directory, name):
    for extension in ("png", "svg"):
        fig.savefig(directory/f"{name}.{extension}", dpi=160, bbox_inches="tight")
    plt.close(fig)


def trajectory(record, directory, name, title):
    target, prediction = np.asarray(record["target_physical"]), np.asarray(record["prediction_physical"])
    state = np.asarray(record["state_physical"])
    x = np.arange(1, len(target)+1)/10
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    for i, ax in enumerate(list(axes.flat)[:5]):
        scale = 1000 if i == 4 else 1
        ax.plot(x, target[:, i]*scale, color="#202124", linewidth=2, label="Target reference")
        ax.plot(x, prediction[:, i]*scale, color="#1769ba", linewidth=1.4, label="Saved model prediction")
        ax.axhline(state[i]*scale, color="#777777", linestyle="--", label="Hold current state")
        ax.set(title=["Shoulder pan", "Shoulder lift", "Elbow", "Wrist pitch", "Gripper width"][i],
               xlabel="Future action time (s)", ylabel="Width (mm)" if i == 4 else "Angle (rad)")
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
        ax.grid(alpha=.2)
    ax = axes.flat[5]
    ax.plot(x, abs(prediction[:, 4]-target[:, 4])*1000, color="#1769ba", label="Prediction")
    ax.plot(x, abs(state[4]-target[:, 4])*1000, color="#777777", linestyle="--", label="Hold")
    ax.set(title="Gripper absolute error", xlabel="Future action time (s)", ylabel="Error (mm)")
    ax.grid(alpha=.2)
    fig.legend(*axes.flat[0].get_legend_handles_labels(), loc="lower center", ncol=3, frameon=False)
    fig.suptitle(f"{title}\nStart index {record['dataset_index']} | start phase: {record['start_phase']} | open-loop prediction", fontsize=13)
    fig.tight_layout(rect=(0, .05, 1, .93))
    save_figure(fig, directory, name)


def plot_results(directory):
    root = Path(directory).expanduser().resolve(strict=True)
    report = json.loads((root/"evaluation_report.json").read_text())
    if report.get("status") != "completed":
        raise ValueError("只绘制已完成的完整验证结果")
    records = [json.loads(line) for line in (root/"predictions.jsonl").read_text().splitlines() if line.strip()]
    if len(records) != report["summary"]["evaluated_starts"]:
        raise ValueError("逐起点结果数量与报告不一致")
    output = root/"plots"
    output.mkdir(exist_ok=False)
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    tasks = list(dict.fromkeys(record["source_task_id"] for record in records))
    for task_index, task in enumerate(tasks):
        group = [r for r in records if r["source_task_id"] == task]
        origin = min(r["source_stamp_ns"] for r in group)
        x = np.asarray([(r["source_stamp_ns"]-origin)/1e9 for r in group])
        fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
        axes[0].plot(x, [r["physical_errors"]["gripper_mae_mm"] for r in group], color="#1769ba", label="Saved model")
        axes[0].plot(x, [r["baseline"]["gripper_mae_mm"] for r in group], color="#777777", linestyle="--", label="Hold current state")
        axes[0].set(ylabel="Gripper MAE (mm)", title=f"{len(group[0]['target_physical'])}-step action error at every evaluated validation start")
        axes[0].legend(loc="upper right")
        for i, label in enumerate(("Shoulder pan", "Shoulder lift", "Elbow", "Wrist pitch")):
            axes[1].plot(x, [r["physical_errors"]["joint_mae_rad"][i] for r in group], linewidth=1.1, label=label)
        axes[1].set(ylabel="Joint MAE (rad)")
        axes[1].legend(ncol=4, loc="upper right")
        axes[2].plot(x, [r["fixed_flow_loss"] for r in group], color="#1769ba")
        axes[2].set(ylabel="Fixed 5D flow loss", xlabel="Observation start time from first eligible start (s)")
        start = 0
        for end in range(1, len(group)+1):
            if end == len(group) or group[end]["start_phase"] != group[start]["start_phase"]:
                left, right = x[start]-.05, x[end-1]+.05
                for ax in axes:
                    ax.axvspan(left, right, color="#777777", alpha=.045 if start % 2 else .1)
                axes[0].text((left+right)/2, 1.02, group[start]["start_phase"].replace("before_or_between_phases", "before/between"),
                             ha="center", fontsize=8, transform=axes[0].get_xaxis_transform())
                start = end
        for ax in axes:
            ax.grid(alpha=.2)
        fig.suptitle("Full validation coverage | overlapping windows, not independent trials", fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, .94))
        save_figure(fig, output, f"all_starts_{task_index}")
    closing = max(records, key=lambda r: r["target_physical"][0][4]-r["target_physical"][-1][4])
    worst = max(records, key=lambda r: r["physical_errors"]["gripper_mae_mm"])
    trajectory(closing, output, "largest_closing_window", "Largest target closing motion")
    trajectory(worst, output, "worst_gripper_mae_window", "Largest gripper MAE among ALL evaluated starts")
    print(f"[轨迹图] {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evaluation_dir")
    plot_results(parser.parse_args().evaluation_dir)
