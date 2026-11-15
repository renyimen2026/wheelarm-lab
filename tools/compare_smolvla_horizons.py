"""Compare matched saved predictions by physical future horizon; no model loading."""

import argparse
import json
from pathlib import Path

import numpy as np

BANDS = (("0.1-1.0s", 0, 10), ("1.1-3.0s", 10, 30), ("3.1-5.0s", 30, 50))


def load(path):
    rows = [json.loads(line) for line in (Path(path)/"predictions.jsonl").read_text().splitlines() if line.strip()]
    keyed = {(r["source_task_id"], r["dataset_index"]): r for r in rows}
    if not rows or len(keyed) != len(rows):
        raise ValueError("Empty or duplicate prediction starts")
    return keyed


def compare(old_dir, new_dir):
    old, new = load(old_dir), load(new_dir)
    if old.keys() != new.keys():
        raise ValueError("Evaluation starts differ")
    targets, states, phases, predictions = [], [], [], [[], []]
    for key, a in old.items():
        b = new[key]
        for field in ("source_stamp_ns", "source_grid_index", "dataset_episode_index", "action_phases",
                      "target_physical", "state_physical"):
            if a[field] != b[field]:
                raise ValueError(f"Matched source differs: {key} / {field}")
        target, state = np.asarray(a["target_physical"]), np.asarray(a["state_physical"])
        if target.shape != (50, 5) or state.shape != (5,) or len(a["action_phases"]) != 50:
            raise ValueError("Invalid trajectory dimensions")
        targets.append(target)
        states.append(state)
        phases.append(a["action_phases"])
        for group, row in zip(predictions, (a, b)):
            pred = np.asarray(row["prediction_physical"])
            if pred.shape != target.shape or not np.isfinite(pred).all():
                raise ValueError("Invalid prediction")
            group.append(pred)
    target, state, phases = np.asarray(targets), np.asarray(states), np.asarray(phases)
    if not np.isfinite(target).all() or not np.isfinite(state).all():
        raise ValueError("Invalid physical source")
    errors = [np.asarray(p)-target for p in predictions]
    baseline = np.broadcast_to(state[:, None, :], target.shape)-target

    def metrics(error):
        return {"joint_mae_rad": abs(error[:, :4]).mean(axis=0).tolist(),
                "gripper_mae_mm": float(abs(error[:, 4]).mean()*1000),
                "gripper_signed_bias_mm": float(error[:, 4].mean()*1000),
                "gripper_p95_abs_mm": float(np.percentile(abs(error[:, 4]), 95)*1000)}

    result = {"matched_starts": len(old), "physical_offsets_seconds": [0.1*(i+1) for i in range(50)],
              "note": "Overlapping windows are not independent trials. Positive gripper bias means too open; negative means too closed. Bias alone does not prove a timing error.",
              "old": str(Path(old_dir).resolve()), "new": str(Path(new_dir).resolve()), "bands": {}}
    for label, begin, end in BANDS:
        groups = {}
        for phase in ["all"]+sorted(set(phases.flat)):
            mask = np.ones(phases[:, begin:end].shape, dtype=bool) if phase == "all" else phases[:, begin:end] == phase
            if not mask.any():
                continue
            groups[phase] = {"action_comparisons": int(mask.sum()),
                             **{name: metrics(err[:, begin:end][mask]) for name, err in
                                zip(("old", "new", "hold"), [*errors, baseline])}}
        result["bands"][label] = groups
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("old")
    parser.add_argument("new")
    parser.add_argument("--output", required=True)
    options = parser.parse_args()
    result = compare(options.old, options.new)
    with Path(options.output).open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    for band, groups in result["bands"].items():
        for phase in ("all", "approach", "close", "lift"):
            if phase in groups:
                g = groups[phase]
                print(band, phase, "N=", g["action_comparisons"],
                      "gripper MAE old/new/hold(mm)=", [round(g[k]["gripper_mae_mm"], 3) for k in ("old", "new", "hold")],
                      "bias old/new(mm)=", [round(g[k]["gripper_signed_bias_mm"], 3) for k in ("old", "new")])
    print("Saved:", options.output)


if __name__ == "__main__":
    main()
