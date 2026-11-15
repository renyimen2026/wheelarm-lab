"""Freeze batch02 development/test manifests after successful collection.

The old validation recording remains development validation. The two new
independent recordings are excluded from the development conversion entirely.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "wheel_arm_manipulation"))
from wheel_arm_manipulation.task_phases import parse_phases, phase_at


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def audit(entry):
    episode = Path(entry["episode_dir"])
    task = read(episode / "task_metadata.json")
    events = lines(episode / "task_events.jsonl")
    report = read(episode / "grasp_hold_available_time_grid_report.json")
    if task["task_id"] != entry["task_id"] or task["task_success"] is not True:
        raise ValueError("任务标记与采集报告不一致")
    phases = parse_phases(events, task["task_id"])
    hold_start = next(e["stamp_ns"] for e in events if e["event"] == "hold_started")
    hold_end = next(e["stamp_ns"] for e in events if e["event"] == "hold_verified")
    if hold_end-hold_start < 10_000_000_000:
        raise ValueError("稳定持杯记录不足十个仿真秒")
    rows = lines(episode / "grasp_hold_available_time_grid.jsonl")
    accepted = {r["grid_index"]: r for r in rows if r["status"] == "accepted"}
    starts = [r for i, r in accepted.items() if all(i+j in accepted for j in range(50))]
    close_end = next(end for phase, start, end in phases if phase == "close")
    transition = [r for r in starts if r["stamp_ns"] <= close_end <= r["stamp_ns"]+5_000_000_000]
    holding = [r for r in starts if hold_start <= r["stamp_ns"] and r["stamp_ns"]+5_000_000_000 <= hold_end]
    if not transition or not holding:
        raise ValueError("缺少连续闭合抬升窗口或完整持杯窗口")
    hold_rows = [r for r in accepted.values() if hold_start <= r['stamp_ns'] < hold_end]
    actual_widths = [r['state'][4]*1000 for r in hold_rows]
    reference_widths = [r['action'][4]*1000 for r in hold_rows]
    return {**entry, "hold_recorded_sim_seconds": (hold_end-hold_start)/1e9,
            "initial_observed_state": min(accepted.values(), key=lambda r: r['stamp_ns'])['state'],
            "hold_actual_width_range_mm": [min(actual_widths), max(actual_widths)],
            "hold_reference_width_range_mm": [min(reference_widths), max(reference_widths)],
            "hold_max_abs_reference_minus_actual_mm": max(abs(a-b) for a, b in zip(actual_widths, reference_widths)),
            "complete_50_step_starts": len(starts), "close_lift_boundary_windows": len(transition),
            "stable_hold_windows": len(holding),
            "start_phase_counts": {phase: sum(phase_at(r['stamp_ns'], phases) == phase for r in starts)
                                   for phase in ['open', 'pregrasp', 'approach', 'close', 'lift']},
            "grid_image_age_p95_ms": report["accepted_image_age_ms"]["p95"],
            "task_events_sha256": hashlib.sha256((episode / 'task_events.jsonl').read_bytes()).hexdigest()}


def prepare(root, old_split):
    collection = read(root / "collection_report.json")
    if collection["status"] != "verified" or len(collection["episodes"]) != 8:
        raise ValueError("必须先完成八条示范采集")
    plan = read(root / "collection_plan.json")
    entries = collection["episodes"]
    if len({e['task_id'] for e in entries}) != 8 or len({e['episode_dir'] for e in entries}) != 8:
        raise ValueError("原始示范重复")
    for entry, slot in zip(entries, plan["slots"]):
        if entry['slot'] != slot['slot'] or entry['split'] != slot['split']:
            raise ValueError("采集前固定的集合划分发生变化")
    rows = [audit(e) for e in entries]
    new_train = [e for e in entries if e['split'] == 'train']
    new_test = [e for e in entries if e['split'] == 'independent_test']
    if (len(new_train), len(new_test)) != (6, 2):
        raise ValueError("要求六条训练、两条独立测试")
    old = read(old_split)['episodes']
    common = {'schema_version': 1, 'split_unit': 'original_recording_all_derived_segments_inherit_split',
              'generalization_scope': 'fixed_cup_camera_with_small_initial_state_variations'}
    def entries_for(values, split):
        return [{'episode_dir': e['episode_dir'], 'task_id': e['task_id'], 'split': split} for e in values]
    training = [e for e in old if e['split'] == 'train'] + entries_for(new_train, 'train')
    validation = [e for e in old if e['split'] == 'validation']
    write(root / 'development_split.json', {**common, 'scope': 'training_and_old_development_validation',
                                          'episodes': training + validation})
    write(root / 'independent_test_split.json', {**common,
          'scope': 'final_evaluation_only_not_for_training_or_checkpoint_selection',
          'episodes': training + entries_for(new_test, 'validation')})
    write(root / 'preparation_report.json', {'status': 'verified', 'new_recordings': rows,
          'development': {'train_recordings': len(training), 'validation_recordings': len(validation)},
          'independent_test_recordings': len(new_test),
          'independent_test_excluded_from_development': True, 'training_started': False})
    print(json.dumps({'status': 'verified', 'new_successful_recordings': 8,
          'development_train': len(training), 'development_validation': len(validation),
          'independent_test': 2, 'full50_starts_new_train': sum(r['complete_50_step_starts'] for r in rows if r['split'] == 'train')},
          ensure_ascii=False), flush=True)


def convert(root, destination, normalization_root=None):
    for label, manifest in [('development', 'development_split.json'),
                            ('independent_test', 'independent_test_split.json')]:
        output = destination / label
        command = [str(REPO / '.venv-lerobot/bin/python'), '-I',
                   str(REPO / 'tools/convert_lerobot_batch.py'), str(root / manifest),
                   '--output', str(output)]
        if normalization_root is not None:
            command.extend(['--normalization-root',str(normalization_root)])
        print(f'[转换] {label}: {output}', flush=True)
        with (root / f'conversion_{label}.log').open('x') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        report = read(output / 'batch_conversion_report.json')
        if report['status'] != 'verified':
            raise ValueError('转换读回未通过')
        print(f'[读回通过] {label}', flush=True)
    finalize(root, destination)


def finalize(root, destination):
    dev, heldout = destination / 'development', destination / 'independent_test'
    # LeRobot aggregates identical numbers in a dict with process-dependent key
    # order. Match values AND training provenance before sharing frozen bytes.
    if read(dev / 'training_stats.json') != read(heldout / 'training_stats.json'):
        raise ValueError('两份数据集的训练统计数值不同')
    if ((dev / 'train/source_mapping.jsonl').read_bytes() !=
            (heldout / 'train/source_mapping.jsonl').read_bytes()):
        raise ValueError('独立测试数据集包含不同的训练行')
    frozen = (dev / 'training_stats.json').read_bytes()
    checksum = hashlib.sha256(frozen).hexdigest()
    held_report = read(heldout / 'batch_conversion_report.json')
    if (heldout / 'training_stats.json').read_bytes() != frozen:
        original = (heldout / 'training_stats.json').read_bytes()
        with (heldout / 'original_training_stats_serialization.json').open('xb') as stream:
            stream.write(original)
        for path in [heldout / 'train/meta/stats.json', heldout / 'training_stats.json']:
            if read(path) != json.loads(frozen):
                raise ValueError('冻结统计前检测到数值差异')
            temporary = path.with_name(path.name+'.freeze.tmp')
            with temporary.open('xb') as stream:
                stream.write(frozen)
            temporary.replace(path)
        held_report['normalization_stats'].update(sha256=checksum,
            identical_values_shared_serialization_from=str(dev / 'training_stats.json'),
            original_serialization_sha256=hashlib.sha256(original).hexdigest())
        path = heldout / 'batch_conversion_report.json'
        temporary = path.with_name(path.name+'.freeze.tmp')
        write(temporary, held_report)
        temporary.replace(path)
    outputs = {}
    for label in ['development', 'independent_test']:
        output = destination / label
        report = read(output / 'batch_conversion_report.json')
        if (report['status'] != 'verified' or
                report['normalization_stats']['sha256'] != checksum or
                (output / 'train/meta/stats.json').read_bytes() != frozen or
                (output / 'training_stats.json').read_bytes() != frozen):
            raise ValueError('冻结训练统计的内容或报告哈希不一致')
        outputs[label] = {'path': str(output), 'splits': report['splits'],
                          'normalization_sha256': checksum}
    write(root / 'dataset_release.json', {'status': 'verified', 'datasets': outputs,
          'independent_test_usage': 'final_evaluation_only; excluded_from_development',
          'training_started': False, 'uploaded': False})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--old-split', type=Path, required=True)
    parser.add_argument('--convert-root', type=Path)
    parser.add_argument('--finalize-conversion-only', action='store_true')
    parser.add_argument('--normalization-root',type=Path,help='扩充训练数据时继承原模型冻结统计')
    args = parser.parse_args()
    if args.finalize_conversion_only:
        if args.convert_root is None:
            parser.error('--finalize-conversion-only 需要 --convert-root')
        finalize(args.root, args.convert_root)
    else:
        prepare(args.root, args.old_split)
    if args.convert_root is not None and not args.finalize_conversion_only:
        convert(args.root, args.convert_root,args.normalization_root)
