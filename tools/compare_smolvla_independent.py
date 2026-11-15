"""Compare matched saved predictions, including held-cup and transition errors."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'wheel_arm_manipulation'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from wheel_arm_manipulation.smolvla_coordination import TCPKinematics, coordination_metrics, summarize_coordination
from wheel_arm_manipulation.task_phases import parse_phases, phase_at
from compare_smolvla_horizons import compare, load


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def metrics(errors):
    values = np.asarray(errors)
    return {'action_comparisons': len(values), 'joint_mae_rad': abs(values[:, :4]).mean(axis=0).tolist(),
            'gripper_mae_mm': float(abs(values[:, 4]).mean()*1000),
            'gripper_bias_mm': float(values[:, 4].mean()*1000),
            'gripper_p95_abs_mm': float(np.percentile(abs(values[:, 4]), 95)*1000)}


def comparison(old, new, dataset, kinematics):
    horizons = compare(old, new)  # Rejects differing inputs, targets, starts or phases.
    old_rows, new_rows = load(old), load(new)
    fk = TCPKinematics(read(kinematics))
    sources = [s for s in read(dataset/'source_manifest.json')['sources'] if s['split'] == 'validation']
    details, eligible = {}, []
    for source in sources:
        root = Path(source['source_episode'])
        events_path = root/'task_events.jsonl'
        if hashlib.sha256(events_path.read_bytes()).hexdigest() != source['source_file_sha256']['task_events.jsonl']:
            raise ValueError('独立测试阶段事件已变化')
        events = rows(events_path)
        phases = parse_phases(events, source['task_id'])
        holding = next(e['stamp_ns'] for e in events if e['event'] == 'hold_started')
        held_end = next(e['stamp_ns'] for e in events if e['event'] == 'hold_verified')
        grid = rows(root/'grasp_hold_available_time_grid.jsonl')
        closing = [r['action'] for r in grid if r['status'] == 'accepted' and
                   phase_at(r['action_stamp_ns'], phases) == 'close']
        closing = np.asarray(closing)
        details[source['task_id']] = {'phases': phases, 'hold': (holding, held_end),
            'plane': float(np.median(fk.positions(closing)[:, 2])),
            'closed': float(np.percentile(closing[:, 4], 10))}
    for key, row in old_rows.items():
        spec = details[row['source_task_id']]
        if (row['start_phase'] == 'close' and
                abs(fk.positions(row['state_physical'])[2]-spec['plane']) <= .005):
            ref = coordination_metrics(row['target_physical'], row['target_physical'],
                                      row['state_physical'], fk, spec['plane'], spec['closed'])
            if ref['reference_closed_before_lift'] is True:
                eligible.append(key)
    result = {'matched_starts': len(old_rows), 'independent_recordings': len(sources),
              'scope': 'matched_teacher_forced_offline_predictions_not_closed_loop_success',
              'horizons': horizons, 'models': {}}
    for label, model_rows in [('old', old_rows), ('new', new_rows)]:
        errors, coordination = {}, []
        out_of_range = 0
        for key, row in model_rows.items():
            spec = details[row['source_task_id']]
            prediction, target = np.asarray(row['prediction_physical']), np.asarray(row['target_physical'])
            out_of_range += int(((prediction[:, 4] < 0) | (prediction[:, 4] > .08)).sum())
            for i, error in enumerate(prediction-target):
                stamp = row['source_stamp_ns']+(i+1)*100_000_000
                phase = phase_at(stamp, spec['phases'])
                if spec['hold'][0] <= stamp < spec['hold'][1]:
                    phase = 'stable_hold'
                elif phase == 'lift':
                    phase = 'lift_before_hold'
                errors.setdefault(phase, []).append(error)
                errors.setdefault('all', []).append(error)
            if key in eligible:
                coordination.append({'source_task_id': key[0], 'dataset_index': key[1],
                    'coordination': coordination_metrics(prediction, target, row['state_physical'],
                                                        fk, spec['plane'], spec['closed'])})
        result['models'][label] = {'physical_errors': {k: metrics(v) for k, v in errors.items()},
            'gripper_out_of_physical_range_action_comparisons': out_of_range,
            'coordination': summarize_coordination(coordination), 'transition_windows': coordination}
    return result


def plot_transitions(result, old, new, kinematics, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    a, b = load(old), load(new)
    fk = TCPKinematics(read(kinematics))
    tasks = sorted({task for task, _ in a})
    fig, axes = plt.subplots(2, len(tasks), figsize=(6*len(tasks), 7), squeeze=False)
    for column, task in enumerate(tasks):
        candidates = [r for r in result['models']['old']['transition_windows'] if r['source_task_id'] == task]
        if not candidates:
            continue
        # Choose by reference closing range, before inspecting either prediction.
        chosen = max(candidates, key=lambda r: np.ptp(np.asarray(a[(task, r['dataset_index'])]['target_physical'])[:, 4]))
        key = task, chosen['dataset_index']
        plane = chosen['coordination']['grasp_plane_z_m']
        times = np.arange(1, 51)/10
        for label, trajectory, style in [('reference', a[key]['target_physical'], 'k-'),
                                         ('old model', a[key]['prediction_physical'], 'C1--'),
                                         ('new model', b[key]['prediction_physical'], 'C0-')]:
            trajectory = np.asarray(trajectory)
            axes[0, column].plot(times, trajectory[:, 4]*1000, style, label=label)
            axes[1, column].plot(times, (fk.positions(trajectory)[:, 2]-plane)*1000, style, label=label)
        axes[0, column].set_title(f'Independent recording {column+1}, start {key[1]}')
        axes[0, column].set_ylabel('Gripper reference opening (mm)')
        axes[1, column].set_ylabel('Predicted TCP rise (mm)')
        axes[1, column].axhline(10, color='gray', linewidth=.8, linestyle=':')
        axes[1, column].set_xlabel('Physical future offset (s)')
        for ax in axes[:, column]:
            ax.grid(alpha=.25)
            ax.legend()
    fig.suptitle('Same held-out inputs; raw predictions, no execution compensation')
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('old', type=Path)
    parser.add_argument('new', type=Path)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--kinematics', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = comparison(args.old, args.new, args.dataset, args.kinematics)
    with args.output.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    plot_transitions(result, args.old, args.new, args.kinematics, args.output.with_suffix('.png'))
    for label, report in result['models'].items():
        print(label, {k: round(v['gripper_mae_mm'], 3) for k, v in report['physical_errors'].items()},
              report['coordination'], flush=True)
