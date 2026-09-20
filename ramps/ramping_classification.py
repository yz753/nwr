"""Run all trial/track combinations for one NWB unit selected by manifest row."""
import argparse
import json
from pathlib import Path
import time
import traceback
import pynapple as nap


def atomic_json(path, obj):
    tmp = path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(obj, indent=2))
    tmp.replace(path)


def get_task(run_dir, task_id):
    if task_id < 1:
        raise ValueError('task_id must be positive')
    with (run_dir / 'tasks.jsonl').open() as f:
        for index, line in enumerate(f, 1):
            if index == task_id:
                task = json.loads(line)
                if task['task_id'] != task_id:
                    raise ValueError('Manifest row/task ID mismatch')
                return task
    raise ValueError(f'No manifest row {task_id}')


def classify(task, output_dir):
    import matplotlib
    matplotlib.use('Agg')
    import pandas as pd
    from functools import partial
    from pynwb import NWBHDF5IO
    from pynts.tuning_scores import ramps, spatial_information
    from pynts import wrappers
    from nwb_ramps_input import get_ramps_input
    import ramps_util
    import si_util

    path = Path(task['nwb_path'])
    stat = path.stat()
    if (stat.st_size, stat.st_mtime_ns) != (task['source_size'], task['source_mtime_ns']):
        raise RuntimeError('NWB file changed after manifest creation; prepare a new run')
    
    # ramps should only be run on VR sessions
    if task['session_type'] != 'VR':
        return [], [], 0, 'skipped_non_vr'
    
    unit_id = task['unit_id']
    
    with NWBHDF5IO(str(path), mode="r", load_namespaces=True) as io:
        nwbfile = nap.NWBFile(io.read())
        single_unit_beh_dict = get_ramps_input(nwbfile, unit_id)
        
        cluster = single_unit_beh_dict['cluster']
        experiment = str(path.parent.parent.parent.parent.parent.name) # this should be named as NWR1/NWR2..., it defines the cohort
        mouse_id, day = task['mouse'], task['day']
        
        training_or_test = single_unit_beh_dict['training_or_test']
        bounds = (0, 200)
        reward_position = (90, 110)
        outbound = (0, 90)
        homebound = (110, 200)
        num_bins = 200
        n_shuffles = 500
        context = 'rz'
        
        trial_types = [[0], [1]]
        test_blocks = [[0], [1], [2]] if training_or_test == 'test' else [None]
        rows, errors = [], []
        for trial_type in trial_types:
            for test_block in test_blocks:
                try:
                    ramps_tuning_fn = partial(
                        ramps.compute_ramps, 
                        range=bounds, 
                        context=context,
                        trial_types=trial_type, 
                        test_blocks=test_block,
                        outbound=outbound, 
                        homebound=homebound, 
                        num_bins=num_bins,
                        smooth_sigma='cv', 
                        epoch=None,
                    )
                    ramp_score_null = wrappers.with_null_distribution(
                        ramps_tuning_fn, 
                        ramps.classify_ramps, 
                        n_shuffles=n_shuffles,
                    )
                    ramps_results_null = ramp_score_null(
                        single_unit_beh_dict, 
                        task['session_type'], 
                        cluster
                    )
                    
                    # store tuning curves
                    ramps_util.store_results(
                        ramps_results_null, 
                        experiment, 
                        mouse_id, 
                        day, 
                        unit_id, 
                        output_dir,
                        reward_position, 
                        homebound, outbound, 
                        trial_type, training_or_test, test_block
                        )
                    
                    # spatial information
                    si_tuning_fn = partial(
                        spatial_information.compute_spatial_information,
                        num_bins=None, 
                        range=None, 
                        smooth_sigma=None, 
                        epoch=None,
                    )
                    si_null = wrappers.with_null_distribution(
                        si_tuning_fn, 
                        spatial_information.classify_spatial_information, 
                        n_shuffles=n_shuffles,
                    )
                    si_results_null = si_null(
                        single_unit_beh_dict, 
                        task['session_type'], 
                        cluster
                    )
                    
                    rows.append({
                        'task_id': task['task_id'], 'nwb_path': task['nwb_path'],
                        'experiment': experiment, 'mouse': mouse_id, 'day': day,
                        'unit_id': unit_id, 'session': task['session_type'],
                        'trial_type': int(trial_type[0]), 'test_block': int(test_block[0]),
                        'reward_position': reward_position,
                        **ramps_util.make_csv(ramps_results_null), 
                        **si_util.make_csv(si_results_null),
                    })
                    
                except Exception:
                    error = {'trial_type': int(trial_type[0]), 'test_block': int(test_block[0]),
                            'traceback': traceback.format_exc()}
                    errors.append(error)
                    print(error['traceback'], flush=True)
                    
        if rows:
            tmp = output_dir / 'results.csv.tmp'
            pd.DataFrame(rows).to_csv(tmp, index=False)
            tmp.replace(output_dir / 'results.csv')
        return rows, errors, len(trial_types) * len(test_blocks), 'complete' if not errors else 'failed'


def run_task(run_dir, task_id, classifier=classify):
    task = get_task(run_dir, task_id)
    out = run_dir / 'tasks' / f'{task_id:08d}'
    out.mkdir(parents=True, exist_ok=True)
    status_file = out / 'status.json'
    start = time.monotonic()
    status = {'task': task, 'state': 'running'}
    atomic_json(status_file, status)
    try:
        rows, errors, expected, state = classifier(task, out)
        status.update(state=state, rows=len(rows), expected_rows=expected, errors=errors)
    except Exception:
        status.update(state='failed', rows=0, errors=[{'traceback': traceback.format_exc()}])
        print(status['errors'][0]['traceback'], flush=True)
    status['elapsed_seconds'] = time.monotonic() - start
    atomic_json(status_file, status)
    return 0 if status['state'] in ('complete', 'skipped_non_vr') else 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run_dir', type=Path, required=True)
    p.add_argument('--task_id', type=int, required=True)
    a = p.parse_args()
    return run_task(a.run_dir.resolve(), a.task_id)


if __name__ == '__main__':
    raise SystemExit(main())
