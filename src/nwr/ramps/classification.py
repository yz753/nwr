from pathlib import Path
import traceback
import pynapple as nap

def classify(task, output_dir):
    import matplotlib
    matplotlib.use('Agg')
    import pandas as pd
    from functools import partial
    from pynwb import NWBHDF5IO
    from pynts.tuning_scores import ramps, spatial_information
    from pynts import wrappers
    from nwr.io.nwb import get_ramps_input
    from nwr.ramps import output as ramps_util
    from nwr.ramps import spatial_information as si_util

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
        experiment = task['experiment'] # this should be named as NWR1/NWR2..., it defines the cohort
        mouse_id, day = task['mouse'], task['day']
        
        training_or_test = single_unit_beh_dict['training_or_test']
        bounds = (0, 200)
        reward_position = (90, 110)
        outbound = (30, 90)
        homebound = (110, 170)
        num_bins_ramps = 200
        num_bins_si = 80
        n_shuffles = 500
        context = 'rz'
        
        trial_types = [[0], [1]]
        test_blocks = [[0], [1], [2]] if training_or_test == 'test' else [None]
        rows, errors = [], []
        for trial_type in trial_types:
            for test_block in test_blocks:
                try:
                    print('Start processing:', experiment, mouse_id, day, task['session_type'], ', unit_id:', unit_id, ', trial_type:', trial_type[0], ', test_block:', test_block[0], flush=True)
                    
                    ramps_tuning_fn = partial(
                        ramps.compute_ramps, 
                        range=bounds, 
                        context=context,
                        trial_types=trial_type, 
                        test_blocks=test_block,
                        outbound=outbound, 
                        homebound=homebound, 
                        num_bins=num_bins_ramps,
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
                        num_bins=num_bins_si, 
                        range=bounds, 
                        trial_types=trial_type, 
                        test_blocks=test_block,
                        epoch=None,
                        # default smooth sigma = 2
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
                        'nwb_path': task['nwb_path'],
                        'experiment': experiment, 'mouse': mouse_id, 'day': day,
                        'unit_id': unit_id, 'session': task['session_type'],
                        'trial_type': int(trial_type[0]), 'test_block': int(test_block[0]),
                        'reward_position': reward_position,
                        **ramps_util.make_csv(ramps_results_null), 
                        **si_util.make_csv(si_results_null),
                    })
                    
                    print('Finished processing:', experiment, mouse_id, day, task['session_type'], ', unit_id:', unit_id, ', trial_type:', trial_type[0], ', test_block:', test_block[0], flush=True)

                except Exception:
                    error = {'trial_type': int(trial_type[0]), 'test_block': int(test_block[0]),
                            'traceback': traceback.format_exc()}
                    errors.append(error)
                    print(error['traceback'], flush=True)
                    
        if rows:
            tmp = output_dir / f'{experiment}_{mouse_id}_{day}_unit{unit_id}_results.csv.tmp'
            pd.DataFrame(rows).to_csv(tmp, index=False)
            tmp.replace(output_dir / f'{experiment}_{mouse_id}_{day}_unit{unit_id}_results.csv')
            print('Results saved to:', output_dir / f'{experiment}_{mouse_id}_{day}_unit{unit_id}_results.csv', flush=True)
        return rows, errors, len(trial_types) * len(test_blocks), 'complete' if not errors else 'failed'

