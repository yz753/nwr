from pynwb import NWBHDF5IO
import pynapple as nap
import matplotlib.pyplot as plt
import re, sys
import numpy as np
from pathlib import Path
import pandas as pd
from scipy.stats import poisson
from itertools import product
import json


from nwr.metadata import Mouse_Info, Genotype_Info
from nwr.behaviour.intervals import get_intervalset

'''this uses stops intervalsets to generate plots, and use the stop_locations within reward zone to filt poisson distributions to compare the behaviour changes between genotypes,
the t-test and sample size calculation is in another file.'''

def select_stops_in_rz(stop_positions: nap.Tsd, 
                       trials: nap.IntervalSet, 
                       rz: tuple):
    in_region = (
        (stop_positions.values >= rz[0])
        & (stop_positions.values <= rz[1])
    )
    counts = stop_positions[in_region].count(ep=trials).values
    
    return pd.Series(
        counts.astype(int),
        index = pd.Index(
            np.asarray(trials['number']), name='trial_id'
        ),
            name = 'stop_count',
    )


def evaluate_poisson(y, cuts): # nll is used to select the optimal cut points
    segs = np.split(y, cuts)
    if any(len(seg) < 10 for seg in segs): # there should be at least 10 trials in each segment to fit poisson
        return None
    mu_list = []
    nll = 0.0
    
    for seg in segs:
        mu = seg.mean() # by definition, mu is mean of the samples
        mu_list.append(mu)
        fitted_poisson = poisson(mu=mu)
        nll -= fitted_poisson.logpmf(seg).sum()
    return np.asarray(mu_list), nll


def fit_poisson(stop_series: pd.Series,
                block_start_trials):
    trials = stop_series.index.to_numpy()
    y = stop_series.to_numpy()
    
    # define the window of cut points to fit 3 poisson distributions
    switch_window_0_to_1 = np.flatnonzero(
        (trials >= block_start_trials[1]) & (trials < block_start_trials[2])
    )
    switch_window_1_to_2 = np.flatnonzero(
        (trials >= block_start_trials[2])
    )
    
    # iterate over cuts
    best_nll = np.inf
    best_result = None
    
    for switch0, switch1 in product(switch_window_0_to_1, switch_window_1_to_2):
        cuts = [switch0, switch1]
        result = evaluate_poisson(y, cuts)
        if result is None:
            continue
        mu_list, nll = result
        if nll < best_nll: # for nll, the lower the better
            best_nll = nll
            best_result = (switch0, switch1, mu_list)

    return best_result


def plot_and_compare_stops(
    nwb_path,
    output_dir,
    test_days_dict,
    trial_type_colour_palette_dict,
    rz: tuple
):
    nwb_path = Path(nwb_path)
    match = re.search(
        r"sub-M(\d+)_ses-D(\d+)(?=_|$)",
        nwb_path.stem,
    )
    if match is None:
        raise ValueError(f"Cannot identify mouse/day: {nwb_path.name}")
    mouse, day = map(int, match.groups())
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    with NWBHDF5IO(nwb_path, "r", load_namespaces=True) as io:
        nwb = nap.NWBFile(io.read())
        stops = nwb['stationary'] # intervalset
        P = nwb['P'] # Tsd
        trial_number = nwb['trial_number'] # Tsd
        block_idx = nwb['block_idx'] # Tsd
        trial_type = nwb['trial_type'] # Tsd
        trials = nwb['trials'] # intervalset # used for defining episodes
        
        stop_positions = P.restrict(stops)
        stop_trials = stop_positions.value_from(trial_number)
        stop_trial_types = stop_positions.value_from(trial_type)
        
        thresholding_0_1_2 = [
            (None, 0.5),
            (0.5, 1.5),
            (1.5, None),
        ]
        test_blocks = get_intervalset(
            block_idx,
            thresholding_0_1_2,
        )
        block_start_trials = nap.Ts(
            t=test_blocks.start,
        ).value_from(
            trial_number, mode='after'
        ).values.astype(int)

        # compare between genotypes
        # this uses number of stops from the intervalset instead of simply stop_positions, because a long stop could contribute to much much much more data points
        stop_onset_positions = nap.Ts(
            t=stops.start,
            time_support=trials,
        ).value_from(P, ep=trials)
        stop_series = select_stops_in_rz(stop_onset_positions, trials, rz)
        poisson_result = fit_poisson(
            stop_series,
            block_start_trials)
        
        # store the difference between poisson results & manually defined first trial for each test block switch -> used for t-test
        switch0_diff = int(stop_series.index[poisson_result[0]]) - int(block_start_trials[1])
        switch1_diff = int(stop_series.index[poisson_result[1]]) - int(block_start_trials[2])
        
        # plot
        ax.axvspan(90, 110, color="lightgreen", alpha=0.3)
        ax.axvspan(0, 30, color="grey", alpha=0.3)
        ax.axvspan(170, 200, color="grey", alpha=0.3)
        ax.scatter(
            stop_positions.values, 
            stop_trials.values,
            c=[
                trial_type_colour_palette_dict[value]
                for value in stop_trial_types.values
            ],
            marker='|',
            s=7,
            # alpha=0.5,
        )
        # make y-axis inverted so that trial 0 is on top
        max_trial = np.nanmax(trial_number)
        ax.set_ylim(max_trial + 0.5, -0.5)
        # add 4 vertical lines for test days
        mouse_test_days = {
            day
            for days in test_days_dict.get(f"M{mouse}", {}).values()
            for day in days
        }
        if day in mouse_test_days:
            for (i, block_start) in enumerate(block_start_trials[1:]):
                plt.axhline(
                    block_start,
                    color='red',
                    linestyle='--',
                    linewidth=1,
                    label='test block switch' if i == 0 else None
                )
            for (i, br) in enumerate((int(stop_series.index[poisson_result[0]]), int(stop_series.index[poisson_result[1]]))):
                plt.axhline(
                    br,
                    color='blue',
                    linestyle='--',
                    linewidth=1,
                    label='fitted poisson distribution switch' if i == 0 else None
                )
            plt.legend()
            
        output = output_dir / f"M{mouse}_D{day}.png"
        plt.savefig(output, dpi=200)
        print(f"Saved {output}", flush=True)
        plt.close()
        
    return switch0_diff, switch1_diff, mouse, day
            

    
