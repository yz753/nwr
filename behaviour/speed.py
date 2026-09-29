from pathlib import Path
from util import Mouse_Info
import pandas as pd
import re
import pynapple as nap
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np  

'''This generates 2 types of plots for speed: 1. tc-like, x axis: position, y axis: speed (cm/s); 2. headmap, x axis: position, y axis: trials,
and for test days, 3 separate plots will be generate for every blocks.'''

def generate_speed_position_plots(
    csv_list, 
    output_dir,
    test_days_dict
    ):
    # deal with prefix & padded version
    test_days = {
        int(mouse.removeprefix('M')): {
            day for days in conditions.values() for day in days
        }
        for mouse, conditions in test_days_dict.items()
    }
    
    for file in csv_list:
        file = Path(file)
        match = re.search(r"(?:^|_)M(\d+)_D(\d+)(?=_|$)", file.stem)
        if match is None:
            raise ValueError(f"Cannot identify mouse/day: {file.name}")

        mouse, day = map(int, match.groups())
        df = pd.read_csv(file, delimiter=";")

        for trial_type in ("beaconed", "nonbeaconed"):
            assert (
                df[f"{trial_type}_success"].iloc[-1]
                + df[f"{trial_type}_failure"].iloc[-1]
                == df[f"{trial_type}_total_trials"].iloc[-1]
            ), f"Issues with {trial_type} trial number: {file.name}"

        is_test = day in test_days.get(mouse, set())
        test_blocks = [0, 1, 2] if is_test else [None]
        
        trial_types = [1,0] # nb: 1, b: 0
        
        for trial_type in trial_types:
            output_dir_per_trial_type = Path(output_dir) / 'speed_tc' / str(trial_type)
            output_dir_per_trial_type.mkdir(parents=True, exist_ok=True)
            
            sub = df[(df["trial_type"]/10).astype(int) == trial_type]
            for block in test_blocks:
                sub2 = sub if block is None else sub[sub["block_idx"] == block]
                if sub2.empty:
                    print(f"Skipping {file.name}: block {block} has no data")
                    continue
                
                # use the tc function in pynapple to compute mean speed-position relationship
                speed_tsd = nap.Tsd(
                    t=sub2["time"].to_numpy(),
                    d=sub2["vr_running_speed"].to_numpy(),
                )
                position_tsd = nap.Tsd(
                    t=sub2["time"].to_numpy(),
                    d=sub2["position"].to_numpy() * 10,
                )
                
                # add error bars
                n_splits = 5
                full_session = position_tsd.time_support
                split_length = full_session.tot_length() / n_splits
                splits = full_session.split(split_length)
                splits = [
                    split for split in splits
                    if len(position_tsd.restrict(split)) >= 2
                ]
                if len(splits) < 2:
                    print(
                        f"Skipping M{mouse} D{day}, trial type {trial_type}, block {block}: "
                        f"insufficient data"
                    )
                    continue
                
                speed_position_per_split = [
                    nap.compute_tuning_curves(
                        speed_tsd,
                        epochs=split,
                        features=position_tsd,
                        bins=200,
                        range=(0, 200),
                    )[0]
                    for split in splits
                ]
                
                positions = speed_position_per_split[0].coords["0"].values

                speed_position_per_split = np.stack([
                    curve.values for curve in speed_position_per_split
                ])
                
                # mice may already run on the wheel at a very fast speed when the blender file starts, which often causes it to stuck for a sec and leaves a huge spike of speed
                # to solve this, find that outlier and replace with interpolated value
                bin_median = np.nanmedian(speed_position_per_split, axis=0)
                bin_mad = np.nanmedian(
                    np.abs(speed_position_per_split - bin_median),
                    axis=0,
                )
                outliers = (
                    np.abs(speed_position_per_split - bin_median)
                    > np.maximum(4 * 1.4826 * bin_mad, 20)
                )
                for i in range(len(speed_position_per_split)):
                    y = speed_position_per_split[i]
                    bad = outliers[i]
                    good = np.isfinite(y) & ~bad
                    y[bad] = np.interp(
                        positions[bad],
                        positions[good],
                        y[good],
                    )
                
                speed_position_means = np.nanmean(speed_position_per_split, axis=0)
                speed_position_stds = np.nanstd(speed_position_per_split, axis=0)
    
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.axvspan(90, 110, color="lightgreen", alpha=0.3)
                ax.axvspan(0, 30, color="grey", alpha=0.3)
                ax.axvspan(170, 200, color="grey", alpha=0.3)
                ax.plot(positions, speed_position_means, color="black")
                ax.fill_between(
                    positions,
                    speed_position_means - speed_position_stds,
                    speed_position_means + speed_position_stds,
                    color="black",
                    alpha=0.2,
                )
                ax.set(
                    title=f"M{mouse} D{day} Trial type {trial_type} Test block {block}",
                    xlabel="Position",
                    ylabel="Mean speed across trials",
                    xlim=(0, 200),
                )
                fig.tight_layout()
    
                output = output_dir_per_trial_type / f"M{mouse}_D{day}_trial_type-{trial_type}_test_block-{block}.png"
                fig.savefig(output, dpi=200)
                plt.close(fig)
                print(f"Saved: {output}")


def generate_trial_speed_position_heatmaps(
    csv_list, 
    output_dir,
    test_days_dict
):
    test_days = {
        int(mouse.removeprefix('M')): {
            day for days in conditions.values() for day in days
        }
        for mouse, conditions in test_days_dict.items()
    }
    
    for file in csv_list:
        file = Path(file)
        match = re.search(r"(?:^|_)M(\d+)_D(\d+)(?=_|$)", file.stem)
        if match is None:
            raise ValueError(f"Cannot identify mouse/day: {file.name}")

        mouse, day = map(int, match.groups())
        df = pd.read_csv(file, delimiter=";")
        
        # the trial number stats in blender csv file might be broken. Regenerate it using the timepoints when position resets
        position = df["position"].to_numpy() * 10
        resets = np.r_[False, np.diff(position) < -100]
        df["trial_idx"] = np.cumsum(resets)
        
        is_test = day in test_days.get(mouse, set())
        test_blocks = [0, 1, 2] if is_test else [None]
        
        trial_types = [1,0] # nb: 1, b: 0
        
        for trial_type in trial_types:
            output_dir = Path(output_dir)
            output_dir_per_trial_type = output_dir / 'heatmap' / str(trial_type)
            output_dir_per_trial_type.mkdir(parents=True, exist_ok=True)
            
            sub = df[(df["trial_type"]/10).astype(int) == trial_type]
            for block in test_blocks:
                sub2 = sub if block is None else sub[sub["block_idx"] == block]
                if sub2.empty:
                    print(f"Skipping {file.name}: block {block} has no data")
                    continue
                
                # bin position
                bin_size = 3
                n_bins = int(200/bin_size)
                sub2['position_cm'] = sub2['position'] * 10
                sub2 = sub2[
                    sub2['position_cm'].between(0, 200, inclusive='both')
                ]
                sub2['position_bin'] = np.minimum(
                    (sub2['position_cm'] // bin_size).astype(int),
                    n_bins - 1,
                )
                
                heatmap = sub2.pivot_table(
                    index='trial_idx',
                    columns='position_bin',
                    values='vr_running_speed',
                    aggfunc='mean',
                ).reindex(columns=np.arange(n_bins))
                
                # similar to the tc plot, remove outliers
                speed_position_by_trial = heatmap.to_numpy().copy()
                valid_bins = np.any(np.isfinite(speed_position_by_trial), axis=0)
                bin_median = np.full(speed_position_by_trial.shape[1], np.nan)
                bin_mad = np.full(speed_position_by_trial.shape[1], np.nan)
                bin_median[valid_bins] = np.nanmedian(
                    speed_position_by_trial[:, valid_bins],
                    axis=0,
                )
                bin_mad[valid_bins] = np.nanmedian(
                    np.abs(
                        speed_position_by_trial[:, valid_bins]
                        - bin_median[valid_bins]
                    ),
                    axis=0,
                )

                outliers = np.zeros_like(speed_position_by_trial, dtype=bool)
                outliers[:, valid_bins] = (
                    np.abs(
                        speed_position_by_trial[:, valid_bins]
                        - bin_median[valid_bins]
                    )
                    > np.maximum(4 * 1.4826 * bin_mad[valid_bins], 20)
                )

                for i in range(len(speed_position_by_trial)):
                    y = speed_position_by_trial[i]
                    bad = outliers[i]
                    good = np.isfinite(y) & ~bad

                    if np.any(bad) and np.sum(good) >= 2:
                        y[bad] = np.interp(
                            heatmap.columns[bad],
                            heatmap.columns[good],
                            y[good],
                        )
                
                fig, ax = plt.subplots(figsize=(10, 7)) if is_test else plt.subplots(figsize=(10, 4))
                image = ax.imshow(
                    speed_position_by_trial,
                    aspect="auto",
                    origin="upper",
                    extent=(0, 200, len(heatmap) - 0.5, -0.5),
                    cmap="Grays",
                )

                ax.axvline(90, color="lightgreen", alpha=1)
                ax.axvline(110, color="lightgreen", alpha=1)

                tick_positions = np.linspace(
                    0, len(heatmap) - 1,
                    min(10, len(heatmap)),
                    dtype=int,
                )

                # because b & nb are separated, there are gaps in trial index
                # left y axis: actual trial index (with gaps)
                actual_trials = heatmap.index.to_numpy()
                ax.set_yticks(tick_positions)
                ax.set_yticklabels(actual_trials[tick_positions])
                ax.set_ylabel("Trial index")

                # right y axis: index after separating beaconed/non-beaconed trials, sorted, no gaps
                right_ax = ax.twinx()
                right_ax.set_ylim(ax.get_ylim())
                right_ax.set_yticks(tick_positions)
                right_ax.set_yticklabels(tick_positions)
                right_ax.set_ylabel("Sorted trial index")

                ax.set(
                    title=f"M{mouse} D{day} Trial type {trial_type} Test block {block}",
                    xlabel="Position (cm)",
                    xlim=(0, 200),
                )

                fig.colorbar(image, ax=[ax, right_ax], label="speed")
                output = output_dir_per_trial_type / f"M{mouse}_D{day}_trial_type-{trial_type}_test_block-{block}_heatmap.png"
                fig.savefig(output, dpi=200, bbox_inches="tight")
                plt.close(fig)
                print(f"Saved: {output}")


def main():
    root = Path('/Volumes/INCR-NolanLab/ActiveProjects/Yiming/NWR1/blender')
    
    csv_list = list(root.glob('YZ*_M*_D*.csv'))
    
    test_days_dict = Mouse_Info.test_days_dict
    
    output_dir = '/Volumes/INCR-NolanLab/ActiveProjects/Yiming/NWR1/analysis/speed_fig'
    
    # generate_speed_position_plots(
    #     csv_list, 
    #     output_dir,
    #     test_days_dict
    # )
    generate_trial_speed_position_heatmaps(
        csv_list, 
        output_dir,
        test_days_dict
    )


if __name__ == "__main__":
    main()
