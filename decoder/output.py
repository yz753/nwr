'''save decode results & make plots'''
import traceback
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_trial_rmse(summary, path, metadata):
    summary = summary.sort_values("trial").reset_index(drop=True)
    x = np.arange(len(summary))
    blocks = summary.test_block.to_numpy()
    boundaries = np.flatnonzero(blocks[1:] != blocks[:-1]) + 1
    fig, ax = plt.subplots(figsize=(12, 4))
    try:
        for idx in np.split(x, boundaries):
            ax.plot(idx, summary.loc[idx, "rmse_cm"], "o-", ms=4, color="black")
            
        for boundary in boundaries:
            ax.axvline(boundary - .5, color="blue", ls="--", lw=1)

        ticks = x[::max(1, int(np.ceil(len(x) / 30)))]
        ax.set_xticks(ticks)
        ax.set_xticklabels(summary.loc[ticks, "trial"].astype(int), rotation=90)
        ax.set(xlabel="test trial idx", ylabel="RMSE (cm)",
               title=f'RMSE for M{metadata["mouse"]} D{metadata["day"]} trial_type{metadata["trial_type"]} {metadata["decoder"]} decoder')
        fig.tight_layout()
        fig.savefig(path, dpi=200)
    finally:
        plt.close(fig)
        
        
def plot_decoded_position(samples, path, metadata, block_change_time):
    samples = samples.sort_values("time").reset_index(drop=True)
    bin_size = metadata['bin_size']

    fig, ax = plt.subplots(figsize=(30, 4))
    
    try:
        new_segment = (
            samples["trial"].ne(samples["trial"].shift())
            | samples["time"].diff().gt(1.5 * bin_size)
        )

        for i, (_, segment) in enumerate(
            samples.groupby(new_segment.cumsum(), sort=False)
        ):
            ax.plot(
                segment["time"], segment["true_position"],
                color="black", linewidth=0.8,
                label="real position" if i == 0 else None,
            )

        ax.scatter(
            samples["time"], samples["decoded_position"],
            s=8, color="C1", alpha=0.6, linewidths=0,
            label="decoded position",
        )
        for timestamp in block_change_time:
            ax.axvline(timestamp, color="blue", linestyle="--", linewidth=1,)
    
        ax.axhspan(90, 110, color='lightgreen', alpha=0.5)
        ax.axhspan(0, 30, color='lightgrey', alpha=0.5)
        ax.axhspan(170, 200, color='lightgrey', alpha=0.5)
        
        ax.set(
            xlabel="time",
            ylabel="position",
            title=(
                f'M{metadata["mouse"]} D{metadata["day"]} '
                f'trial_type {metadata["trial_type"]} '
                f'{metadata["scheme"]} {metadata["decoder"]}'
            ),
        )
        ax.legend(bbox_to_anchor = (1, 1), frameon=False)
        fig.tight_layout()
        fig.savefig(path, dpi=400)
    finally:
        plt.close(fig)
    
        
def save_and_plot(result, output_dir):
    """Write one decode() result. Array workers must pass unique output dirs."""
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    
    summaries = []
    
    for run in result["runs"]:
        metadata = run["metadata"]
        prefix = (
            f'M{metadata["mouse"]}_D{metadata["day"]}'
            f'_trial_type-{metadata["trial_type"]}'
            f'_scheme-{metadata["scheme"]}'
            f'_decoder-{metadata["decoder"]}'
        )

        try:
            samples = run["samples"]
            
            np.savez_compressed(folder / f"{prefix}.npz",
                **{c: samples[c].to_numpy() for c in samples},
                probability=run["probability"],
                train_trials=run["train_trials"],
                test_trials=run["test_trials"],
                **{key: value for key, value in metadata.items() if key not in samples.columns},
            )
            summaries.append(run["summary"])
            plot_decoded_position(samples, folder / f"{prefix}_decoded_position.png", metadata, run["block_change_time"])
            plot_trial_rmse(run["summary"], folder / f"{prefix}_trial_rmse.png", metadata)
        except Exception as exc:
            error = {
                "info": metadata,
                "traceback": traceback.format_exc(),
            }
            print(error["traceback"], flush=True)  
    if summaries:
        pd.concat(summaries, ignore_index=True).to_csv(folder / "summary.csv", index=False)
