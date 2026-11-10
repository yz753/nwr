'''save decode results & make plots'''
import traceback
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_trial_rmse(summary, path):
    summary = summary.sort_values("trial").reset_index(drop=True)
    x = np.arange(len(summary))
    blocks = summary.test_block.to_numpy()
    boundaries = np.flatnonzero(blocks[1:] != blocks[:-1]) + 1
    fig, ax = plt.subplots(figsize=(12, 4))
    try:
        for idx in np.split(x, boundaries):
            ax.plot(idx, summary.loc[idx, "rmse_cm"], "o-", ms=4, color="C0")
        for boundary in boundaries:
            ax.axvline(boundary - .5, color="grey", ls="--", lw=1)
        ticks = x[::max(1, int(np.ceil(len(x) / 30)))]
        ax.set_xticks(ticks)
        ax.set_xticklabels(summary.loc[ticks, "trial"].astype(int), rotation=90)
        ax.set(xlabel="Test trial ID (equally spaced)", ylabel="RMSE (cm)",
               title=Path(path).stem)
        fig.tight_layout()
        fig.savefig(path, dpi=200)
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
                train_trial_ids=run["train_trial_ids"],
                test_trial_ids=run["test_trial_ids"],
                **{key: value for key, value in metadata.items() if key not in samples.columns},
            )
            summaries.append(run["summary"])
            plot_trial_rmse(run["summary"], folder / f"{prefix}_trial_rmse.png")
        except Exception as exc:
            error = {
                "info": metadata,
                "traceback": traceback.format_exc(),
            }
            print(error["traceback"], flush=True)  
    if summaries:
        pd.concat(summaries, ignore_index=True).to_csv(folder / "summary.csv", index=False)
