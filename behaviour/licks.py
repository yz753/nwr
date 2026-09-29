from pathlib import Path
import matplotlib.pyplot as plt
import pynapple as nap
from pynwb import NWBHDF5IO
import numpy as np
import re
from util import Mouse_Info

'''raster plot of licks for each trial against position'''

def generate_lick_plots(
    nwb_list,
    output_dir,
    test_days_dict,
    trial_type_colour_palette_dict,
):
    for nwb_path in nwb_list:
        match = re.search(
            r"sub-M(\d+)_ses-D(\d+)(?=_|$)",
            nwb_path.stem,
        )
        if match is None:
            raise ValueError(f"Cannot identify mouse/day: {nwb_path.name}")
        mouse, day = map(int, match.groups())
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        with NWBHDF5IO(nwb_path, "r", load_namespaces=True) as io:
            nwb = nap.NWBFile(io.read())
            lick = nwb['lick'] # Tsd
            P = nwb['P'] # Tsd
            trial_number = nwb['trial_number'] # Tsd
            block_idx = nwb['block_idx'] # Tsd
            trial_type = nwb['trial_type'] # Tsd
            
            # downsample P & trial_number to match lick
            down_P = P.interpolate(
                lick, ep=lick.time_support
            )
            down_trial_number = trial_number.interpolate(
                lick, ep=lick.time_support
            )
            down_trial_type = trial_type.interpolate(
                lick, ep=lick.time_support
            )
            is_lick = np.asarray(lick.values) > 0
            trial_type_values = down_trial_type.values[is_lick].astype(int)

            ax.axvspan(90, 110, color="lightgreen", alpha=0.3)
            ax.axvspan(0, 30, color="grey", alpha=0.3)
            ax.axvspan(170, 200, color="grey", alpha=0.3)
            ax.scatter(
                down_P.values[is_lick],
                down_trial_number[is_lick],
                c=[
                    trial_type_colour_palette_dict[value]
                    for value in trial_type_values
                ],
                marker='|',
                s=10,
            )
            # make y-axis inverted so that trial 0 is on top
            max_trial = np.nanmax(down_trial_number)
            ax.set_ylim(max_trial + 0.5, -0.5)

            mouse_test_days = {
                day
                for days in test_days_dict.get(f"M{mouse}", {}).values()
                for day in days
            }
            if day in mouse_test_days:
                block_values = np.asarray(block_idx.values[:])
                trial_values = np.asarray(trial_number.values[:])

                changes = np.flatnonzero(np.diff(block_values) != 0) + 1

                for i in changes:
                    ax.axhline(
                        float(trial_values[i]) - 1,
                        color="red",
                        linestyle="--",
                        linewidth=1,
                    )
            
            output = output_dir / f"M{mouse}_D{day}.png"
            plt.savefig(output, dpi=200)
            print(f"Saved {output}", flush=True)
            plt.close()
        
        
def main():
    root = Path('/Volumes/INCR-NolanLab/ActiveProjects/Yiming/NWR1/processed')
    
    nwb_list = list(root.rglob('sub*M*D*beh.nwb'))
    
    output_dir = Path('/Volumes/INCR-NolanLab/ActiveProjects/Yiming/NWR1/analysis/licks')
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    test_days_dict, trial_type_colour_palette_dict = Mouse_Info.test_days_dict, Mouse_Info.trial_type_colour_palette_dict
    
    generate_lick_plots(
        nwb_list,
        output_dir,
        test_days_dict,
        trial_type_colour_palette_dict,
    )
    
    
if __name__ == "__main__":
    main()