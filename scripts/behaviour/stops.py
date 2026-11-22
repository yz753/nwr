# Allow direct execution with the existing local/Eddie environment.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from nwr.behaviour.stops import (
    Genotype_Info,
    Mouse_Info,
    NWBHDF5IO,
    Path,
    evaluate_poisson,
    fit_poisson,
    get_intervalset,
    json,
    nap,
    np,
    pd,
    plot_and_compare_stops,
    plt,
    poisson,
    product,
    re,
    select_stops_in_rz,
    sys,
)

def main():
    root = Path('/Volumes/INCR-NolanLab/ActiveProjects/Yiming/NWR1/processed')
    
    nwb_list = list(root.rglob('sub*M*D*beh.nwb'))
    
    output_dir = Path('/Volumes/INCR-NolanLab/ActiveProjects/Yiming/NWR1/analysis/stops')
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    test_days_dict, trial_type_colour_palette_dict = Mouse_Info.test_days_dict, Mouse_Info.trial_type_colour_palette_dict
    
    switch0_dict, switch1_dict = {}, {}
    
    for nwb_path in nwb_list:
        switch0_diff, switch1_diff, mouse, day = plot_and_compare_stops(
                nwb_path,
                output_dir,
                test_days_dict,
                trial_type_colour_palette_dict,
                rz=(75,110),
            )
        
        switch0_dict.setdefault(f"M{mouse}", {})[f"D{day}"] = int(switch0_diff)
        switch1_dict.setdefault(f"M{mouse}", {})[f"D{day}"] = int(switch1_diff)
    
    # save to a json file
    with open(output_dir / 'block0_to_block1_switch_trial_number.json', 'w') as f:
        json.dump(switch0_dict, f)
    with open(output_dir / 'block1_to_block2_switch_trial_number.json', 'w') as f:
        json.dump(switch1_dict, f)
    print("Saved json files.", flush=True)

if __name__ == "__main__":
    main()
