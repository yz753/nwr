# Allow direct execution with the existing local/Eddie environment.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from nwr.behaviour.licks import (
    Mouse_Info,
    NWBHDF5IO,
    Path,
    generate_lick_plots,
    nap,
    np,
    plt,
    re,
    sys,
)

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
