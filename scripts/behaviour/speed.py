# Allow direct execution with the existing local/Eddie environment.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from nwr.behaviour.speed import (
    Mouse_Info,
    Path,
    generate_speed_position_plots,
    generate_trial_speed_position_heatmaps,
    nap,
    np,
    pd,
    plt,
    re,
    sys,
    xr,
)

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
