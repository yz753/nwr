# Allow direct execution with the existing local/Eddie environment.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from nwr.ramps.comparison import (
    Genotype_Info,
    MakingPlot,
    Path,
    get_tc,
    json,
    np,
    pd,
    plt,
    redirect_stdout,
    sys,
    warnings,
)

if __name__ == "__main__":
    results_root = Path("/exports/eddie/scratch/s2155699/ephys/ramps/ramps_results")
    (results_root / 'figs').mkdir(parents=True, exist_ok=True)
    (results_root / 'stats').mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(results_root / "ramps_classification.csv")
    df[['tc', 'location']] = df.apply(get_tc, axis=1, args=(results_root,))
    
    plotter = MakingPlot()
    
    for test_block in [0,1,2]:
        for single_trial_type in [0,1]:
            # plot separately for each mouse
            for mouse in df['mouse'].unique().tolist():
                subdf_mouse = df[(df['trial_type'] == single_trial_type) & (df['test_block'] == test_block) & (df['mouse'] == mouse)]
    
                # plot
                print('Making figures... Trial_type:', single_trial_type, ', test_block:', test_block, ', mouse_id:', mouse, flush=True)
                fig = plotter.plot_ramp(
                    subdf_mouse,
                    trial_type=single_trial_type,
                    test_block=test_block,
                )
                fig.savefig(results_root / 'figs' / f'ramping_cells_{mouse}_trial_type-{single_trial_type}_test_block-{test_block}.png')
                plt.close(fig)
                
            # combine all mice
            subdf = df[(df['trial_type'] == single_trial_type) & (df['test_block'] == test_block)]
            print('Making figures... Trial_type:', single_trial_type, ', test_block:', test_block, ', all mice', flush=True)
            fig = plotter.plot_ramp_by_genotype(
                subdf,
                trial_type=single_trial_type,
                test_block=test_block,
            )
            fig.savefig(results_root / 'figs' / f'ramping_cells_trial_type-{single_trial_type}_test_block-{test_block}.png')
            plt.close(fig)
