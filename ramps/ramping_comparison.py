import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from contextlib import redirect_stdout
# from LMM_util import LMM
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import json


def get_tc(df_row, results_root, training_or_test='test'):
    run_dir = Path(results_root/'run_20260920_201338_944198')
    with (run_dir / 'tasks.jsonl').open() as f:
        tasks = [json.loads(line) for line in f if line.strip()]
    matches = [
        task for task in tasks
        if task['nwb_path'] == df_row.nwb_path 
        and int(task['unit_id']) == int(df_row.unit_id)
    ]
    if len(matches) != 1:
        raise ValueError(f'Expected one match for {df_row.nwb_path} and unit {df_row.unit_id}, found {len(matches)}')
    
    task = matches[0]
    task_folder = run_dir / 'tasks' / f"{task['task_id']:08d}"
    test_block = (
        'None' if pd.isna(df_row.test_block) 
        else str(int(df_row.test_block))
    )
    
    fr_filename = (
        f'{task['experiment']}_{training_or_test}_'
        f'{task["mouse"]}_{task["day"]}_unit-{int(task["unit_id"])}_'
        f'trial_type-{int(df_row.trial_type)}_'
        f'test_block-{test_block}.npz'
    )
    fr_path = task_folder / fr_filename
    data = np.load(fr_path)
    
    return pd.Series({
        'tc': np.array(data['tc']),
        'location': np.array(data['location'])
        })
    

class MakingPlot:
    ramp_types = ['++', '+-', '-+', '--']
    # genotypes = ['WT', 'FXS']
    
    # genotype_colour_dict = {
    #     'WT': "#8E8EE0",
    #     'FXS': '#FCB94B'
    # }
        
    def plot_ramp_by_genotype(self, df, trial_type, test_block):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=False)
        axes = axes.ravel()
        
        for ax, rt in zip(axes, self.ramp_types):
            sub = df[df['ramp_type'] == rt]
            if sub.empty:
                ax.set_title(f'{rt} (No Data)')
                continue
            # initiate y vals, this will be used to prevent the y axis from being too large to include extreme outliers but masking out the trends of ramps
            y_vals = []
            
            rz = (90, 110)
            track_length = 200
            ax.axvspan(rz[0], rz[1], color='lightgreen', alpha=0.5, zorder=0)
            ax.axvspan(0, 30, color='lightgrey', alpha=0.5, zorder=0)
            ax.axvspan((track_length-30), track_length, color='lightgrey', alpha=0.5, zorder=0)
            
            # for g in self.genotypes:
            #     sub_g = sub[sub['genotype'] == g]
            #     if sub_g.empty:
            #         continue
                
            mean_tc = np.nanmean(np.stack(sub['tc']), axis=0)
            # get SEM
            sem_tc = np.nanstd(np.stack(sub['tc']), axis=0) / np.sqrt(len(sub))
            
            y_vals.append((mean_tc + sem_tc)[5:-5])  # ignore the first and last 5 bins (the large outliers are often at the edges)
            
            # genotype_colour = self.genotype_colour_dict.get(g)
            ax.plot(
                sub.iloc[0]['location'], 
                mean_tc, 
                # label=g, 
                color='black',
                zorder=3
                )
            ax.fill_between(
                sub.iloc[0]['location'], 
                mean_tc - sem_tc, 
                mean_tc + sem_tc, 
                alpha=0.3,
                color='black',
                zorder=2
            )
            ax.set_title(rt)
            # set y limit for each subplot
            if y_vals:
                ymax = np.nanmax(np.concatenate(y_vals)) * 1.1
                ax.set_ylim(0, ymax)
            
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper right')
        fig.suptitle(f'{test_block}: ramping cells trial type {trial_type}')
        fig.tight_layout()
        
        return fig
    
    
    # def run_LMM(self, df, trial_type, test_block, output_path):
    #     with open(output_path, "w") as f, redirect_stdout(f):
    #         for rt in self.ramp_types:
    #             print('\n--------------------------------------------------------------------')
    #             print('Ramp type:', rt, ', trial type:', trial_type, ', test block:', test_block, '\n')
    #             sub = df[df['ramp_type'] == rt]
    #             # skip the ramp types if there are not enough data points for both genotypes
    #             # if sub.empty or sub['genotype'].nunique() < 2:
    #             #     continue
                
    #             # rewrite the df to have multiple rows for tc, with each row for a single bin
    #             rows = []
    #             for _, r in sub.iterrows():
    #                 for loc, fr in zip(r.location, r.tc):
    #                     rows.append({
    #                         "firing_rate": fr,
    #                         "location": loc,
    #                         "genotype": r.genotype,
    #                         "exp_mouse": f"{r.experimenter}_{r.mouse}",
    #                     })
    #             lmm = pd.DataFrame(rows)
                
    #             # run LMM
    #             # divide the track into outbound & homebound
    #             rz_loc = self.rz_dict.get(test_block)
    #             track_segments = [
    #                 [30, rz_loc[0]],
    #                 [rz_loc[1], (200-30) if 'standard_track' in test_block else (300-30)]
    #             ]
    #             for i, seg in enumerate(track_segments):
    #                 print('\n--------------------------------------------------------------------')
    #                 print('Track segment:', 'outbound' if i == 0 else 'homebound', '\n')
    #                 sub_lmm = lmm[(lmm['location'] >= seg[0]) & (lmm['location'] <= seg[1])]
    #                 # LMM coefficient predicts firing rate at loc=0, so make the start of the segment as loc=0
    #                 sub_lmm = sub_lmm.copy()
    #                 sub_lmm['location_centered'] = sub_lmm['location'] - seg[0]
                    
    #                 # this tests the fitted lines differ in intercepts, slopes, or both between genotypes
    #                 m1 = LMM(
    #                     test_formula = 'firing_rate ~ location_centered * genotype + (1|exp_mouse)',
    #                     null_formula = 'firing_rate ~ location_centered + (1|exp_mouse)',
    #                     df = sub_lmm,
    #                     group = 'exp_mouse',
    #                     REML = True, # REML should be false for comparing the 2 models, but lrt automatically sets it to False
    #                     family = None
    #                 )
    #                 m1.compare()
                    
    #                 print('\n--------------------------------------------------------------------')
    #                 # this tests slope difference only
    #                 m2 = LMM(
    #                     test_formula = 'firing_rate ~ location_centered * genotype + (1|exp_mouse)',
    #                     null_formula = 'firing_rate ~ location_centered + genotype + (1|exp_mouse)',
    #                     df = sub_lmm,
    #                     group = 'exp_mouse',
    #                     REML = True,
    #                     family = None
    #                 )
    #                 m2.compare()
                    
    #                 print('\n--------------------------------------------------------------------')
    #                 # this tests intercept difference only
    #                 m3 = LMM(
    #                     test_formula = 'firing_rate ~ location_centered + genotype + (1|exp_mouse)',
    #                     null_formula = 'firing_rate ~ location_centered + (1|exp_mouse)',
    #                     df = sub_lmm,
    #                     group = 'exp_mouse',
    #                     REML = True,
    #                     family = None
    #                 )
    #                 m3.compare()


# compare ramping cells between FXS and WT, compare ramping interneurons between FXS and WT
if __name__ == "__main__":
    results_root = Path("/exports/eddie/scratch/s2155699/ephys/ramps/ramps_results")
    (results_root / 'figs').mkdir(parents=True, exist_ok=True)
    (results_root / 'stats').mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(results_root / "ramps_classification.csv")
    df[['tc', 'location']] = df.apply(get_tc, axis=1, args=(results_root,))
    
    plotter = MakingPlot()
    
    for test_block in [0,1,2]:
        for single_trial_type in [0,1]:
            subdf = df[(df['trial_type'] == single_trial_type) & (df['test_block'] == test_block)]

            # plot
            print('Making figures... Trial_type:', single_trial_type, ', test_block:', test_block, flush=True)
            fig = plotter.plot_ramp_by_genotype(
                subdf,
                trial_type=single_trial_type,
                test_block=test_block,
            )
            fig.savefig(results_root / 'figs' / f'ramping_cells_trial_type-{single_trial_type}_test_block-{test_block}.png')
            plt.close(fig)
            
            # mixed effect: test the 
            # plotter.run_LMM(
            #     subdf,
            #     trial_type=single_trial_type,
            #     task_type=task_type,
            #     output_path=results_root / 'stats' / f'lmm_WTvsFXS_trial_type_{single_trial_type}_{task_type}.txt')