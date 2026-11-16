# Spikesorting, preprocessing & analysis for NWR1 dataset

## Run spikesorting (EDDIE)
First, all NP recording filepaths should be stored in `nolanlab-ephys/scripts/yiming/yiming_filepaths.csv`. To do this, in `nolanlab-ephys/scripts/yiming`, run
```bash
uv run write_filepaths.py
```
Then `git push`.
On EDDIE, `cd REPO`, `git pull`, and
```bash
uv run sort_on_eddie.py --mice 1,2 --days 1,2 --sessions VR -- protocol kilosort4A
```
This will spikesort M1_D1, M1_D2, M2_D1, M2_D2 separately.

To concatenate VR and OF sessions,
```bash
uv run sort_on_eddie.py --mice 1,2 --days 1,2 --sessions VR,OF -- protocol kilosort4A
```
If analysing for another experiment, change the default values of `--data_folder` and `--derive_folder`.

## Run DeepLabCut (EDDIE)
Again, first, in `nolanlab-dlc`, run
```bash
uv run write_filepaths.py
```
Cropping regions for tongue and eye are manually defined. To do this, open avi files in IINA, then Video -> Crop -> Custom. The blue numbers in bottom left corner are top left coords, width, and height of the cropping region. These will be the x,y,w,h values in `nolanlab-dlc\yiming_crops\tongue_crops_yiming.csv` or `eye_crops_yiming.csv`. Update the csvs accordingly. Then `git push` all three files.

On EDDIE, `cd REPO`, `git pull`, and
```bash
uv run pose_estimation_on_eddie.py --mice=1,2 --days=1,2 --sessions=VR,OF --bodyparts=eye,tongue
```
The logic is the same as `sort_on_eddie.py` except that VR and OF are not concantenated.

If analysing for another experiment, change the default values of `--data_folder` and `--derive_folder`.

The DLC models used by the scripts were trained by Harry. They need be copied from `/exports/cmvm/datastore/sbms/groups/INCR-NolanLab/ActiveProjects/Yiming/NWR1/video/models` to `/exports/eddie/scratch/s2155699/ephys/video/models` if not present on EDDIE before running the analysis.

Both scripts automatically stage in and out files before and after running the core analyses.

## Fine tune licks (LOCAL MACHINE)
The DLC results are suboptimal. To reduce false positive, on local machine, `cd lick_detection` then 
```bash
uv run main.py --mouse 1 --day 1
```
On the pop-out window, draw points on the figure to crop the tongue region. This will create a lick_mask.csv and an analysing figure in the `dlc_output_tongue` folder.

## Bombcell curation, syncing data (EDDIE)
For the scripts to run without error, `labels/mice_information/mouse_data.csv` and `metadata/generic_metadata.yml` should be created first in `/Volumes/INCR-NolanLab/ActiveProjects/Yiming/NWR1/ephys/derivatives`. 

First, stage in all necessary files.
```bash
qsub scripts/preprocessing/HPC/stagein.sh M1 kilosort4A VR D1 D2 D3 ...
```
Unfortunately, my current `stagein.sh` does not accept processing multiple mice.

Then, run the analyses. It contains 2 parts: `sorting_sa.py` doing bombcell curation, `behaviour.py` syncing bonsai, blender, ephys, and writing everything into nwb. For now, the nwb file contains CURATED (good and mua) units (`nwb['units']`). To be able to do this, `sorting_sa.py` needs to run before `behaviour.py`, which is the case in `preprocess.sh`.

To run the analyses,
```bash
qsub -N preprocess_M1_D1   -v STORAGE=/exports/eddie/scratch/s2155699/ephys/NWR1_preprocessing_data   scripts/preprocessing/HPC/preprocess.sh   kilosort4A M1 D1
```
Don't change $STORAGE unless the destination in `stagein.sh` has been changed!

To stage out processed files,
```bash
qsub scripts/preprocessing/HPC/stageout.sh M1 VR D1 D2 D3 ...
```
The nwb that has been created contains: units (TsGroup), trials (IntervalSet), trial_type (Tsd), lick (Tsd), S (Tsd), P (Tsd)...

To access (and modify) it,
```
from pynwb import NWBHDF5IO
import pynapple as nap

io = NWBHDF5IO(
    "nwb_filepath",
    mode="r",
)

example_nwb = nap.NWBFile(io.read())
units = example_mwb['units']
```

To read and modify its metadata,
```
example_nwb = io.read()
print(example_nwb.subject)
```

## Data structure (DATASTORE)
In order for all the scripts to run without error, both raw and processed data needs to follow certain structure and names.

Raw data only:
```
project_folder/
    blender/
        YZ_date_M*_D*.csv
    ephys/
        raw/
            session_type/
                M*_D*_date_recordingsite_sessiontype/
                    Record Node */
                        settings.xml
                        experiment*/
    video/
        data/
            session_type/
                M*_D*_side_capture_sessiontype.avi
                M*_D*_side_capture_sessiontype.csv
```
`project_folder` is `/Volumes/INCR-NolanLab/ActiveProjects/Yiming/NWR1`, or `/exports/cmvm/datastore/sbms/groups/INCR-NolanLab/ActiveProjects/Yiming/NWR1` on EDDIE.

After processing:
```
project_folder/
    blender/
        YZ_date_M*_D*.csv
    ephys/
        raw/
            session_type/
                M*_D*_date_recordingsite_sessiontype/
                    ...
        derivatives/
            mouse_id/
                day/
                    session_type/
                        M*_D*_probe_layout.png
                        kilosort4A/
                            recording_quality_plots/
                            sub-*_day-*_ses-*_srt-*_analyzer/
            labels/
                mice_information/
                    mouse_data.csv
            metadata/
                generic_metadata.yml
    video/
        data/
            session_type/
                ...
        derivatives/
            mouse_id/
                day/
                    session_type/
                        dlc_output_eye/
                        dlc_output_tongue/
                            M*_D*_lick_mask.csv
                            ...
        models/
    processed/
        mouse_id/
            day/
                session_type/
                    *.nwb
                    *curation.json
                    *analyzer.zarr/
                    sync/
```

## Upload to Dandi (LOCAL MACHINE)
The nwb files are then uploaded to an online Dandi dataset in a flattened data structure of `mouse_id\*.nwb`.

First, create a new Dandi dataset on the website if hasn't done so.

Then, on local machine, `cd nwr`, then
```bash
bash upload_nwb_to_dandi.sh DANDI_DATASET_ID(6 digits) \
'/Volumes/INCR-NolanLab/ActiveProjects/Yiming/NWR1/processed' \
'/Volumes/INCR-NolanLab/ActiveProjects/Yiming/NWR1/dandi_upload'
```
This will create a folder `/Volumes/INCR-NolanLab/ActiveProjects/Yiming/NWR1/dandi_upload`.

## Further analyses
First, `cd nwr`.

### 1. Behaviour (LOCAL MACHINE)
To compute speeds, stops, and licks with position, run all in `behaviour/` except from `util.py`. This will plot figures separately per mouse per day per trial_type. For test days, reward availability (test_blocks) is plotted independently. `stops.py` also compared the number of trials required by each genotype to realise the transition between test_blocks. The threshold used to classify stops/stationary timestamps is 3cm/s (from mecpy), it's different to the 5cm/s used in Junji & Teris' data.

### 2. Neural activity heatmap (EDDIE)
Before doing any analyses on ephys data, we should first create an inventory of all the units in all the nwb files. Only after this we can assign parallel tasks to each unit. To do this,
```bash
qsub units/prepare_nwb_units.sh
```
This creates `nwb_units.jsonl` in working directory.

Then, to generate heatmaps for each session,
```bash
```

### 3. Ramping classification and spatial information analysis (EDDIE)
Ramping cells are classified based on their firing rate gradient before (outbound) and after (homebound) reward zone. To run all the nwb files that have been generated,
```bash
uv run ramps/ramps_on_eddie.py --all
```
To selectively run for certain mice and days,
```bash
uv run ramps/ramps_on_eddie.py --mice 1,2 --days 1,2
```
Each job will create a separate folder with the date running the job as the name in `/exports/eddie/scratch/s2155699/ephys/ramps/ramps_results`, the results are stored in subfolder called `tasks`.

To stage out,
```bash
qsub ramps/stageout_ramps.sh
```
This will reshape data structure from
```
ramps/
    ramping_comparison_logs/
    ramps_results/
        run_date/
            many_jsons
            logs/
            numba_cache/
            input/
            tasks/
                task_id/
                    experiment_M*_D*_unit*_results.csv
                    experiment_training/test_M*_D*_unit-*_trial_type-*_test_block-*.png
                    experiment_training/test_M*_D*_unit-*_trial_type-*_test_block-*.npz
```
on EDDIE, to
```
analysis/ramps/
    results/
        mouse_id/
            day/
                experiment_M*_D*_unit*_results.csv
                experiment_training/test_M*_D*_unit-*_trial_type-*_test_block-*.png
                experiment_training/test_M*_D*_unit-*_trial_type-*_test_block-*.npz
```
on datastore. **Importantly, if different job folders contain the same session, only the latest one will be staged out.** This was carried out by `select_results_to_stageout.py`

To cleanup,
```bash
qsub ramps/cleanup.sh
```
This will delete the numba_cache, logs, inputs folders in *ALL* job folders doing ramping classification.

The tasks folder contains the results for each unit, including a csv, figues of tuning curve and fitted lines, and npz files containing firing rate. **ONLY DELETE THE TASKS FOLDERS AFTER DOUBLE CHEKING THEY ARE TRUELLY NOT NEEDED ANYMORE**. To delete the tasks folders, add an argument:
```bash
qsub ramps/cleanup.sh --delete-outputs
```

TODO!!!!!!!!!!!!!! Finally, analyse the results. This will stage in ramps results from datastore in datastore format first so it's better to cleanup before doing these analyses.

To concatenate all the individual units' results,
```bash
uv run ramps/concat_csv.py
```
This will simply create a big `ramps_classification.csv` in `/exports/eddie/scratch/s2155699/ephys/ramps/ramps_results`.

To compare ramps between test blocks and between genotypes,
```bash
mkdir -p /exports/eddie/scratch/s2155699/ephys/ramps/ramping_comparison_logs
qsub ramps/run_ramping_comparison.sh
```

The script also performs staging out automatically. So the final data structure on datastore should look like:
```
analysis/ramps/
    results/
        mouse_id/
            day/
                experiment_M*_D*_unit*_results.csv
                experiment_training/test_M*_D*_unit-*_trial_type-*_test_block-*.png
                experiment_training/test_M*_D*_unit-*_trial_type-*_test_block-*.npz
    combined/ (ramping_comparison.py results)
        ramps_classification.csv
        figs/
            ramping_cells_trial_type-*_test_block-*.png (created by ramping_comparison.py)
        stats/
            right_now_it_is_empty
```

If pynts has been modified, update its version at the veryyyyyy beginning:
```bash
uv add "pynts @ git+https://github.com/yz753/pynts.git@NWR1"
```
The NWR1 branch is dedicated to my project. If running for Junji & Teris' data, use the main branch.

### 4. GLM-HMM (EDDIE)

### 5. Decoding (EDDIE)
The script runs the combination of two decoders (Bayesian and DeepNet) and two testing schemes (training on 80% of all test blocks' data & testing on remaining vs training on 80% of test block 0 data & testing on 20% block 0 and all block 1 & 2 data). Two loss functions for DeepNet can be chosen: cross entropy and wasserstein. The latter takes the vals of position error into account but makes the model too conservative on the prediction (the two ends of the track are not predicted). So cross entropy is set as default.

The decoder requires `nwb_units.jsonl`, then
```bash
bash decoder/run_decoder.sh --mice 1 --days 1
```
Use `--all` to run all sessions, and `--loss-fn` to define loss function for DeepNet.

This will stage in, process, and stage out necessary files.

### 6. Task-anchored & task-independent classification (EDDIE)