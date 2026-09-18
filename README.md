# Spikesorting, preprocessing & analysis for NWR1 dataset

## Run spikesorting (EDDIE)
First, all NP recording filepaths should be stored in `nolanlab-ephys/scripts/yimingyiming_filepaths.csv`. To do this, in `nolanlab-ephys/scripts/yiming`, run
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
The DLC results are suboptimal. To reduce false positive, on local machine, `cd lick_detection` then `uv run main --mouse 1 --day 1.py`. On the pop-out window, draw points on the figure to crop the tongue region. This will create a lick_mask.csv and an analysing figure in the `dlc_output_tongue` folder.

## Bombcell curation, syncing data (EDDIE)
For the scripts to run without error, `labels/mice_information/mouse_data.csv` and `metadata/generic_metadata.yml` should be created first in `/Volumes/INCR-NolanLab/ActiveProjects/Yiming/NWR1/ephys/derivatives`. 

First, stage in all necessary files.
```bash
qsub scripts/preprocessing/HPC/stagein.sh M1 VR D1 D2 D3 ...
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