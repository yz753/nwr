#!/bin/bash
#$ -cwd
#$ -q staging
#$ -N ramps_inventory
#$ -l h_rt=02:00:00
set -euo pipefail
SCRIPT_DIR='/exports/eddie/scratch/s2155699/ephys/nwr/ramps'
SOURCE_ROOT="/exports/cmvm/datastore/sbms/groups/INCR-NolanLab/ActiveProjects/Yiming/NWR1/processed"
OUTPUT_FILE="/exports/eddie/scratch/s2155699/ephys/nwr/nwb_units.jsonl"
PYTHON="/exports/eddie/scratch/s2155699/ephys/nwr/.venv/bin/python3"
"$PYTHON" -u "$SCRIPT_DIR/prepare_nwb_units.py" --data_folder "$SOURCE_ROOT" --output "$OUTPUT_FILE"
