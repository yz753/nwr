#!/bin/bash
#$ -cwd
#$ -q staging
#$ -N ramps_stageout
#$ -l h_rt=03:00:00

set -euo pipefail

SOURCE_ROOT="/exports/eddie/scratch/s2155699/ephys/ramps/ramps_results"
DEST_ROOT="/exports/cmvm/datastore/sbms/groups/INCR-NolanLab/ActiveProjects/Yiming/NWR1/analysis/ramps"
HELPER="/exports/eddie/scratch/s2155699/ephys/nwr/ramps/select_results_to_stageout.py"
PLAN="$SOURCE_ROOT/files_to_stageout_list.tsv"

python3 "$HELPER" "$SOURCE_ROOT" "$PLAN"

mkdir -p "$DEST_ROOT/results"

declare -A prepared_sessions

while IFS=$'\t' read -r task_dir mouse day session_type run_dir; do
    destination="$DEST_ROOT/results/$mouse/$day"
    session_key="$mouse/$day"

    # Remove the previously staged outputs for this mouse/day once.
    if [[ -z "${prepared_sessions[$session_key]+x}" ]]; then
        mkdir -p "$destination"

        find "$destination" -maxdepth 1 -type f \
            \( -name "*_results.csv" \
            -o -name "*.png" \
            -o -name "*.npz" \) \
            -delete

        prepared_sessions[$session_key]=1

        echo "Selected run: $run_dir"
        echo "Staging: $mouse $day $session_type"
        echo "Destination: $destination"
    fi

    rsync -rt \
        --no-perms \
        --no-owner \
        --no-group \
        --include="*.csv" \
        --include="*.png" \
        --include="*.npz" \
        --exclude="*" \
        "$task_dir/" \
        "$destination/"
done < "$PLAN"

echo "Stage-out completed."