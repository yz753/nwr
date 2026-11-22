#!/bin/bash
#$ -cwd
#$ -N cleanup_glmhmm
#$ -l h_rt=01:00:00
#$ -l h_rss=2G

set -euo pipefail
shopt -s nullglob

ROOT="/exports/eddie/scratch/s2155699/ephys/glm_hmm"

delete_outputs=false

case "${1:-}" in
    "")
        ;;
    --delete-outputs)
        delete_outputs=true
        ;;
    *)
        echo "Usage: qsub $0 [--delete-outputs]" >&2
        exit 1
        ;;
esac

folders=(numba_cache mpl_cache logs input)

if [[ "$delete_outputs" == true ]]; then
    folders+=(units)
fi

for run_dir in "$ROOT"/run_*; do
    [[ -d "$run_dir" ]] || continue

    for name in "${folders[@]}"; do
        target="$run_dir/$name"

        if [[ -d "$target" ]]; then
            echo "Deleting: $target"
            rm -rf -- "$target"
        fi
    done
done

echo "Cleanup finished."