#!/bin/bash
#$ -cwd
#$ -N ramps
#$ -pe sharedmem 1
#$ -l h_vmem=32G,h_rt=9:59:59
set -euo pipefail
SCRIPT_DIR="$1"
RUN_DIR="$2"
PYTHON="$3"
: "${SGE_TASK_ID:?This script must run as an SGE array task}"
if [[ ! -f "$RUN_DIR/stagein_complete.json" ]]; then
    echo "Staging did not complete successfully. See $RUN_DIR/logs" >&2
    exit 1
fi
export MPLBACKEND=Agg
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
"$PYTHON" -u "$SCRIPT_DIR/ramping_classification.py" \
    --run_dir "$RUN_DIR" --task_id "$SGE_TASK_ID"
