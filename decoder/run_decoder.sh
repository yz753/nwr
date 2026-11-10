#!/bin/bash
#$ -cwd
#$ -pe sharedmem 1
#$ -l h_vmem=8G,h_rt=02:00:00
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# Edit these defaults, then submit with: bash decoder/run_decoder.sh --all
export DECODER_OUTPUT_DIR="${DECODER_OUTPUT_DIR:-/exports/eddie/scratch/s2155699/ephys/decoder}"
export DECODER_LOSS_FN="${DECODER_LOSS_FN:-wasserstein}"
PYTHON="${DECODER_PYTHON:-python}"
export MPLBACKEND=Agg
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
if [[ "${1:-}" == stage || "${1:-}" == fit ]]; then
    PHASE="$1"
    TASKS="$2"
    OUTPUT="$3"
    LOSS="$4"
    PYTHON="$5"
    # SGE may execute a spool copy; recover the original script directory.
    SCRIPT_DIR="${DECODER_SCRIPT_DIR:?Missing submitted script directory}"
    export MPLCONFIGDIR="${TMPDIR:-/tmp}/decoder_mpl_${JOB_ID}_${SGE_TASK_ID:-0}"
    export NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/decoder_numba_${JOB_ID}_${SGE_TASK_ID:-0}"
    exec "$PYTHON" -u "$SCRIPT_DIR/run_on_eddie.py" --phase "$PHASE" \
        --tasks "$TASKS" --output-dir "$OUTPUT" --loss-fn "$LOSS"
fi
exec "$PYTHON" -u "$SCRIPT_DIR/run_on_eddie.py" "$@"
