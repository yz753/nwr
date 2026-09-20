#!/bin/bash
#$ -cwd
#$ -pe sharedmem 1
#$ -q *@@uoe_512G_56s

set -euo pipefail
SCRIPT_DIR="$1"
RUN_DIR="$2"
PYTHON="$3"
PHASE="$4"
OFFSET="$5"
: "${SGE_TASK_ID:?Must run as an array task}"
export MPLBACKEND=Agg
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export JAX_PLATFORMS=cpu
export XLA_FLAGS="${XLA_FLAGS:-} --xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
export PYTHONFAULTHANDLER=1
export NUMBA_CACHE_DIR="$RUN_DIR/numba_cache/${JOB_ID}_${SGE_TASK_ID}"
export MPLCONFIGDIR="$RUN_DIR/mpl_cache/${JOB_ID}_${SGE_TASK_ID}"
mkdir -p "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR"
"$PYTHON" -u "$SCRIPT_DIR/glm_hmm_on_eddie.py" --worker "$PHASE" \
    --run-dir "$RUN_DIR" --index "$((OFFSET + SGE_TASK_ID))"

