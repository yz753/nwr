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
export NUMBA_CACHE_DIR="$RUN_DIR/numba_cache/${JOB_ID}_${SGE_TASK_ID}"
mkdir -p "$NUMBA_CACHE_DIR"
export NUMBA_DEBUG_CACHE=1
export PYTHONFAULTHANDLER=1

echo "Host: $(hostname)"
echo "Virtual-memory limit (KB): $(ulimit -v)"

# Record this shell's child processes every 10 seconds.
(
    while true; do
        date
        ps --ppid "$$" -o pid,rss,vsz,etime,args
        sleep 10
    done
) > "$RUN_DIR/logs/memory_${JOB_ID}_${SGE_TASK_ID}.log" &
MEMORY_MONITOR_PID=$!
trap 'kill "$MEMORY_MONITOR_PID" 2>/dev/null || true' EXIT
"$PYTHON" -u "$SCRIPT_DIR/ramping_classification.py" \
    --run_dir "$RUN_DIR" --task_id "$SGE_TASK_ID"
