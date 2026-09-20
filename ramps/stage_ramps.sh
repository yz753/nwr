#!/bin/bash
#$ -cwd
#$ -q staging
#$ -N ramps_stage
#$ -l h_rt=02:00:00
set -euo pipefail
SCRIPT_DIR="$1"
RUN_DIR="$2"
PYTHON="$3"
case "$4" in
    in) "$PYTHON" -u "$SCRIPT_DIR/ramps_on_eddie.py" --stage-in-worker "$RUN_DIR" ;;
    out) "$PYTHON" -u "$SCRIPT_DIR/ramps_on_eddie.py" --stage-out-worker "$RUN_DIR" ;;
    *) echo "Expected in or out" >&2; exit 1 ;;
esac
