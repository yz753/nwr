#!/bin/bash
#$ -cwd
#$ -N ramping_comparison
#$ -pe sharedmem 1
#$ -l h_rss=4G
#$ -l h_rt=00:30:00
#$ -o /exports/eddie/scratch/s2155699/ephys/ramps/ramping_comparison_logs
#$ -e /exports/eddie/scratch/s2155699/ephys/ramps/ramping_comparison_logs

set -euo pipefail

PYTHON="/exports/eddie/scratch/s2155699/ephys/nwr/.venv/bin/python"
SCRIPT="ramping_comparison.py"

"$PYTHON" -u "$SCRIPT"