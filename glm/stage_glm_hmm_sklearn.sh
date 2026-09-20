#!/bin/bash
#$ -cwd
#$ -q staging
#$ -l h_rt=02:00:00
set -euo pipefail
"$3" -u "$1/glm_hmm_sklearn_on_eddie.py" --worker stage --run-dir "$2"

