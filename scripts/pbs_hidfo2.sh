#!/bin/bash
#PBS -N cHidFO2
#PBS -P <PROJECT>
#PBS -q <GPU_QUEUE>
#PBS -l ngpus=1,ncpus=12,mem=64GB,walltime=04:00:00
#PBS -l <SITE_STORAGE_DIRECTIVE>
#PBS -l wd
#PBS -o <ROOT>/outputs/hidfo2.o
#PBS -e <ROOT>/outputs/hidfo2.e
# Locations are read from the environment (see the README); the script stops when one is not set.
ROOT=${ROOT:?set ROOT to the directory of this package}
set -uo pipefail
cd ${ROOT}
export PYTHONPATH=${ROOT}/.tools
  echo "=== Boxing ==="; env -u PYTHONHOME python3 -u scripts/v13_iris_hidden_frames_only.py ALE/Boxing-v5 1 0
echo "=== Breakout ==="; env -u PYTHONHOME python3 -u scripts/v13_iris_hidden_frames_only.py ALE/Breakout-v5 3 0
echo "=== Freeway ==="; env -u PYTHONHOME python3 -u scripts/v13_iris_hidden_frames_only.py ALE/Freeway-v5 1 120
echo "finished $(date -Is)"
