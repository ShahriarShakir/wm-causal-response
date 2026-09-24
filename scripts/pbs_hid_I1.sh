#!/bin/bash
#PBS -N cHid_I1
#PBS -P <PROJECT>
#PBS -q <GPU_QUEUE>
#PBS -l ngpus=1,ncpus=12,mem=64GB,walltime=06:00:00
#PBS -l <SITE_STORAGE_DIRECTIVE>
#PBS -l wd
#PBS -o <ROOT>/outputs/hid_I1.o
#PBS -e <ROOT>/outputs/hid_I1.e
# Locations are read from the environment (see the README); the script stops when one is not set.
ROOT=${ROOT:?set ROOT to the directory of this package}
set -uo pipefail
cd ${ROOT}
export PYTHONPATH=${ROOT}/.tools
  echo "=== Boxing ==="; env -u PYTHONHOME python3 -u scripts/v9_iris_hidden.py ALE/Boxing-v5 1 0
  echo "=== Breakout ==="; env -u PYTHONHOME python3 -u scripts/v9_iris_hidden.py ALE/Breakout-v5 3 0
  echo "=== Freeway ==="; env -u PYTHONHOME python3 -u scripts/v9_iris_hidden.py ALE/Freeway-v5 1 120
echo "finished $(date -Is)"
