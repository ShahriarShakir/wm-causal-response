#!/bin/bash
#PBS -N cKW_I6
#PBS -P <PROJECT>
#PBS -q <GPU_QUEUE>
#PBS -l ngpus=1,ncpus=12,mem=64GB,walltime=05:00:00
#PBS -l <SITE_STORAGE_DIRECTIVE>
#PBS -l wd
#PBS -o <ROOT>/outputs/cKW_I6.o
#PBS -e <ROOT>/outputs/cKW_I6.e
# Locations are read from the environment (see the README); the script stops when one is not set.
ROOT=${ROOT:?set ROOT to the directory of this package}
set -uo pipefail
cd ${ROOT}
export PYTHONPATH=${ROOT}/.tools
  echo "=== Seaquest k=1 w=60 ==="; env -u PYTHONHOME python3 -u scripts/v5_iris_sweep.py ALE/Seaquest-v5 60 8 1 60
  echo "=== UpNDown k=1 w=30 ==="; env -u PYTHONHOME python3 -u scripts/v5_iris_sweep.py ALE/UpNDown-v5 60 8 1 30
echo "finished $(date -Is)"
