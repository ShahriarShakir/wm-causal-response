#!/bin/bash
#PBS -N cKW_I5
#PBS -P <PROJECT>
#PBS -q <GPU_QUEUE>
#PBS -l ngpus=1,ncpus=12,mem=64GB,walltime=05:00:00
#PBS -l <SITE_STORAGE_DIRECTIVE>
#PBS -l wd
#PBS -o <ROOT>/outputs/cKW_I5.o
#PBS -e <ROOT>/outputs/cKW_I5.e
# Locations are read from the environment (see the README); the script stops when one is not set.
ROOT=${ROOT:?set ROOT to the directory of this package}
set -uo pipefail
cd ${ROOT}
export PYTHONPATH=${ROOT}/.tools
  echo "=== Pong k=2 w=0 ==="; env -u PYTHONHOME python3 -u scripts/v5_iris_sweep.py ALE/Pong-v5 60 8 2 0
  echo "=== PrivateEye k=14 w=0 ==="; env -u PYTHONHOME python3 -u scripts/v5_iris_sweep.py ALE/PrivateEye-v5 60 8 14 0
  echo "=== Qbert k=1 w=30 ==="; env -u PYTHONHOME python3 -u scripts/v5_iris_sweep.py ALE/Qbert-v5 60 8 1 30
  echo "=== RoadRunner k=14 w=30 ==="; env -u PYTHONHOME python3 -u scripts/v5_iris_sweep.py ALE/RoadRunner-v5 60 8 14 30
echo "finished $(date -Is)"
