#!/bin/bash
#PBS -N cKW_I2
#PBS -P <PROJECT>
#PBS -q <GPU_QUEUE>
#PBS -l ngpus=1,ncpus=12,mem=64GB,walltime=05:00:00
#PBS -l <SITE_STORAGE_DIRECTIVE>
#PBS -l wd
#PBS -o <ROOT>/outputs/cKW_I2.o
#PBS -e <ROOT>/outputs/cKW_I2.e
# Locations are read from the environment (see the README); the script stops when one is not set.
ROOT=${ROOT:?set ROOT to the directory of this package}
set -uo pipefail
cd ${ROOT}
export PYTHONPATH=${ROOT}/.tools
  echo "=== ChopperCommand k=1 w=120 ==="; env -u PYTHONHOME python3 -u scripts/v5_iris_sweep.py ALE/ChopperCommand-v5 60 8 1 120
  echo "=== CrazyClimber k=1 w=120 ==="; env -u PYTHONHOME python3 -u scripts/v5_iris_sweep.py ALE/CrazyClimber-v5 60 8 1 120
  echo "=== DemonAttack k=3 w=30 ==="; env -u PYTHONHOME python3 -u scripts/v5_iris_sweep.py ALE/DemonAttack-v5 60 8 3 30
  echo "=== Freeway k=1 w=120 ==="; env -u PYTHONHOME python3 -u scripts/v5_iris_sweep.py ALE/Freeway-v5 60 8 1 120
echo "finished $(date -Is)"
