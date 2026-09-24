#!/bin/bash
#PBS -N cSus_i2
#PBS -P <PROJECT>
#PBS -q <GPU_QUEUE>
#PBS -l ngpus=1,ncpus=12,mem=64GB,walltime=03:00:00
#PBS -l <SITE_STORAGE_DIRECTIVE>
#PBS -l wd
#PBS -o <ROOT>/outputs/sus_i2.o
#PBS -e <ROOT>/outputs/sus_i2.e
# Locations are read from the environment (see the README); the script stops when one is not set.
ROOT=${ROOT:?set ROOT to the directory of this package}
set -uo pipefail
cd ${ROOT}
export PYTHONPATH=${ROOT}/.tools
  echo "=== Breakout sustained ==="; HORIZON=48 env -u PYTHONHOME python3 -u scripts/v5_iris_sweep.py ALE/Breakout-v5 60 40 3 0
echo "=== Frostbite sustained ==="; HORIZON=48 env -u PYTHONHOME python3 -u scripts/v5_iris_sweep.py ALE/Frostbite-v5 60 40 1 30
echo "finished $(date -Is)"
