#!/bin/bash
#PBS -N cKW_I3
#PBS -P <PROJECT>
#PBS -q <GPU_QUEUE>
#PBS -l ngpus=1,ncpus=12,mem=64GB,walltime=05:00:00
#PBS -l <SITE_STORAGE_DIRECTIVE>
#PBS -l wd
#PBS -o <ROOT>/outputs/cKW_I3.o
#PBS -e <ROOT>/outputs/cKW_I3.e
# Locations are read from the environment (see the README); the script stops when one is not set.
ROOT=${ROOT:?set ROOT to the directory of this package}
set -uo pipefail
cd ${ROOT}
export PYTHONPATH=${ROOT}/.tools
  echo "=== Frostbite k=1 w=30 ==="; env -u PYTHONHOME python3 -u scripts/v5_iris_sweep.py ALE/Frostbite-v5 60 8 1 30
  echo "=== Gopher k=1 w=120 ==="; env -u PYTHONHOME python3 -u scripts/v5_iris_sweep.py ALE/Gopher-v5 60 8 1 120
  echo "=== Hero k=1 w=30 ==="; env -u PYTHONHOME python3 -u scripts/v5_iris_sweep.py ALE/Hero-v5 60 8 1 30
  echo "=== Jamesbond k=2 w=60 ==="; env -u PYTHONHOME python3 -u scripts/v5_iris_sweep.py ALE/Jamesbond-v5 60 8 2 60
echo "finished $(date -Is)"
