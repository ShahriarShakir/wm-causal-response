#!/bin/bash
#PBS -N cKW_I1
#PBS -P <PROJECT>
#PBS -q <GPU_QUEUE>
#PBS -l ngpus=1,ncpus=12,mem=64GB,walltime=05:00:00
#PBS -l <SITE_STORAGE_DIRECTIVE>
#PBS -l wd
#PBS -o <ROOT>/outputs/cKW_I1.o
#PBS -e <ROOT>/outputs/cKW_I1.e
# Locations are read from the environment (see the README); the script stops when one is not set.
ROOT=${ROOT:?set ROOT to the directory of this package}
set -uo pipefail
cd ${ROOT}
export PYTHONPATH=${ROOT}/.tools
  echo "=== BankHeist k=12 w=0 ==="; env -u PYTHONHOME python3 -u scripts/v5_iris_sweep.py ALE/BankHeist-v5 60 8 12 0
  echo "=== BattleZone k=1 w=30 ==="; env -u PYTHONHOME python3 -u scripts/v5_iris_sweep.py ALE/BattleZone-v5 60 8 1 30
  echo "=== Boxing k=1 w=0 ==="; env -u PYTHONHOME python3 -u scripts/v5_iris_sweep.py ALE/Boxing-v5 60 8 1 0
  echo "=== Breakout k=3 w=0 ==="; env -u PYTHONHOME python3 -u scripts/v5_iris_sweep.py ALE/Breakout-v5 60 8 3 0
echo "finished $(date -Is)"
