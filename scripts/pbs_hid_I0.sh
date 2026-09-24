#!/bin/bash
#PBS -N cHid_I0
#PBS -P <PROJECT>
#PBS -q <GPU_QUEUE>
#PBS -l ngpus=1,ncpus=12,mem=64GB,walltime=06:00:00
#PBS -l <SITE_STORAGE_DIRECTIVE>
#PBS -l wd
#PBS -o <ROOT>/outputs/hid_I0.o
#PBS -e <ROOT>/outputs/hid_I0.e
# Locations are read from the environment (see the README); the script stops when one is not set.
ROOT=${ROOT:?set ROOT to the directory of this package}
set -uo pipefail
cd ${ROOT}
export PYTHONPATH=${ROOT}/.tools
  echo "=== Alien ==="; env -u PYTHONHOME python3 -u scripts/v9_iris_hidden.py ALE/Alien-v5 14 60
  echo "=== Pong ==="; env -u PYTHONHOME python3 -u scripts/v9_iris_hidden.py ALE/Pong-v5 2 0
  echo "=== BankHeist ==="; env -u PYTHONHOME python3 -u scripts/v9_iris_hidden.py ALE/BankHeist-v5 12 0
echo "finished $(date -Is)"
