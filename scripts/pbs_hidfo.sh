#!/bin/bash
#PBS -N cHidFO
#PBS -P <PROJECT>
#PBS -q <GPU_QUEUE>
#PBS -l ngpus=1,ncpus=12,mem=64GB,walltime=04:00:00
#PBS -l <SITE_STORAGE_DIRECTIVE>
#PBS -l wd
#PBS -o <ROOT>/outputs/hidfo.o
#PBS -e <ROOT>/outputs/hidfo.e
# Locations are read from the environment (see the README); the script stops when one is not set.
ROOT=${ROOT:?set ROOT to the directory of this package}
set -uo pipefail
cd ${ROOT}
export PYTHONPATH=${ROOT}/.tools
  echo "=== Alien ==="; env -u PYTHONHOME python3 -u scripts/v13_iris_hidden_frames_only.py ALE/Alien-v5 14 60
echo "=== Pong ==="; env -u PYTHONHOME python3 -u scripts/v13_iris_hidden_frames_only.py ALE/Pong-v5 2 0
echo "=== BankHeist ==="; env -u PYTHONHOME python3 -u scripts/v13_iris_hidden_frames_only.py ALE/BankHeist-v5 12 0
echo "finished $(date -Is)"
