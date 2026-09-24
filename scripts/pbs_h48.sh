#!/bin/bash
#PBS -N cH48
#PBS -P <PROJECT>
#PBS -q <GPU_QUEUE>
#PBS -l ngpus=1,ncpus=12,mem=64GB,walltime=03:00:00
#PBS -l <SITE_STORAGE_DIRECTIVE>
#PBS -l wd
#PBS -o <ROOT>/outputs/h48.o
#PBS -e <ROOT>/outputs/h48.e
# Locations are read from the environment (see the README); the script stops when one is not set.
ROOT=${ROOT:?set ROOT to the directory of this package}
set -uo pipefail
cd ${ROOT}
export PYTHONPATH=${ROOT}/.tools
  echo "=== Alien h48 ==="; HORIZON=48 env -u PYTHONHOME python3 -u scripts/v4_video_sweep.py ALE/Alien-v5 60 8 14 60
  echo "=== Pong h48 ==="; HORIZON=48 env -u PYTHONHOME python3 -u scripts/v4_video_sweep.py ALE/Pong-v5 60 8 2 0
echo "finished $(date -Is)"
