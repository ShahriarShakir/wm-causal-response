#!/bin/bash
#PBS -N cDP_D
#PBS -P <PROJECT>
#PBS -q <GPU_QUEUE>
#PBS -l ngpus=1,ncpus=12,mem=64GB,walltime=03:00:00
#PBS -l <SITE_STORAGE_DIRECTIVE>
#PBS -l wd
#PBS -o <ROOT>/outputs/cDP_D.o
#PBS -e <ROOT>/outputs/cDP_D.e
# Locations are read from the environment (see the README); the script stops when one is not set.
ROOT=${ROOT:?set ROOT to the directory of this package}
set -uo pipefail
cd ${ROOT}
export PYTHONPATH=${ROOT}/.tools
  echo "=== Alien ==="; env -u PYTHONHOME python3 -u scripts/v4_video_sweep.py ALE/Alien-v5 60 8 14 60
  echo "=== Pong ==="; env -u PYTHONHOME python3 -u scripts/v4_video_sweep.py ALE/Pong-v5 60 8 2 0
  echo "=== BankHeist ==="; env -u PYTHONHOME python3 -u scripts/v4_video_sweep.py ALE/BankHeist-v5 60 8 12 0
  echo "=== Boxing ==="; env -u PYTHONHOME python3 -u scripts/v4_video_sweep.py ALE/Boxing-v5 60 8 1 0
  echo "=== Breakout ==="; env -u PYTHONHOME python3 -u scripts/v4_video_sweep.py ALE/Breakout-v5 60 8 3 0
  echo "=== Freeway ==="; env -u PYTHONHOME python3 -u scripts/v4_video_sweep.py ALE/Freeway-v5 60 8 1 120
  echo "=== CrazyClimber ==="; env -u PYTHONHOME python3 -u scripts/v4_video_sweep.py ALE/CrazyClimber-v5 60 8 1 120
  echo "=== DemonAttack ==="; env -u PYTHONHOME python3 -u scripts/v4_video_sweep.py ALE/DemonAttack-v5 60 8 3 30
echo "finished $(date -Is)"
