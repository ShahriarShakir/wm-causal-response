#!/bin/bash
#PBS -N cKW_D1
#PBS -P <PROJECT>
#PBS -q <GPU_QUEUE>
#PBS -l ngpus=1,ncpus=12,mem=64GB,walltime=03:00:00
#PBS -l <SITE_STORAGE_DIRECTIVE>
#PBS -l wd
#PBS -o <ROOT>/outputs/cKW_D1.o
#PBS -e <ROOT>/outputs/cKW_D1.e
# Locations are read from the environment (see the README); the script stops when one is not set.
ROOT=${ROOT:?set ROOT to the directory of this package}
set -uo pipefail
cd ${ROOT}
export PYTHONPATH=${ROOT}/.tools
  echo "=== Breakout k=3 w=0 ==="; env -u PYTHONHOME python3 -u scripts/v4_video_sweep.py ALE/Breakout-v5 60 8 3 0
  echo "=== ChopperCommand k=1 w=120 ==="; env -u PYTHONHOME python3 -u scripts/v4_video_sweep.py ALE/ChopperCommand-v5 60 8 1 120
  echo "=== CrazyClimber k=1 w=120 ==="; env -u PYTHONHOME python3 -u scripts/v4_video_sweep.py ALE/CrazyClimber-v5 60 8 1 120
  echo "=== DemonAttack k=3 w=30 ==="; env -u PYTHONHOME python3 -u scripts/v4_video_sweep.py ALE/DemonAttack-v5 60 8 3 30
  echo "=== Freeway k=1 w=120 ==="; env -u PYTHONHOME python3 -u scripts/v4_video_sweep.py ALE/Freeway-v5 60 8 1 120
  echo "=== Frostbite k=1 w=30 ==="; env -u PYTHONHOME python3 -u scripts/v4_video_sweep.py ALE/Frostbite-v5 60 8 1 30
  echo "=== Gopher k=1 w=120 ==="; env -u PYTHONHOME python3 -u scripts/v4_video_sweep.py ALE/Gopher-v5 60 8 1 120
echo "finished $(date -Is)"
