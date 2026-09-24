#!/bin/bash
#PBS -N cKW_D0
#PBS -P <PROJECT>
#PBS -q <GPU_QUEUE>
#PBS -l ngpus=1,ncpus=12,mem=64GB,walltime=03:00:00
#PBS -l <SITE_STORAGE_DIRECTIVE>
#PBS -l wd
#PBS -o <ROOT>/outputs/cKW_D0.o
#PBS -e <ROOT>/outputs/cKW_D0.e
# Locations are read from the environment (see the README); the script stops when one is not set.
ROOT=${ROOT:?set ROOT to the directory of this package}
set -uo pipefail
cd ${ROOT}
export PYTHONPATH=${ROOT}/.tools
  echo "=== Alien k=14 w=60 ==="; env -u PYTHONHOME python3 -u scripts/v4_video_sweep.py ALE/Alien-v5 60 8 14 60
  echo "=== Amidar k=1 w=60 ==="; env -u PYTHONHOME python3 -u scripts/v4_video_sweep.py ALE/Amidar-v5 60 8 1 60
  echo "=== Assault k=2 w=0 ==="; env -u PYTHONHOME python3 -u scripts/v4_video_sweep.py ALE/Assault-v5 60 8 2 0
  echo "=== Asterix k=3 w=0 ==="; env -u PYTHONHOME python3 -u scripts/v4_video_sweep.py ALE/Asterix-v5 60 8 3 0
  echo "=== BankHeist k=12 w=0 ==="; env -u PYTHONHOME python3 -u scripts/v4_video_sweep.py ALE/BankHeist-v5 60 8 12 0
  echo "=== BattleZone k=1 w=30 ==="; env -u PYTHONHOME python3 -u scripts/v4_video_sweep.py ALE/BattleZone-v5 60 8 1 30
  echo "=== Boxing k=1 w=0 ==="; env -u PYTHONHOME python3 -u scripts/v4_video_sweep.py ALE/Boxing-v5 60 8 1 0
echo "finished $(date -Is)"
