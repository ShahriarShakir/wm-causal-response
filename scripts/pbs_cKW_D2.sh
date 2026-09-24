#!/bin/bash
#PBS -N cKW_D2
#PBS -P <PROJECT>
#PBS -q <GPU_QUEUE>
#PBS -l ngpus=1,ncpus=12,mem=64GB,walltime=03:00:00
#PBS -l <SITE_STORAGE_DIRECTIVE>
#PBS -l wd
#PBS -o <ROOT>/outputs/cKW_D2.o
#PBS -e <ROOT>/outputs/cKW_D2.e
# Locations are read from the environment (see the README); the script stops when one is not set.
ROOT=${ROOT:?set ROOT to the directory of this package}
set -uo pipefail
cd ${ROOT}
export PYTHONPATH=${ROOT}/.tools
  echo "=== Hero k=1 w=30 ==="; env -u PYTHONHOME python3 -u scripts/v4_video_sweep.py ALE/Hero-v5 60 8 1 30
  echo "=== Jamesbond k=2 w=60 ==="; env -u PYTHONHOME python3 -u scripts/v4_video_sweep.py ALE/Jamesbond-v5 60 8 2 60
  echo "=== Kangaroo k=1 w=30 ==="; env -u PYTHONHOME python3 -u scripts/v4_video_sweep.py ALE/Kangaroo-v5 60 8 1 30
  echo "=== Krull k=1 w=120 ==="; env -u PYTHONHOME python3 -u scripts/v4_video_sweep.py ALE/Krull-v5 60 8 1 120
  echo "=== KungFuMaster k=1 w=120 ==="; env -u PYTHONHOME python3 -u scripts/v4_video_sweep.py ALE/KungFuMaster-v5 60 8 1 120
  echo "=== MsPacman k=1 w=60 ==="; env -u PYTHONHOME python3 -u scripts/v4_video_sweep.py ALE/MsPacman-v5 60 8 1 60
  echo "=== Pong k=2 w=0 ==="; env -u PYTHONHOME python3 -u scripts/v4_video_sweep.py ALE/Pong-v5 60 8 2 0
echo "finished $(date -Is)"
