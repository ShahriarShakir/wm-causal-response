#!/bin/bash
#PBS -N cIPS
#PBS -P <PROJECT>
#PBS -q <GPU_QUEUE>
#PBS -l ngpus=1,ncpus=12,mem=64GB,walltime=05:00:00
#PBS -l <SITE_STORAGE_DIRECTIVE>
#PBS -l wd
#PBS -o <ROOT>/outputs/ipsnr.o
#PBS -e <ROOT>/outputs/ipsnr.e
# Locations are read from the environment (see the README); the script stops when one is not set.
ROOT=${ROOT:?set ROOT to the directory of this package}
set -uo pipefail
cd ${ROOT}
export PYTHONPATH=${ROOT}/.tools
  echo "=== Alien v4_video_sweep.py ==="; env -u PYTHONHOME python3 -u scripts/v4_video_sweep.py ALE/Alien-v5 60 8 14 60
echo "=== Pong v4_video_sweep.py ==="; env -u PYTHONHOME python3 -u scripts/v4_video_sweep.py ALE/Pong-v5 60 8 2 0
echo "=== Alien v5_iris_sweep.py ==="; env -u PYTHONHOME python3 -u scripts/v5_iris_sweep.py ALE/Alien-v5 60 8 14 60
echo "=== Pong v5_iris_sweep.py ==="; env -u PYTHONHOME python3 -u scripts/v5_iris_sweep.py ALE/Pong-v5 60 8 2 0
echo "finished $(date -Is)"
