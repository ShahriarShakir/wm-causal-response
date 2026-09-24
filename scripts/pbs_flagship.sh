#!/bin/bash
#PBS -N cFlag
#PBS -P <PROJECT>
#PBS -q <GPU_QUEUE>
#PBS -l ngpus=1,ncpus=12,mem=64GB,walltime=02:00:00
#PBS -l <SITE_STORAGE_DIRECTIVE>
#PBS -l wd
#PBS -o <ROOT>/outputs/flagship.o
#PBS -e <ROOT>/outputs/flagship.e
# Locations are read from the environment (see the README); the script stops when one is not set.
ROOT=${ROOT:?set ROOT to the directory of this package}
set -uo pipefail
cd ${ROOT}
export PYTHONPATH=${ROOT}/.tools
echo "started $(date -Is)"
for G in BankHeist Pong; do
  for D in 8 16; do
    echo "=== $G duration $D ==="
    env -u PYTHONHOME python3 -u \
        scripts/v4_video_sweep.py ALE/$G-v5 60 $D
  done
done
echo "finished $(date -Is)"
