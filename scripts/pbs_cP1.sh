#!/bin/bash
#PBS -N cP1
#PBS -P <PROJECT>
#PBS -q <GPU_QUEUE>
#PBS -l ngpus=1,ncpus=12,mem=64GB,walltime=03:00:00
#PBS -l <SITE_STORAGE_DIRECTIVE>
#PBS -l wd
#PBS -o <ROOT>/outputs/cP1.o
#PBS -e <ROOT>/outputs/cP1.e
# Locations are read from the environment (see the README); the script stops when one is not set.
ROOT=${ROOT:?set ROOT to the directory of this package}
set -uo pipefail
cd ${ROOT}
export PYTHONPATH=${ROOT}/.tools
for G in Breakout ChopperCommand CrazyClimber DemonAttack Freeway Frostbite Gopher; do
  echo "=== $G ==="
  env -u PYTHONHOME python3 -u scripts/v4_video_sweep.py ALE/$G-v5 60 8
done
echo "finished $(date -Is)"
