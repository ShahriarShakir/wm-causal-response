#!/bin/bash
#PBS -N cI5
#PBS -P <PROJECT>
#PBS -q <GPU_QUEUE>
#PBS -l ngpus=1,ncpus=12,mem=64GB,walltime=05:00:00
#PBS -l <SITE_STORAGE_DIRECTIVE>
#PBS -l wd
#PBS -o <ROOT>/outputs/cI5.o
#PBS -e <ROOT>/outputs/cI5.e
# Locations are read from the environment (see the README); the script stops when one is not set.
ROOT=${ROOT:?set ROOT to the directory of this package}
set -uo pipefail
cd ${ROOT}
export PYTHONPATH=${ROOT}/.tools
for G in Pong PrivateEye Qbert RoadRunner; do
  echo "=== $G ==="
  env -u PYTHONHOME python3 -u scripts/v5_iris_sweep.py ALE/$G-v5 60 8
done
echo "finished $(date -Is)"
