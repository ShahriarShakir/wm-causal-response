#!/bin/bash
#PBS -N cFS_d1
#PBS -P <PROJECT>
#PBS -q <GPU_QUEUE>
#PBS -l ngpus=1,ncpus=12,mem=64GB,walltime=08:00:00
#PBS -l <SITE_STORAGE_DIRECTIVE>
#PBS -l wd
#PBS -o <ROOT>/outputs/fsrank_diamond_1.o
#PBS -e <ROOT>/outputs/fsrank_diamond_1.e
# Locations are read from the environment (see the README); the script stops when one is not set.
ROOT=${ROOT:?set ROOT to the directory of this package}
set -uo pipefail
cd ${ROOT}
export PYTHONPATH=${ROOT}/.tools
  echo "=== Freeway ==="; env -u PYTHONHOME python3 -u scripts/v14_fixed_scorer_ranking.py diamond ALE/Freeway-v5
echo "=== Gopher ==="; env -u PYTHONHOME python3 -u scripts/v14_fixed_scorer_ranking.py diamond ALE/Gopher-v5
echo "=== Seaquest ==="; env -u PYTHONHOME python3 -u scripts/v14_fixed_scorer_ranking.py diamond ALE/Seaquest-v5
echo "=== MsPacman ==="; env -u PYTHONHOME python3 -u scripts/v14_fixed_scorer_ranking.py diamond ALE/MsPacman-v5
echo "=== Frostbite ==="; env -u PYTHONHOME python3 -u scripts/v14_fixed_scorer_ranking.py diamond ALE/Frostbite-v5
echo "finished $(date -Is)"
