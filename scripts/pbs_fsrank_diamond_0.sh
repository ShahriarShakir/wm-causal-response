#!/bin/bash
#PBS -N cFS_d0
#PBS -P <PROJECT>
#PBS -q <GPU_QUEUE>
#PBS -l ngpus=1,ncpus=12,mem=64GB,walltime=08:00:00
#PBS -l <SITE_STORAGE_DIRECTIVE>
#PBS -l wd
#PBS -o <ROOT>/outputs/fsrank_diamond_0.o
#PBS -e <ROOT>/outputs/fsrank_diamond_0.e
# Locations are read from the environment (see the README); the script stops when one is not set.
ROOT=${ROOT:?set ROOT to the directory of this package}
set -uo pipefail
cd ${ROOT}
export PYTHONPATH=${ROOT}/.tools
  echo "=== Alien ==="; env -u PYTHONHOME python3 -u scripts/v14_fixed_scorer_ranking.py diamond ALE/Alien-v5
echo "=== Pong ==="; env -u PYTHONHOME python3 -u scripts/v14_fixed_scorer_ranking.py diamond ALE/Pong-v5
echo "=== BankHeist ==="; env -u PYTHONHOME python3 -u scripts/v14_fixed_scorer_ranking.py diamond ALE/BankHeist-v5
echo "=== Boxing ==="; env -u PYTHONHOME python3 -u scripts/v14_fixed_scorer_ranking.py diamond ALE/Boxing-v5
echo "=== Breakout ==="; env -u PYTHONHOME python3 -u scripts/v14_fixed_scorer_ranking.py diamond ALE/Breakout-v5
echo "finished $(date -Is)"
