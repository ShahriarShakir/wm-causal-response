#!/bin/bash
#PBS -N cFSD_d
#PBS -P <PROJECT>
#PBS -q <GPU_QUEUE>
#PBS -l ngpus=1,ncpus=12,mem=64GB,walltime=08:00:00
#PBS -l <SITE_STORAGE_DIRECTIVE>
#PBS -l wd
#PBS -o <ROOT>/outputs/fsrank_diamond_dense.o
#PBS -e <ROOT>/outputs/fsrank_diamond_dense.e
# Locations are read from the environment (see the README); the script stops when one is not set.
ROOT=${ROOT:?set ROOT to the directory of this package}
set -uo pipefail
cd ${ROOT}
export PYTHONPATH=${ROOT}/.tools
  echo "=== Freeway dense ==="; env -u PYTHONHOME python3 -u scripts/v14_fixed_scorer_ranking.py diamond ALE/Freeway-v5 --densedata
echo "=== Gopher dense ==="; env -u PYTHONHOME python3 -u scripts/v14_fixed_scorer_ranking.py diamond ALE/Gopher-v5 --densedata
echo "=== BankHeist dense ==="; env -u PYTHONHOME python3 -u scripts/v14_fixed_scorer_ranking.py diamond ALE/BankHeist-v5 --densedata
echo "finished $(date -Is)"
