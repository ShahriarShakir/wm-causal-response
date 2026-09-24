#!/bin/bash
#PBS -N cRepr_I0
#PBS -P <PROJECT>
#PBS -q <GPU_QUEUE>
#PBS -l ngpus=1,ncpus=12,mem=64GB,walltime=05:00:00
#PBS -l <SITE_STORAGE_DIRECTIVE>
#PBS -l wd
#PBS -o <ROOT>/outputs/repr_I0.o
#PBS -e <ROOT>/outputs/repr_I0.e
# Locations are read from the environment (see the README); the script stops when one is not set.
ROOT=${ROOT:?set ROOT to the directory of this package}
set -uo pipefail
cd ${ROOT}
export PYTHONPATH=${ROOT}/.tools
  echo "=== Alien ==="; env -u PYTHONHOME python3 -u scripts/v6_iris_repr.py ALE/Alien-v5 60 14 60
  echo "=== Pong ==="; env -u PYTHONHOME python3 -u scripts/v6_iris_repr.py ALE/Pong-v5 60 2 0
  echo "=== BankHeist ==="; env -u PYTHONHOME python3 -u scripts/v6_iris_repr.py ALE/BankHeist-v5 60 12 0
echo "finished $(date -Is)"
