#!/bin/bash
#PBS -N cIDM_D
#PBS -P <PROJECT>
#PBS -q <GPU_QUEUE>
#PBS -l ngpus=1,ncpus=12,mem=64GB,walltime=04:00:00
#PBS -l <SITE_STORAGE_DIRECTIVE>
#PBS -l wd
#PBS -o <ROOT>/outputs/cIDM_D.o
#PBS -e <ROOT>/outputs/cIDM_D.e
# Locations are read from the environment (see the README); the script stops when one is not set.
ROOT=${ROOT:?set ROOT to the directory of this package}
set -uo pipefail
cd ${ROOT}
export PYTHONPATH=${ROOT}/.tools
  echo "=== Alien ==="; env -u PYTHONHOME python3 -u scripts/v8_idm_consistency.py DIAMOND ALE/Alien-v5 14 60
  echo "=== Pong ==="; env -u PYTHONHOME python3 -u scripts/v8_idm_consistency.py DIAMOND ALE/Pong-v5 2 0
  echo "=== BankHeist ==="; env -u PYTHONHOME python3 -u scripts/v8_idm_consistency.py DIAMOND ALE/BankHeist-v5 12 0
  echo "=== Boxing ==="; env -u PYTHONHOME python3 -u scripts/v8_idm_consistency.py DIAMOND ALE/Boxing-v5 1 0
  echo "=== Breakout ==="; env -u PYTHONHOME python3 -u scripts/v8_idm_consistency.py DIAMOND ALE/Breakout-v5 3 0
  echo "=== Freeway ==="; env -u PYTHONHOME python3 -u scripts/v8_idm_consistency.py DIAMOND ALE/Freeway-v5 1 120
  echo "=== CrazyClimber ==="; env -u PYTHONHOME python3 -u scripts/v8_idm_consistency.py DIAMOND ALE/CrazyClimber-v5 1 120
  echo "=== DemonAttack ==="; env -u PYTHONHOME python3 -u scripts/v8_idm_consistency.py DIAMOND ALE/DemonAttack-v5 3 30
echo "finished $(date -Is)"
