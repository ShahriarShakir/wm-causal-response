#!/bin/bash
#PBS -N cRankL_d
#PBS -P <PROJECT>
#PBS -q <GPU_QUEUE>
#PBS -l ngpus=1,ncpus=12,mem=64GB,walltime=08:00:00
#PBS -l <SITE_STORAGE_DIRECTIVE>
#PBS -l wd
#PBS -o <ROOT>/outputs/rankL_diamond.o
#PBS -e <ROOT>/outputs/rankL_diamond.e
# Locations are read from the environment (see the README); the script stops when one is not set.
ROOT=${ROOT:?set ROOT to the directory of this package}
set -uo pipefail
cd ${ROOT}
export PYTHONPATH=${ROOT}/.tools
  echo "=== Alien ==="; env -u PYTHONHOME python3 -u scripts/v11_action_ranking.py diamond ALE/Alien-v5 --long
  echo "=== Pong ==="; env -u PYTHONHOME python3 -u scripts/v11_action_ranking.py diamond ALE/Pong-v5 --long
  echo "=== BankHeist ==="; env -u PYTHONHOME python3 -u scripts/v11_action_ranking.py diamond ALE/BankHeist-v5 --long
  echo "=== Boxing ==="; env -u PYTHONHOME python3 -u scripts/v11_action_ranking.py diamond ALE/Boxing-v5 --long
  echo "=== Breakout ==="; env -u PYTHONHOME python3 -u scripts/v11_action_ranking.py diamond ALE/Breakout-v5 --long
  echo "=== Freeway ==="; env -u PYTHONHOME python3 -u scripts/v11_action_ranking.py diamond ALE/Freeway-v5 --long
  echo "=== Gopher ==="; env -u PYTHONHOME python3 -u scripts/v11_action_ranking.py diamond ALE/Gopher-v5 --long
  echo "=== Seaquest ==="; env -u PYTHONHOME python3 -u scripts/v11_action_ranking.py diamond ALE/Seaquest-v5 --long
  echo "=== MsPacman ==="; env -u PYTHONHOME python3 -u scripts/v11_action_ranking.py diamond ALE/MsPacman-v5 --long
  echo "=== Frostbite ==="; env -u PYTHONHOME python3 -u scripts/v11_action_ranking.py diamond ALE/Frostbite-v5 --long
echo "finished $(date -Is)"
