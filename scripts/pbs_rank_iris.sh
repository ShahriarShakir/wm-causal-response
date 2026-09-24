#!/bin/bash
#PBS -N cRank_i
#PBS -P <PROJECT>
#PBS -q <GPU_QUEUE>
#PBS -l ngpus=1,ncpus=12,mem=64GB,walltime=05:00:00
#PBS -l <SITE_STORAGE_DIRECTIVE>
#PBS -l wd
#PBS -o <ROOT>/outputs/rank_iris.o
#PBS -e <ROOT>/outputs/rank_iris.e
# Locations are read from the environment (see the README); the script stops when one is not set.
ROOT=${ROOT:?set ROOT to the directory of this package}
set -uo pipefail
cd ${ROOT}
export PYTHONPATH=${ROOT}/.tools
  echo "=== Alien ==="; env -u PYTHONHOME python3 -u scripts/v11_action_ranking.py iris ALE/Alien-v5
  echo "=== Pong ==="; env -u PYTHONHOME python3 -u scripts/v11_action_ranking.py iris ALE/Pong-v5
  echo "=== BankHeist ==="; env -u PYTHONHOME python3 -u scripts/v11_action_ranking.py iris ALE/BankHeist-v5
  echo "=== Boxing ==="; env -u PYTHONHOME python3 -u scripts/v11_action_ranking.py iris ALE/Boxing-v5
  echo "=== Breakout ==="; env -u PYTHONHOME python3 -u scripts/v11_action_ranking.py iris ALE/Breakout-v5
  echo "=== Freeway ==="; env -u PYTHONHOME python3 -u scripts/v11_action_ranking.py iris ALE/Freeway-v5
  echo "=== Gopher ==="; env -u PYTHONHOME python3 -u scripts/v11_action_ranking.py iris ALE/Gopher-v5
  echo "=== Seaquest ==="; env -u PYTHONHOME python3 -u scripts/v11_action_ranking.py iris ALE/Seaquest-v5
  echo "=== MsPacman ==="; env -u PYTHONHOME python3 -u scripts/v11_action_ranking.py iris ALE/MsPacman-v5
  echo "=== Frostbite ==="; env -u PYTHONHOME python3 -u scripts/v11_action_ranking.py iris ALE/Frostbite-v5
echo "finished $(date -Is)"
