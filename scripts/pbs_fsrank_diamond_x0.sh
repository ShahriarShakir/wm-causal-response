#!/bin/bash
#PBS -N cFSX_d0
#PBS -P <PROJECT>
#PBS -q <GPU_QUEUE>
#PBS -l ngpus=1,ncpus=12,mem=64GB,walltime=10:00:00
#PBS -l <SITE_STORAGE_DIRECTIVE>
#PBS -l wd
#PBS -o <ROOT>/outputs/fsrank_diamond_x0.o
#PBS -e <ROOT>/outputs/fsrank_diamond_x0.e
# Locations are read from the environment (see the README); the script stops when one is not set.
ROOT=${ROOT:?set ROOT to the directory of this package}
set -uo pipefail
cd ${ROOT}
export PYTHONPATH=${ROOT}/.tools
  echo "=== Amidar dense ==="; env -u PYTHONHOME python3 -u scripts/v14_fixed_scorer_ranking.py diamond ALE/Amidar-v5 --densedata
echo "=== Assault dense ==="; env -u PYTHONHOME python3 -u scripts/v14_fixed_scorer_ranking.py diamond ALE/Assault-v5 --densedata
echo "=== Asterix dense ==="; env -u PYTHONHOME python3 -u scripts/v14_fixed_scorer_ranking.py diamond ALE/Asterix-v5 --densedata
echo "=== BattleZone dense ==="; env -u PYTHONHOME python3 -u scripts/v14_fixed_scorer_ranking.py diamond ALE/BattleZone-v5 --densedata
echo "=== ChopperCommand dense ==="; env -u PYTHONHOME python3 -u scripts/v14_fixed_scorer_ranking.py diamond ALE/ChopperCommand-v5 --densedata
echo "=== CrazyClimber dense ==="; env -u PYTHONHOME python3 -u scripts/v14_fixed_scorer_ranking.py diamond ALE/CrazyClimber-v5 --densedata
echo "=== DemonAttack dense ==="; env -u PYTHONHOME python3 -u scripts/v14_fixed_scorer_ranking.py diamond ALE/DemonAttack-v5 --densedata
echo "=== Hero dense ==="; env -u PYTHONHOME python3 -u scripts/v14_fixed_scorer_ranking.py diamond ALE/Hero-v5 --densedata
echo "finished $(date -Is)"
