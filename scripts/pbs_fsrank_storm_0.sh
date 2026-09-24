#!/bin/bash
#PBS -N cFS_s0
#PBS -P <PROJECT>
#PBS -q <GPU_QUEUE>
#PBS -l ngpus=1,ncpus=12,mem=64GB,walltime=10:00:00
#PBS -l <SITE_STORAGE_DIRECTIVE>
#PBS -l wd
#PBS -o <ROOT>/outputs/fsrank_storm_0.o
#PBS -e <ROOT>/outputs/fsrank_storm_0.e
# Locations are read from the environment (see the README); the script stops when one is not set.
ROOT=${ROOT:?set ROOT to the directory of this package}
STORM_SRC=${STORM_SRC:?set STORM_SRC to the STORM source tree}
STORM_PKGS=${STORM_PKGS:?set STORM_PKGS to the directory of the extra packages STORM needs}
set -uo pipefail
cd ${ROOT}
export PYTHONPATH=${ROOT}/.tools:${STORM_SRC}:${STORM_PKGS}
  echo "=== Alien dense ==="; env -u PYTHONHOME python3 -u scripts/v14_fixed_scorer_ranking.py storm ALE/Alien-v5 --densedata
echo "=== Amidar dense ==="; env -u PYTHONHOME python3 -u scripts/v14_fixed_scorer_ranking.py storm ALE/Amidar-v5 --densedata
echo "=== Assault dense ==="; env -u PYTHONHOME python3 -u scripts/v14_fixed_scorer_ranking.py storm ALE/Assault-v5 --densedata
echo "=== Asterix dense ==="; env -u PYTHONHOME python3 -u scripts/v14_fixed_scorer_ranking.py storm ALE/Asterix-v5 --densedata
echo "=== BankHeist dense ==="; env -u PYTHONHOME python3 -u scripts/v14_fixed_scorer_ranking.py storm ALE/BankHeist-v5 --densedata
echo "=== BattleZone dense ==="; env -u PYTHONHOME python3 -u scripts/v14_fixed_scorer_ranking.py storm ALE/BattleZone-v5 --densedata
echo "=== Boxing dense ==="; env -u PYTHONHOME python3 -u scripts/v14_fixed_scorer_ranking.py storm ALE/Boxing-v5 --densedata
echo "=== Breakout dense ==="; env -u PYTHONHOME python3 -u scripts/v14_fixed_scorer_ranking.py storm ALE/Breakout-v5 --densedata
echo "=== ChopperCommand dense ==="; env -u PYTHONHOME python3 -u scripts/v14_fixed_scorer_ranking.py storm ALE/ChopperCommand-v5 --densedata
echo "=== CrazyClimber dense ==="; env -u PYTHONHOME python3 -u scripts/v14_fixed_scorer_ranking.py storm ALE/CrazyClimber-v5 --densedata
echo "=== DemonAttack dense ==="; env -u PYTHONHOME python3 -u scripts/v14_fixed_scorer_ranking.py storm ALE/DemonAttack-v5 --densedata
echo "=== Freeway dense ==="; env -u PYTHONHOME python3 -u scripts/v14_fixed_scorer_ranking.py storm ALE/Freeway-v5 --densedata
echo "=== Frostbite dense ==="; env -u PYTHONHOME python3 -u scripts/v14_fixed_scorer_ranking.py storm ALE/Frostbite-v5 --densedata
echo "finished $(date -Is)"
