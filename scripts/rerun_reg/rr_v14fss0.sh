#!/bin/bash
#PBS -P <PROJECT>
#PBS -q <GPU_QUEUE>
#PBS -l ngpus=1,ncpus=12,mem=64GB,walltime=04:00:00
#PBS -l <SITE_STORAGE_DIRECTIVE>
#PBS -l wd
#PBS -N rr_v14fss0
#PBS -o <ROOT>/outputs/rerun_reg/rr_v14fss0.o
#PBS -e <ROOT>/outputs/rerun_reg/rr_v14fss0.e
# Locations are read from the environment (see the README); the script stops when one is not set.
ROOT=${ROOT:?set ROOT to the directory of this package}
STORM_SRC=${STORM_SRC:?set STORM_SRC to the STORM source tree}
STORM_PKGS=${STORM_PKGS:?set STORM_PKGS to the directory of the extra packages STORM needs}
set -uo pipefail
cd ${ROOT}
export PYTHONPATH=${ROOT}/.tools:${STORM_SRC}:${STORM_PKGS}
# registration-fix rerun 2026-09-12: IRIS action slice acts[CONTEXT-1:], STORM decode-first (see ccaus/storm_harness.py)
echo "=== scripts/v14_fixed_scorer_ranking.py storm ALE/Alien-v5 --densedata ==="; env -u PYTHONHOME python3 -u scripts/v14_fixed_scorer_ranking.py storm ALE/Alien-v5 --densedata
echo "=== scripts/v14_fixed_scorer_ranking.py storm ALE/Amidar-v5 --densedata ==="; env -u PYTHONHOME python3 -u scripts/v14_fixed_scorer_ranking.py storm ALE/Amidar-v5 --densedata
echo "=== scripts/v14_fixed_scorer_ranking.py storm ALE/Assault-v5 --densedata ==="; env -u PYTHONHOME python3 -u scripts/v14_fixed_scorer_ranking.py storm ALE/Assault-v5 --densedata
echo "=== scripts/v14_fixed_scorer_ranking.py storm ALE/Asterix-v5 --densedata ==="; env -u PYTHONHOME python3 -u scripts/v14_fixed_scorer_ranking.py storm ALE/Asterix-v5 --densedata
echo "=== scripts/v14_fixed_scorer_ranking.py storm ALE/Boxing-v5 --densedata ==="; env -u PYTHONHOME python3 -u scripts/v14_fixed_scorer_ranking.py storm ALE/Boxing-v5 --densedata
echo "=== scripts/v14_fixed_scorer_ranking.py storm ALE/Breakout-v5 --densedata ==="; env -u PYTHONHOME python3 -u scripts/v14_fixed_scorer_ranking.py storm ALE/Breakout-v5 --densedata
echo "=== scripts/v14_fixed_scorer_ranking.py storm ALE/ChopperCommand-v5 --densedata ==="; env -u PYTHONHOME python3 -u scripts/v14_fixed_scorer_ranking.py storm ALE/ChopperCommand-v5 --densedata
echo "=== scripts/v14_fixed_scorer_ranking.py storm ALE/CrazyClimber-v5 --densedata ==="; env -u PYTHONHOME python3 -u scripts/v14_fixed_scorer_ranking.py storm ALE/CrazyClimber-v5 --densedata
echo "=== scripts/v14_fixed_scorer_ranking.py storm ALE/DemonAttack-v5 --densedata ==="; env -u PYTHONHOME python3 -u scripts/v14_fixed_scorer_ranking.py storm ALE/DemonAttack-v5 --densedata
echo "=== scripts/v14_fixed_scorer_ranking.py storm ALE/Freeway-v5 --densedata ==="; env -u PYTHONHOME python3 -u scripts/v14_fixed_scorer_ranking.py storm ALE/Freeway-v5 --densedata
echo "=== scripts/v14_fixed_scorer_ranking.py storm ALE/Frostbite-v5 --densedata ==="; env -u PYTHONHOME python3 -u scripts/v14_fixed_scorer_ranking.py storm ALE/Frostbite-v5 --densedata
echo "=== scripts/v14_fixed_scorer_ranking.py storm ALE/MsPacman-v5 --densedata ==="; env -u PYTHONHOME python3 -u scripts/v14_fixed_scorer_ranking.py storm ALE/MsPacman-v5 --densedata
echo "=== scripts/v14_fixed_scorer_ranking.py storm ALE/Qbert-v5 --densedata ==="; env -u PYTHONHOME python3 -u scripts/v14_fixed_scorer_ranking.py storm ALE/Qbert-v5 --densedata
echo "=== scripts/v14_fixed_scorer_ranking.py storm ALE/Seaquest-v5 --densedata ==="; env -u PYTHONHOME python3 -u scripts/v14_fixed_scorer_ranking.py storm ALE/Seaquest-v5 --densedata
echo "=== scripts/v14_fixed_scorer_ranking.py storm ALE/UpNDown-v5 --densedata ==="; env -u PYTHONHOME python3 -u scripts/v14_fixed_scorer_ranking.py storm ALE/UpNDown-v5 --densedata
echo "=== job done ==="
