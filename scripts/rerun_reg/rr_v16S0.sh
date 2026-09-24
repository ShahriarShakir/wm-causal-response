#!/bin/bash
#PBS -P <PROJECT>
#PBS -q <GPU_QUEUE>
#PBS -l ngpus=1,ncpus=12,mem=64GB,walltime=04:00:00
#PBS -l <SITE_STORAGE_DIRECTIVE>
#PBS -l wd
#PBS -N rr_v16S0
#PBS -o <ROOT>/outputs/rerun_reg/rr_v16S0.o
#PBS -e <ROOT>/outputs/rerun_reg/rr_v16S0.e
# Locations are read from the environment (see the README); the script stops when one is not set.
ROOT=${ROOT:?set ROOT to the directory of this package}
STORM_SRC=${STORM_SRC:?set STORM_SRC to the STORM source tree}
STORM_PKGS=${STORM_PKGS:?set STORM_PKGS to the directory of the extra packages STORM needs}
set -uo pipefail
cd ${ROOT}
export PYTHONPATH=${ROOT}/.tools:${STORM_SRC}:${STORM_PKGS}
# registration-fix rerun 2026-09-12: IRIS action slice acts[CONTEXT-1:], STORM decode-first (see ccaus/storm_harness.py)
echo "=== scripts/v16_storm_sweep.py ALE/Alien-v5 60 8 14 60 ==="; env -u PYTHONHOME python3 -u scripts/v16_storm_sweep.py ALE/Alien-v5 60 8 14 60
echo "=== scripts/v16_storm_sweep.py ALE/Amidar-v5 60 8 1 60 ==="; env -u PYTHONHOME python3 -u scripts/v16_storm_sweep.py ALE/Amidar-v5 60 8 1 60
echo "=== scripts/v16_storm_sweep.py ALE/Assault-v5 60 8 2 0 ==="; env -u PYTHONHOME python3 -u scripts/v16_storm_sweep.py ALE/Assault-v5 60 8 2 0
echo "=== scripts/v16_storm_sweep.py ALE/Asterix-v5 60 8 3 0 ==="; env -u PYTHONHOME python3 -u scripts/v16_storm_sweep.py ALE/Asterix-v5 60 8 3 0
echo "=== scripts/v16_storm_sweep.py ALE/BankHeist-v5 60 8 12 0 ==="; env -u PYTHONHOME python3 -u scripts/v16_storm_sweep.py ALE/BankHeist-v5 60 8 12 0
echo "=== scripts/v16_storm_sweep.py ALE/BattleZone-v5 60 8 1 30 ==="; env -u PYTHONHOME python3 -u scripts/v16_storm_sweep.py ALE/BattleZone-v5 60 8 1 30
echo "=== scripts/v16_storm_sweep.py ALE/Boxing-v5 60 8 1 0 ==="; env -u PYTHONHOME python3 -u scripts/v16_storm_sweep.py ALE/Boxing-v5 60 8 1 0
echo "=== scripts/v16_storm_sweep.py ALE/Breakout-v5 60 8 3 0 ==="; env -u PYTHONHOME python3 -u scripts/v16_storm_sweep.py ALE/Breakout-v5 60 8 3 0
echo "=== scripts/v16_storm_sweep.py ALE/ChopperCommand-v5 60 8 1 120 ==="; env -u PYTHONHOME python3 -u scripts/v16_storm_sweep.py ALE/ChopperCommand-v5 60 8 1 120
echo "=== scripts/v16_storm_sweep.py ALE/CrazyClimber-v5 60 8 1 120 ==="; env -u PYTHONHOME python3 -u scripts/v16_storm_sweep.py ALE/CrazyClimber-v5 60 8 1 120
echo "=== scripts/v16_storm_sweep.py ALE/DemonAttack-v5 60 8 3 30 ==="; env -u PYTHONHOME python3 -u scripts/v16_storm_sweep.py ALE/DemonAttack-v5 60 8 3 30
echo "=== scripts/v16_storm_sweep.py ALE/Freeway-v5 60 8 1 120 ==="; env -u PYTHONHOME python3 -u scripts/v16_storm_sweep.py ALE/Freeway-v5 60 8 1 120
echo "=== job done ==="
