#!/bin/bash
#PBS -P <PROJECT>
#PBS -q <GPU_QUEUE>
#PBS -l ngpus=1,ncpus=12,mem=64GB,walltime=02:00:00
#PBS -l <SITE_STORAGE_DIRECTIVE>
#PBS -l wd
#PBS -N rr_v17hs0
#PBS -o <ROOT>/outputs/rerun_reg/rr_v17hs0.o
#PBS -e <ROOT>/outputs/rerun_reg/rr_v17hs0.e
# Locations are read from the environment (see the README); the script stops when one is not set.
ROOT=${ROOT:?set ROOT to the directory of this package}
STORM_SRC=${STORM_SRC:?set STORM_SRC to the STORM source tree}
STORM_PKGS=${STORM_PKGS:?set STORM_PKGS to the directory of the extra packages STORM needs}
set -uo pipefail
cd ${ROOT}
export PYTHONPATH=${ROOT}/.tools:${STORM_SRC}:${STORM_PKGS}
# registration-fix rerun 2026-09-12: IRIS action slice acts[CONTEXT-1:], STORM decode-first (see ccaus/storm_harness.py)
echo "=== scripts/v17_storm_hidden.py ALE/Alien-v5 14 60 ==="; env -u PYTHONHOME python3 -u scripts/v17_storm_hidden.py ALE/Alien-v5 14 60
echo "=== scripts/v17_storm_hidden.py ALE/Pong-v5 2 0 ==="; env -u PYTHONHOME python3 -u scripts/v17_storm_hidden.py ALE/Pong-v5 2 0
echo "=== scripts/v17_storm_hidden.py ALE/BankHeist-v5 12 0 ==="; env -u PYTHONHOME python3 -u scripts/v17_storm_hidden.py ALE/BankHeist-v5 12 0
echo "=== scripts/v17_storm_hidden.py ALE/Boxing-v5 1 0 ==="; env -u PYTHONHOME python3 -u scripts/v17_storm_hidden.py ALE/Boxing-v5 1 0
echo "=== scripts/v17_storm_hidden.py ALE/Breakout-v5 3 0 ==="; env -u PYTHONHOME python3 -u scripts/v17_storm_hidden.py ALE/Breakout-v5 3 0
echo "=== scripts/v17_storm_hidden.py ALE/Freeway-v5 1 120 ==="; env -u PYTHONHOME python3 -u scripts/v17_storm_hidden.py ALE/Freeway-v5 1 120
echo "=== job done ==="
