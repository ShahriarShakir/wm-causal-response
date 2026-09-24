#!/bin/bash
#PBS -P <PROJECT>
#PBS -q <GPU_QUEUE>
#PBS -l ngpus=1,ncpus=12,mem=64GB,walltime=04:00:00
#PBS -l <SITE_STORAGE_DIRECTIVE>
#PBS -l wd
#PBS -N rr_v9hi0
#PBS -o <ROOT>/outputs/rerun_reg/rr_v9hi0.o
#PBS -e <ROOT>/outputs/rerun_reg/rr_v9hi0.e
# Locations are read from the environment (see the README); the script stops when one is not set.
ROOT=${ROOT:?set ROOT to the directory of this package}
STORM_SRC=${STORM_SRC:?set STORM_SRC to the STORM source tree}
STORM_PKGS=${STORM_PKGS:?set STORM_PKGS to the directory of the extra packages STORM needs}
set -uo pipefail
cd ${ROOT}
export PYTHONPATH=${ROOT}/.tools:${STORM_SRC}:${STORM_PKGS}
# registration-fix rerun 2026-09-12: IRIS action slice acts[CONTEXT-1:], STORM decode-first (see ccaus/storm_harness.py)
echo "=== scripts/v9_iris_hidden.py ALE/Alien-v5 14 60 ==="; env -u PYTHONHOME python3 -u scripts/v9_iris_hidden.py ALE/Alien-v5 14 60
echo "=== scripts/v9_iris_hidden.py ALE/Pong-v5 2 0 ==="; env -u PYTHONHOME python3 -u scripts/v9_iris_hidden.py ALE/Pong-v5 2 0
echo "=== scripts/v9_iris_hidden.py ALE/BankHeist-v5 12 0 ==="; env -u PYTHONHOME python3 -u scripts/v9_iris_hidden.py ALE/BankHeist-v5 12 0
echo "=== job done ==="
