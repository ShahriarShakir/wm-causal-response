#!/bin/bash
#PBS -P <PROJECT>
#PBS -q <GPU_QUEUE>
#PBS -l ngpus=1,ncpus=12,mem=64GB,walltime=05:00:00
#PBS -l <SITE_STORAGE_DIRECTIVE>
#PBS -l wd
#PBS -N rr_v5I8
#PBS -o <ROOT>/outputs/rerun_reg/rr_v5I8.o
#PBS -e <ROOT>/outputs/rerun_reg/rr_v5I8.e
# Locations are read from the environment (see the README); the script stops when one is not set.
ROOT=${ROOT:?set ROOT to the directory of this package}
STORM_SRC=${STORM_SRC:?set STORM_SRC to the STORM source tree}
STORM_PKGS=${STORM_PKGS:?set STORM_PKGS to the directory of the extra packages STORM needs}
set -uo pipefail
cd ${ROOT}
export PYTHONPATH=${ROOT}/.tools:${STORM_SRC}:${STORM_PKGS}
# registration-fix rerun 2026-09-12: IRIS action slice acts[CONTEXT-1:], STORM decode-first (see ccaus/storm_harness.py)
echo "=== HORIZON=48 scripts/v5_iris_sweep.py ALE/BankHeist-v5 60 40 12 0 ==="; HORIZON=48 env -u PYTHONHOME python3 -u scripts/v5_iris_sweep.py ALE/BankHeist-v5 60 40 12 0
echo "=== HORIZON=48 scripts/v5_iris_sweep.py ALE/Boxing-v5 60 40 1 0 ==="; HORIZON=48 env -u PYTHONHOME python3 -u scripts/v5_iris_sweep.py ALE/Boxing-v5 60 40 1 0
echo "=== HORIZON=48 scripts/v5_iris_sweep.py ALE/Breakout-v5 60 40 3 0 ==="; HORIZON=48 env -u PYTHONHOME python3 -u scripts/v5_iris_sweep.py ALE/Breakout-v5 60 40 3 0
echo "=== HORIZON=48 scripts/v5_iris_sweep.py ALE/Freeway-v5 60 40 1 120 ==="; HORIZON=48 env -u PYTHONHOME python3 -u scripts/v5_iris_sweep.py ALE/Freeway-v5 60 40 1 120
echo "=== job done ==="
