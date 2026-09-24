#!/bin/bash
#PBS -N cNull_s
#PBS -P <PROJECT>
#PBS -q <GPU_QUEUE>
#PBS -l ngpus=1,ncpus=12,mem=64GB,walltime=03:00:00
#PBS -l <SITE_STORAGE_DIRECTIVE>
#PBS -l wd
#PBS -o <ROOT>/outputs/null_storm.o
#PBS -e <ROOT>/outputs/null_storm.e
# Locations are read from the environment (see the README); the script stops when one is not set.
ROOT=${ROOT:?set ROOT to the directory of this package}
STORM_SRC=${STORM_SRC:?set STORM_SRC to the STORM source tree}
STORM_PKGS=${STORM_PKGS:?set STORM_PKGS to the directory of the extra packages STORM needs}
set -uo pipefail
cd ${ROOT}
export PYTHONPATH=${ROOT}/.tools:${STORM_SRC}:${STORM_PKGS}
  echo "=== Alien ==="; env -u PYTHONHOME python3 -u scripts/v18_baseline_null.py storm ALE/Alien-v5
echo "=== Pong ==="; env -u PYTHONHOME python3 -u scripts/v18_baseline_null.py storm ALE/Pong-v5
echo "=== BankHeist ==="; env -u PYTHONHOME python3 -u scripts/v18_baseline_null.py storm ALE/BankHeist-v5
echo "=== Boxing ==="; env -u PYTHONHOME python3 -u scripts/v18_baseline_null.py storm ALE/Boxing-v5
echo "=== Breakout ==="; env -u PYTHONHOME python3 -u scripts/v18_baseline_null.py storm ALE/Breakout-v5
echo "=== Freeway ==="; env -u PYTHONHOME python3 -u scripts/v18_baseline_null.py storm ALE/Freeway-v5
echo "=== Gopher ==="; env -u PYTHONHOME python3 -u scripts/v18_baseline_null.py storm ALE/Gopher-v5
echo "=== Seaquest ==="; env -u PYTHONHOME python3 -u scripts/v18_baseline_null.py storm ALE/Seaquest-v5
echo "=== MsPacman ==="; env -u PYTHONHOME python3 -u scripts/v18_baseline_null.py storm ALE/MsPacman-v5
echo "=== Frostbite ==="; env -u PYTHONHOME python3 -u scripts/v18_baseline_null.py storm ALE/Frostbite-v5
echo "finished $(date -Is)"
