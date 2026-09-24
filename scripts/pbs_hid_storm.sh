#!/bin/bash
#PBS -N cHid_s
#PBS -P <PROJECT>
#PBS -q <GPU_QUEUE>
#PBS -l ngpus=1,ncpus=12,mem=64GB,walltime=06:00:00
#PBS -l <SITE_STORAGE_DIRECTIVE>
#PBS -l wd
#PBS -o <ROOT>/outputs/hid_storm.o
#PBS -e <ROOT>/outputs/hid_storm.e
# Locations are read from the environment (see the README); the script stops when one is not set.
ROOT=${ROOT:?set ROOT to the directory of this package}
STORM_SRC=${STORM_SRC:?set STORM_SRC to the STORM source tree}
STORM_PKGS=${STORM_PKGS:?set STORM_PKGS to the directory of the extra packages STORM needs}
set -uo pipefail
cd ${ROOT}
export PYTHONPATH=${ROOT}/.tools:${STORM_SRC}:${STORM_PKGS}
  echo "=== Alien ==="; env -u PYTHONHOME python3 -u scripts/v17_storm_hidden.py ALE/Alien-v5 14 60
echo "=== Pong ==="; env -u PYTHONHOME python3 -u scripts/v17_storm_hidden.py ALE/Pong-v5 2 0
echo "=== BankHeist ==="; env -u PYTHONHOME python3 -u scripts/v17_storm_hidden.py ALE/BankHeist-v5 12 0
echo "=== Boxing ==="; env -u PYTHONHOME python3 -u scripts/v17_storm_hidden.py ALE/Boxing-v5 1 0
echo "=== Breakout ==="; env -u PYTHONHOME python3 -u scripts/v17_storm_hidden.py ALE/Breakout-v5 3 0
echo "=== Freeway ==="; env -u PYTHONHOME python3 -u scripts/v17_storm_hidden.py ALE/Freeway-v5 1 120
echo "finished $(date -Is)"
