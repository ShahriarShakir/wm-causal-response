#!/bin/bash
#PBS -P <PROJECT>
#PBS -q <GPU_QUEUE>
#PBS -l ngpus=1,ncpus=12,mem=64GB,walltime=06:00:00
#PBS -l <SITE_STORAGE_DIRECTIVE>
#PBS -l wd
#PBS -N rr_v15spi1
#PBS -o <ROOT>/outputs/rerun_reg/rr_v15spi1.o
#PBS -e <ROOT>/outputs/rerun_reg/rr_v15spi1.e
# Locations are read from the environment (see the README); the script stops when one is not set.
ROOT=${ROOT:?set ROOT to the directory of this package}
STORM_SRC=${STORM_SRC:?set STORM_SRC to the STORM source tree}
STORM_PKGS=${STORM_PKGS:?set STORM_PKGS to the directory of the extra packages STORM needs}
set -uo pipefail
cd ${ROOT}
export PYTHONPATH=${ROOT}/.tools:${STORM_SRC}:${STORM_PKGS}
# registration-fix rerun 2026-09-12: IRIS action slice acts[CONTEXT-1:], STORM decode-first (see ccaus/storm_harness.py)
echo "=== scripts/v15_specificity.py iris ALE/Gopher-v5 ==="; env -u PYTHONHOME python3 -u scripts/v15_specificity.py iris ALE/Gopher-v5
echo "=== scripts/v15_specificity.py iris ALE/Hero-v5 ==="; env -u PYTHONHOME python3 -u scripts/v15_specificity.py iris ALE/Hero-v5
echo "=== scripts/v15_specificity.py iris ALE/Jamesbond-v5 ==="; env -u PYTHONHOME python3 -u scripts/v15_specificity.py iris ALE/Jamesbond-v5
echo "=== scripts/v15_specificity.py iris ALE/Kangaroo-v5 ==="; env -u PYTHONHOME python3 -u scripts/v15_specificity.py iris ALE/Kangaroo-v5
echo "=== scripts/v15_specificity.py iris ALE/Krull-v5 ==="; env -u PYTHONHOME python3 -u scripts/v15_specificity.py iris ALE/Krull-v5
echo "=== scripts/v15_specificity.py iris ALE/KungFuMaster-v5 ==="; env -u PYTHONHOME python3 -u scripts/v15_specificity.py iris ALE/KungFuMaster-v5
echo "=== scripts/v15_specificity.py iris ALE/MsPacman-v5 ==="; env -u PYTHONHOME python3 -u scripts/v15_specificity.py iris ALE/MsPacman-v5
echo "=== scripts/v15_specificity.py iris ALE/Pong-v5 ==="; env -u PYTHONHOME python3 -u scripts/v15_specificity.py iris ALE/Pong-v5
echo "=== scripts/v15_specificity.py iris ALE/PrivateEye-v5 ==="; env -u PYTHONHOME python3 -u scripts/v15_specificity.py iris ALE/PrivateEye-v5
echo "=== scripts/v15_specificity.py iris ALE/Qbert-v5 ==="; env -u PYTHONHOME python3 -u scripts/v15_specificity.py iris ALE/Qbert-v5
echo "=== scripts/v15_specificity.py iris ALE/RoadRunner-v5 ==="; env -u PYTHONHOME python3 -u scripts/v15_specificity.py iris ALE/RoadRunner-v5
echo "=== scripts/v15_specificity.py iris ALE/Seaquest-v5 ==="; env -u PYTHONHOME python3 -u scripts/v15_specificity.py iris ALE/Seaquest-v5
echo "=== scripts/v15_specificity.py iris ALE/UpNDown-v5 ==="; env -u PYTHONHOME python3 -u scripts/v15_specificity.py iris ALE/UpNDown-v5
echo "=== job done ==="
