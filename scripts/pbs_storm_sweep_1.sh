#!/bin/bash
#PBS -N cStormS1
#PBS -P <PROJECT>
#PBS -q <GPU_QUEUE>
#PBS -l ngpus=1,ncpus=12,mem=64GB,walltime=10:00:00
#PBS -l <SITE_STORAGE_DIRECTIVE>
#PBS -l wd
#PBS -o <ROOT>/outputs/storm_sweep_1.o
#PBS -e <ROOT>/outputs/storm_sweep_1.e
# Locations are read from the environment (see the README); the script stops when one is not set.
ROOT=${ROOT:?set ROOT to the directory of this package}
STORM_SRC=${STORM_SRC:?set STORM_SRC to the STORM source tree}
STORM_PKGS=${STORM_PKGS:?set STORM_PKGS to the directory of the extra packages STORM needs}
set -uo pipefail
cd ${ROOT}
export PYTHONPATH=${ROOT}/.tools:${STORM_SRC}:${STORM_PKGS}
  echo "=== Gopher ==="; env -u PYTHONHOME python3 -u scripts/v16_storm_sweep.py ALE/Gopher-v5 60 8 1 120
echo "=== Hero ==="; env -u PYTHONHOME python3 -u scripts/v16_storm_sweep.py ALE/Hero-v5 60 8 1 30
echo "=== Jamesbond ==="; env -u PYTHONHOME python3 -u scripts/v16_storm_sweep.py ALE/Jamesbond-v5 60 8 2 60
echo "=== Kangaroo ==="; env -u PYTHONHOME python3 -u scripts/v16_storm_sweep.py ALE/Kangaroo-v5 60 8 1 30
echo "=== Krull ==="; env -u PYTHONHOME python3 -u scripts/v16_storm_sweep.py ALE/Krull-v5 60 8 1 120
echo "=== KungFuMaster ==="; env -u PYTHONHOME python3 -u scripts/v16_storm_sweep.py ALE/KungFuMaster-v5 60 8 1 120
echo "=== MsPacman ==="; env -u PYTHONHOME python3 -u scripts/v16_storm_sweep.py ALE/MsPacman-v5 60 8 1 60
echo "=== Pong ==="; env -u PYTHONHOME python3 -u scripts/v16_storm_sweep.py ALE/Pong-v5 60 8 2 0
echo "=== PrivateEye ==="; env -u PYTHONHOME python3 -u scripts/v16_storm_sweep.py ALE/PrivateEye-v5 60 8 14 0
echo "=== Qbert ==="; env -u PYTHONHOME python3 -u scripts/v16_storm_sweep.py ALE/Qbert-v5 60 8 1 30
echo "=== RoadRunner ==="; env -u PYTHONHOME python3 -u scripts/v16_storm_sweep.py ALE/RoadRunner-v5 60 8 14 30
echo "=== Seaquest ==="; env -u PYTHONHOME python3 -u scripts/v16_storm_sweep.py ALE/Seaquest-v5 60 8 1 60
echo "=== UpNDown ==="; env -u PYTHONHOME python3 -u scripts/v16_storm_sweep.py ALE/UpNDown-v5 60 8 1 30
echo "finished $(date -Is)"
