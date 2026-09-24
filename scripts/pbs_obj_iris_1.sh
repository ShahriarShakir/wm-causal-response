#!/bin/bash
#PBS -N cObj_i1
#PBS -P <PROJECT>
#PBS -q <GPU_QUEUE>
#PBS -l ngpus=1,ncpus=12,mem=64GB,walltime=06:00:00
#PBS -l <SITE_STORAGE_DIRECTIVE>
#PBS -l wd
#PBS -o <ROOT>/outputs/obj_iris_1.o
#PBS -e <ROOT>/outputs/obj_iris_1.e
# Locations are read from the environment (see the README); the script stops when one is not set.
ROOT=${ROOT:?set ROOT to the directory of this package}
STORM_SRC=${STORM_SRC:?set STORM_SRC to the STORM source tree}
STORM_PKGS=${STORM_PKGS:?set STORM_PKGS to the directory of the extra packages STORM needs}
set -uo pipefail
cd ${ROOT}
export PYTHONPATH=${ROOT}/.tools:${STORM_SRC}:${STORM_PKGS}
  echo "=== Gopher ==="; env -u PYTHONHOME python3 -u scripts/v19_object_level.py iris ALE/Gopher-v5
echo "=== Hero ==="; env -u PYTHONHOME python3 -u scripts/v19_object_level.py iris ALE/Hero-v5
echo "=== Jamesbond ==="; env -u PYTHONHOME python3 -u scripts/v19_object_level.py iris ALE/Jamesbond-v5
echo "=== Kangaroo ==="; env -u PYTHONHOME python3 -u scripts/v19_object_level.py iris ALE/Kangaroo-v5
echo "=== Krull ==="; env -u PYTHONHOME python3 -u scripts/v19_object_level.py iris ALE/Krull-v5
echo "=== KungFuMaster ==="; env -u PYTHONHOME python3 -u scripts/v19_object_level.py iris ALE/KungFuMaster-v5
echo "=== MsPacman ==="; env -u PYTHONHOME python3 -u scripts/v19_object_level.py iris ALE/MsPacman-v5
echo "=== Pong ==="; env -u PYTHONHOME python3 -u scripts/v19_object_level.py iris ALE/Pong-v5
echo "=== PrivateEye ==="; env -u PYTHONHOME python3 -u scripts/v19_object_level.py iris ALE/PrivateEye-v5
echo "=== Qbert ==="; env -u PYTHONHOME python3 -u scripts/v19_object_level.py iris ALE/Qbert-v5
echo "=== RoadRunner ==="; env -u PYTHONHOME python3 -u scripts/v19_object_level.py iris ALE/RoadRunner-v5
echo "=== Seaquest ==="; env -u PYTHONHOME python3 -u scripts/v19_object_level.py iris ALE/Seaquest-v5
echo "=== UpNDown ==="; env -u PYTHONHOME python3 -u scripts/v19_object_level.py iris ALE/UpNDown-v5
echo "finished $(date -Is)"
