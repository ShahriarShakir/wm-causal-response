#!/bin/bash
#PBS -N cFS_s1
#PBS -P <PROJECT>
#PBS -q <GPU_QUEUE>
#PBS -l ngpus=1,ncpus=12,mem=64GB,walltime=10:00:00
#PBS -l <SITE_STORAGE_DIRECTIVE>
#PBS -l wd
#PBS -o <ROOT>/outputs/fsrank_storm_1.o
#PBS -e <ROOT>/outputs/fsrank_storm_1.e
# Locations are read from the environment (see the README); the script stops when one is not set.
ROOT=${ROOT:?set ROOT to the directory of this package}
STORM_SRC=${STORM_SRC:?set STORM_SRC to the STORM source tree}
STORM_PKGS=${STORM_PKGS:?set STORM_PKGS to the directory of the extra packages STORM needs}
set -uo pipefail
cd ${ROOT}
export PYTHONPATH=${ROOT}/.tools:${STORM_SRC}:${STORM_PKGS}
  echo "=== Gopher dense ==="; env -u PYTHONHOME python3 -u scripts/v14_fixed_scorer_ranking.py storm ALE/Gopher-v5 --densedata
echo "=== Hero dense ==="; env -u PYTHONHOME python3 -u scripts/v14_fixed_scorer_ranking.py storm ALE/Hero-v5 --densedata
echo "=== Jamesbond dense ==="; env -u PYTHONHOME python3 -u scripts/v14_fixed_scorer_ranking.py storm ALE/Jamesbond-v5 --densedata
echo "=== Kangaroo dense ==="; env -u PYTHONHOME python3 -u scripts/v14_fixed_scorer_ranking.py storm ALE/Kangaroo-v5 --densedata
echo "=== Krull dense ==="; env -u PYTHONHOME python3 -u scripts/v14_fixed_scorer_ranking.py storm ALE/Krull-v5 --densedata
echo "=== KungFuMaster dense ==="; env -u PYTHONHOME python3 -u scripts/v14_fixed_scorer_ranking.py storm ALE/KungFuMaster-v5 --densedata
echo "=== MsPacman dense ==="; env -u PYTHONHOME python3 -u scripts/v14_fixed_scorer_ranking.py storm ALE/MsPacman-v5 --densedata
echo "=== Pong dense ==="; env -u PYTHONHOME python3 -u scripts/v14_fixed_scorer_ranking.py storm ALE/Pong-v5 --densedata
echo "=== PrivateEye dense ==="; env -u PYTHONHOME python3 -u scripts/v14_fixed_scorer_ranking.py storm ALE/PrivateEye-v5 --densedata
echo "=== Qbert dense ==="; env -u PYTHONHOME python3 -u scripts/v14_fixed_scorer_ranking.py storm ALE/Qbert-v5 --densedata
echo "=== RoadRunner dense ==="; env -u PYTHONHOME python3 -u scripts/v14_fixed_scorer_ranking.py storm ALE/RoadRunner-v5 --densedata
echo "=== Seaquest dense ==="; env -u PYTHONHOME python3 -u scripts/v14_fixed_scorer_ranking.py storm ALE/Seaquest-v5 --densedata
echo "=== UpNDown dense ==="; env -u PYTHONHOME python3 -u scripts/v14_fixed_scorer_ranking.py storm ALE/UpNDown-v5 --densedata
echo "finished $(date -Is)"
