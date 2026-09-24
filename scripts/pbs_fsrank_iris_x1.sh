#!/bin/bash
#PBS -N cFSX_i1
#PBS -P <PROJECT>
#PBS -q <GPU_QUEUE>
#PBS -l ngpus=1,ncpus=12,mem=64GB,walltime=10:00:00
#PBS -l <SITE_STORAGE_DIRECTIVE>
#PBS -l wd
#PBS -o <ROOT>/outputs/fsrank_iris_x1.o
#PBS -e <ROOT>/outputs/fsrank_iris_x1.e
# Locations are read from the environment (see the README); the script stops when one is not set.
ROOT=${ROOT:?set ROOT to the directory of this package}
set -uo pipefail
cd ${ROOT}
export PYTHONPATH=${ROOT}/.tools
  echo "=== Jamesbond dense ==="; env -u PYTHONHOME python3 -u scripts/v14_fixed_scorer_ranking.py iris ALE/Jamesbond-v5 --densedata
echo "=== Kangaroo dense ==="; env -u PYTHONHOME python3 -u scripts/v14_fixed_scorer_ranking.py iris ALE/Kangaroo-v5 --densedata
echo "=== Krull dense ==="; env -u PYTHONHOME python3 -u scripts/v14_fixed_scorer_ranking.py iris ALE/Krull-v5 --densedata
echo "=== KungFuMaster dense ==="; env -u PYTHONHOME python3 -u scripts/v14_fixed_scorer_ranking.py iris ALE/KungFuMaster-v5 --densedata
echo "=== PrivateEye dense ==="; env -u PYTHONHOME python3 -u scripts/v14_fixed_scorer_ranking.py iris ALE/PrivateEye-v5 --densedata
echo "=== Qbert dense ==="; env -u PYTHONHOME python3 -u scripts/v14_fixed_scorer_ranking.py iris ALE/Qbert-v5 --densedata
echo "=== RoadRunner dense ==="; env -u PYTHONHOME python3 -u scripts/v14_fixed_scorer_ranking.py iris ALE/RoadRunner-v5 --densedata
echo "=== UpNDown dense ==="; env -u PYTHONHOME python3 -u scripts/v14_fixed_scorer_ranking.py iris ALE/UpNDown-v5 --densedata
echo "finished $(date -Is)"
