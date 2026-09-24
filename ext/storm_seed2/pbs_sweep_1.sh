#!/bin/bash
#PBS -N cS2sw1
#PBS -P <PROJECT>
#PBS -q <GPU_QUEUE>
#PBS -l ngpus=1,ncpus=12,mem=64GB,walltime=10:00:00
#PBS -l <SITE_STORAGE_DIRECTIVE>
#PBS -l wd
#PBS -o <ROOT>/ext/outputs/s2_sweep_1.o
#PBS -e <ROOT>/ext/outputs/s2_sweep_1.e
# Locations are read from the environment (see the README); the script stops when one is not set.
ROOT=${ROOT:?set ROOT to the directory of this package}
STORM_SRC=${STORM_SRC:?set STORM_SRC to the STORM source tree}
STORM_PKGS=${STORM_PKGS:?set STORM_PKGS to the directory of the extra packages STORM needs}
set -uo pipefail
cd ${ROOT}
export PYTHONPATH=${ROOT}/.tools:${STORM_SRC}:${STORM_PKGS}
  echo "=== Gopher ==="; STORM_OUT=${ROOT}/ext/outputs STORM_CKPT=${STORM_SRC}/ckpt/Gopher-life_done-wm_2L512D8H-100k-seed2/world_model_100000.pth env -u PYTHONHOME python3 -u scripts/v16_storm_sweep.py ALE/Gopher-v5 60 8 1 120
echo "=== Hero ==="; STORM_OUT=${ROOT}/ext/outputs STORM_CKPT=${STORM_SRC}/ckpt/Hero-life_done-wm_2L512D8H-100k-seed2/world_model_100000.pth env -u PYTHONHOME python3 -u scripts/v16_storm_sweep.py ALE/Hero-v5 60 8 1 30
echo "=== Jamesbond ==="; STORM_OUT=${ROOT}/ext/outputs STORM_CKPT=${STORM_SRC}/ckpt/Jamesbond-life_done-wm_2L512D8H-100k-seed2/world_model_100000.pth env -u PYTHONHOME python3 -u scripts/v16_storm_sweep.py ALE/Jamesbond-v5 60 8 2 60
echo "=== Kangaroo ==="; STORM_OUT=${ROOT}/ext/outputs STORM_CKPT=${STORM_SRC}/ckpt/Kangaroo-life_done-wm_2L512D8H-100k-seed2/world_model_100000.pth env -u PYTHONHOME python3 -u scripts/v16_storm_sweep.py ALE/Kangaroo-v5 60 8 1 30
echo "=== Krull ==="; STORM_OUT=${ROOT}/ext/outputs STORM_CKPT=${STORM_SRC}/ckpt/Krull-life_done-wm_2L512D8H-100k-seed2/world_model_100000.pth env -u PYTHONHOME python3 -u scripts/v16_storm_sweep.py ALE/Krull-v5 60 8 1 120
echo "=== KungFuMaster ==="; STORM_OUT=${ROOT}/ext/outputs STORM_CKPT=${STORM_SRC}/ckpt/KungFuMaster-life_done-wm_2L512D8H-100k-seed2/world_model_100000.pth env -u PYTHONHOME python3 -u scripts/v16_storm_sweep.py ALE/KungFuMaster-v5 60 8 1 120
echo "=== MsPacman ==="; STORM_OUT=${ROOT}/ext/outputs STORM_CKPT=${STORM_SRC}/ckpt/MsPacman-life_done-wm_2L512D8H-100k-seed2/world_model_100000.pth env -u PYTHONHOME python3 -u scripts/v16_storm_sweep.py ALE/MsPacman-v5 60 8 1 60
echo "=== Pong ==="; STORM_OUT=${ROOT}/ext/outputs STORM_CKPT=${STORM_SRC}/ckpt/Pong-life_done-wm_2L512D8H-100k-seed2/world_model_100000.pth env -u PYTHONHOME python3 -u scripts/v16_storm_sweep.py ALE/Pong-v5 60 8 2 0
echo "=== PrivateEye ==="; STORM_OUT=${ROOT}/ext/outputs STORM_CKPT=${STORM_SRC}/ckpt/PrivateEye-life_done-wm_2L512D8H-100k-seed2/world_model_100000.pth env -u PYTHONHOME python3 -u scripts/v16_storm_sweep.py ALE/PrivateEye-v5 60 8 14 0
echo "=== Qbert ==="; STORM_OUT=${ROOT}/ext/outputs STORM_CKPT=${STORM_SRC}/ckpt/Qbert-life_done-wm_2L512D8H-100k-seed2/world_model_100000.pth env -u PYTHONHOME python3 -u scripts/v16_storm_sweep.py ALE/Qbert-v5 60 8 1 30
echo "=== RoadRunner ==="; STORM_OUT=${ROOT}/ext/outputs STORM_CKPT=${STORM_SRC}/ckpt/RoadRunner-life_done-wm_2L512D8H-100k-seed2/world_model_100000.pth env -u PYTHONHOME python3 -u scripts/v16_storm_sweep.py ALE/RoadRunner-v5 60 8 14 30
echo "=== Seaquest ==="; STORM_OUT=${ROOT}/ext/outputs STORM_CKPT=${STORM_SRC}/ckpt/Seaquest-life_done-wm_2L512D8H-100k-seed2/world_model_100000.pth env -u PYTHONHOME python3 -u scripts/v16_storm_sweep.py ALE/Seaquest-v5 60 8 1 60
echo "=== UpNDown ==="; STORM_OUT=${ROOT}/ext/outputs STORM_CKPT=${STORM_SRC}/ckpt/UpNDown-life_done-wm_2L512D8H-100k-seed2/world_model_100000.pth env -u PYTHONHOME python3 -u scripts/v16_storm_sweep.py ALE/UpNDown-v5 60 8 1 30
echo "finished $(date -Is)"
