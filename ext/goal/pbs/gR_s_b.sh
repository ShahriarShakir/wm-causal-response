#!/bin/bash
#PBS -P <PROJECT>
#PBS -q <CPU_QUEUE>
#PBS -l ncpus=12,mem=48GB,walltime=01:00:00
#PBS -l <SITE_STORAGE_DIRECTIVE>
#PBS -l wd
#PBS -N gR_s_b
#PBS -o <ROOT>/ext/goal/pbs/gR_s_b.o
#PBS -e <ROOT>/ext/goal/pbs/gR_s_b.e
# Locations are read from the environment (see the README); the script stops when one is not set.
ROOT=${ROOT:?set ROOT to the directory of this package}
STORM_SRC=${STORM_SRC:?set STORM_SRC to the STORM source tree}
STORM_PKGS=${STORM_PKGS:?set STORM_PKGS to the directory of the extra packages STORM needs}
set -uo pipefail
cd ${ROOT}
export PYTHONPATH=${ROOT}/.tools:${STORM_SRC}:${STORM_PKGS}
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12 MKL_NUM_THREADS=12
echo "=== storm ALE/Gopher-v5 ==="; env -u PYTHONHOME python3 -u ext/goal/v25_goal_reaching.py storm ALE/Gopher-v5
echo "=== storm ALE/Hero-v5 ==="; env -u PYTHONHOME python3 -u ext/goal/v25_goal_reaching.py storm ALE/Hero-v5
echo "=== storm ALE/Jamesbond-v5 ==="; env -u PYTHONHOME python3 -u ext/goal/v25_goal_reaching.py storm ALE/Jamesbond-v5
echo "=== storm ALE/Kangaroo-v5 ==="; env -u PYTHONHOME python3 -u ext/goal/v25_goal_reaching.py storm ALE/Kangaroo-v5
echo "=== storm ALE/Krull-v5 ==="; env -u PYTHONHOME python3 -u ext/goal/v25_goal_reaching.py storm ALE/Krull-v5
echo "=== storm ALE/KungFuMaster-v5 ==="; env -u PYTHONHOME python3 -u ext/goal/v25_goal_reaching.py storm ALE/KungFuMaster-v5
echo "=== storm ALE/MsPacman-v5 ==="; env -u PYTHONHOME python3 -u ext/goal/v25_goal_reaching.py storm ALE/MsPacman-v5
echo "=== storm ALE/Pong-v5 ==="; env -u PYTHONHOME python3 -u ext/goal/v25_goal_reaching.py storm ALE/Pong-v5
echo "=== storm ALE/PrivateEye-v5 ==="; env -u PYTHONHOME python3 -u ext/goal/v25_goal_reaching.py storm ALE/PrivateEye-v5
echo "=== storm ALE/Qbert-v5 ==="; env -u PYTHONHOME python3 -u ext/goal/v25_goal_reaching.py storm ALE/Qbert-v5
echo "=== storm ALE/RoadRunner-v5 ==="; env -u PYTHONHOME python3 -u ext/goal/v25_goal_reaching.py storm ALE/RoadRunner-v5
echo "=== storm ALE/Seaquest-v5 ==="; env -u PYTHONHOME python3 -u ext/goal/v25_goal_reaching.py storm ALE/Seaquest-v5
echo "=== storm ALE/UpNDown-v5 ==="; env -u PYTHONHOME python3 -u ext/goal/v25_goal_reaching.py storm ALE/UpNDown-v5
echo "=== job done ==="
