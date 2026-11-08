#!/bin/bash
#PBS -P <PROJECT>
#PBS -q <CPU_QUEUE>
#PBS -l ncpus=12,mem=48GB,walltime=02:00:00
#PBS -l <SITE_STORAGE_DIRECTIVE>
#PBS -l wd
#PBS -N vA_s
#PBS -o <ROOT>/ext/visited/pbs/vA_s.o
#PBS -e <ROOT>/ext/visited/pbs/vA_s.e
# Locations are read from the environment (see the README); the script stops when one is not set.
ROOT=${ROOT:?set ROOT to the directory of this package}
STORM_SRC=${STORM_SRC:?set STORM_SRC to the STORM source tree}
STORM_PKGS=${STORM_PKGS:?set STORM_PKGS to the directory of the extra packages STORM needs}
set -uo pipefail
cd ${ROOT}
export PYTHONPATH=${ROOT}/.tools:${STORM_SRC}:${STORM_PKGS}
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12 MKL_NUM_THREADS=12
echo "=== storm ALE/Alien-v5 ==="; env -u PYTHONHOME python3 -u ext/visited/v24_visited_states.py storm ALE/Alien-v5
echo "=== storm ALE/Asterix-v5 ==="; env -u PYTHONHOME python3 -u ext/visited/v24_visited_states.py storm ALE/Asterix-v5
echo "=== storm ALE/Boxing-v5 ==="; env -u PYTHONHOME python3 -u ext/visited/v24_visited_states.py storm ALE/Boxing-v5
echo "=== storm ALE/Breakout-v5 ==="; env -u PYTHONHOME python3 -u ext/visited/v24_visited_states.py storm ALE/Breakout-v5
echo "=== storm ALE/CrazyClimber-v5 ==="; env -u PYTHONHOME python3 -u ext/visited/v24_visited_states.py storm ALE/CrazyClimber-v5
echo "=== storm ALE/Frostbite-v5 ==="; env -u PYTHONHOME python3 -u ext/visited/v24_visited_states.py storm ALE/Frostbite-v5
echo "=== storm ALE/Qbert-v5 ==="; env -u PYTHONHOME python3 -u ext/visited/v24_visited_states.py storm ALE/Qbert-v5
echo "=== storm ALE/Seaquest-v5 ==="; env -u PYTHONHOME python3 -u ext/visited/v24_visited_states.py storm ALE/Seaquest-v5
echo "=== job done ==="
