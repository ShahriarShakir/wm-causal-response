#!/bin/bash
#PBS -P <PROJECT>
#PBS -q <GPU_QUEUE>
#PBS -l ngpus=1,ncpus=12,mem=64GB,walltime=05:00:00
#PBS -l <SITE_STORAGE_DIRECTIVE>
#PBS -l wd
#PBS -N cCL_s1
#PBS -o <ROOT>/ext/closed_loop/pbs/cCL_s1.o
#PBS -e <ROOT>/ext/closed_loop/pbs/cCL_s1.e
# Locations are read from the environment (see the README); the script stops when one is not set.
ROOT=${ROOT:?set ROOT to the directory of this package}
STORM_SRC=${STORM_SRC:?set STORM_SRC to the STORM source tree}
STORM_PKGS=${STORM_PKGS:?set STORM_PKGS to the directory of the extra packages STORM needs}
set -uo pipefail
cd ${ROOT}
export PYTHONPATH=${ROOT}/.tools:${STORM_SRC}:${STORM_PKGS}
echo "=== storm ALE/Freeway-v5 ==="; env -u PYTHONHOME python3 -u ext/closed_loop/v20_closed_loop.py storm ALE/Freeway-v5
echo "=== storm ALE/Frostbite-v5 ==="; env -u PYTHONHOME python3 -u ext/closed_loop/v20_closed_loop.py storm ALE/Frostbite-v5
echo "=== storm ALE/MsPacman-v5 ==="; env -u PYTHONHOME python3 -u ext/closed_loop/v20_closed_loop.py storm ALE/MsPacman-v5
echo "=== storm ALE/Qbert-v5 ==="; env -u PYTHONHOME python3 -u ext/closed_loop/v20_closed_loop.py storm ALE/Qbert-v5
echo "=== storm ALE/Seaquest-v5 ==="; env -u PYTHONHOME python3 -u ext/closed_loop/v20_closed_loop.py storm ALE/Seaquest-v5
echo "=== storm ALE/UpNDown-v5 ==="; env -u PYTHONHOME python3 -u ext/closed_loop/v20_closed_loop.py storm ALE/UpNDown-v5
echo "=== job done ==="
