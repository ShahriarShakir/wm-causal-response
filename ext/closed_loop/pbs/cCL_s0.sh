#!/bin/bash
#PBS -P <PROJECT>
#PBS -q <GPU_QUEUE>
#PBS -l ngpus=1,ncpus=12,mem=64GB,walltime=05:00:00
#PBS -l <SITE_STORAGE_DIRECTIVE>
#PBS -l wd
#PBS -N cCL_s0
#PBS -o <ROOT>/ext/closed_loop/pbs/cCL_s0.o
#PBS -e <ROOT>/ext/closed_loop/pbs/cCL_s0.e
# Locations are read from the environment (see the README); the script stops when one is not set.
ROOT=${ROOT:?set ROOT to the directory of this package}
STORM_SRC=${STORM_SRC:?set STORM_SRC to the STORM source tree}
STORM_PKGS=${STORM_PKGS:?set STORM_PKGS to the directory of the extra packages STORM needs}
set -uo pipefail
cd ${ROOT}
export PYTHONPATH=${ROOT}/.tools:${STORM_SRC}:${STORM_PKGS}
echo "=== storm ALE/Alien-v5 ==="; env -u PYTHONHOME python3 -u ext/closed_loop/v20_closed_loop.py storm ALE/Alien-v5
echo "=== storm ALE/Amidar-v5 ==="; env -u PYTHONHOME python3 -u ext/closed_loop/v20_closed_loop.py storm ALE/Amidar-v5
echo "=== storm ALE/Asterix-v5 ==="; env -u PYTHONHOME python3 -u ext/closed_loop/v20_closed_loop.py storm ALE/Asterix-v5
echo "=== storm ALE/Boxing-v5 ==="; env -u PYTHONHOME python3 -u ext/closed_loop/v20_closed_loop.py storm ALE/Boxing-v5
echo "=== storm ALE/Breakout-v5 ==="; env -u PYTHONHOME python3 -u ext/closed_loop/v20_closed_loop.py storm ALE/Breakout-v5
echo "=== storm ALE/CrazyClimber-v5 ==="; env -u PYTHONHOME python3 -u ext/closed_loop/v20_closed_loop.py storm ALE/CrazyClimber-v5
echo "=== job done ==="
