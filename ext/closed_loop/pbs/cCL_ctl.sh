#!/bin/bash
#PBS -P <PROJECT>
#PBS -q <GPU_QUEUE>
#PBS -l ngpus=1,ncpus=12,mem=64GB,walltime=04:00:00
#PBS -l <SITE_STORAGE_DIRECTIVE>
#PBS -l wd
#PBS -N cCL_ctl
#PBS -o <ROOT>/ext/closed_loop/pbs/cCL_ctl.o
#PBS -e <ROOT>/ext/closed_loop/pbs/cCL_ctl.e
# Locations are read from the environment (see the README); the script stops when one is not set.
ROOT=${ROOT:?set ROOT to the directory of this package}
STORM_SRC=${STORM_SRC:?set STORM_SRC to the STORM source tree}
STORM_PKGS=${STORM_PKGS:?set STORM_PKGS to the directory of the extra packages STORM needs}
set -uo pipefail
cd ${ROOT}
export PYTHONPATH=${ROOT}/.tools:${STORM_SRC}:${STORM_PKGS}
echo "=== random ALE/Alien-v5 ==="; env -u PYTHONHOME python3 -u ext/closed_loop/v20_closed_loop.py random ALE/Alien-v5
echo "=== oracle ALE/Alien-v5 ==="; env -u PYTHONHOME python3 -u ext/closed_loop/v20_closed_loop.py oracle ALE/Alien-v5
echo "=== random ALE/Amidar-v5 ==="; env -u PYTHONHOME python3 -u ext/closed_loop/v20_closed_loop.py random ALE/Amidar-v5
echo "=== oracle ALE/Amidar-v5 ==="; env -u PYTHONHOME python3 -u ext/closed_loop/v20_closed_loop.py oracle ALE/Amidar-v5
echo "=== random ALE/Asterix-v5 ==="; env -u PYTHONHOME python3 -u ext/closed_loop/v20_closed_loop.py random ALE/Asterix-v5
echo "=== oracle ALE/Asterix-v5 ==="; env -u PYTHONHOME python3 -u ext/closed_loop/v20_closed_loop.py oracle ALE/Asterix-v5
echo "=== random ALE/Boxing-v5 ==="; env -u PYTHONHOME python3 -u ext/closed_loop/v20_closed_loop.py random ALE/Boxing-v5
echo "=== oracle ALE/Boxing-v5 ==="; env -u PYTHONHOME python3 -u ext/closed_loop/v20_closed_loop.py oracle ALE/Boxing-v5
echo "=== random ALE/Breakout-v5 ==="; env -u PYTHONHOME python3 -u ext/closed_loop/v20_closed_loop.py random ALE/Breakout-v5
echo "=== oracle ALE/Breakout-v5 ==="; env -u PYTHONHOME python3 -u ext/closed_loop/v20_closed_loop.py oracle ALE/Breakout-v5
echo "=== random ALE/CrazyClimber-v5 ==="; env -u PYTHONHOME python3 -u ext/closed_loop/v20_closed_loop.py random ALE/CrazyClimber-v5
echo "=== oracle ALE/CrazyClimber-v5 ==="; env -u PYTHONHOME python3 -u ext/closed_loop/v20_closed_loop.py oracle ALE/CrazyClimber-v5
echo "=== random ALE/Freeway-v5 ==="; env -u PYTHONHOME python3 -u ext/closed_loop/v20_closed_loop.py random ALE/Freeway-v5
echo "=== oracle ALE/Freeway-v5 ==="; env -u PYTHONHOME python3 -u ext/closed_loop/v20_closed_loop.py oracle ALE/Freeway-v5
echo "=== random ALE/Frostbite-v5 ==="; env -u PYTHONHOME python3 -u ext/closed_loop/v20_closed_loop.py random ALE/Frostbite-v5
echo "=== oracle ALE/Frostbite-v5 ==="; env -u PYTHONHOME python3 -u ext/closed_loop/v20_closed_loop.py oracle ALE/Frostbite-v5
echo "=== random ALE/MsPacman-v5 ==="; env -u PYTHONHOME python3 -u ext/closed_loop/v20_closed_loop.py random ALE/MsPacman-v5
echo "=== oracle ALE/MsPacman-v5 ==="; env -u PYTHONHOME python3 -u ext/closed_loop/v20_closed_loop.py oracle ALE/MsPacman-v5
echo "=== random ALE/Qbert-v5 ==="; env -u PYTHONHOME python3 -u ext/closed_loop/v20_closed_loop.py random ALE/Qbert-v5
echo "=== oracle ALE/Qbert-v5 ==="; env -u PYTHONHOME python3 -u ext/closed_loop/v20_closed_loop.py oracle ALE/Qbert-v5
echo "=== random ALE/Seaquest-v5 ==="; env -u PYTHONHOME python3 -u ext/closed_loop/v20_closed_loop.py random ALE/Seaquest-v5
echo "=== oracle ALE/Seaquest-v5 ==="; env -u PYTHONHOME python3 -u ext/closed_loop/v20_closed_loop.py oracle ALE/Seaquest-v5
echo "=== random ALE/UpNDown-v5 ==="; env -u PYTHONHOME python3 -u ext/closed_loop/v20_closed_loop.py random ALE/UpNDown-v5
echo "=== oracle ALE/UpNDown-v5 ==="; env -u PYTHONHOME python3 -u ext/closed_loop/v20_closed_loop.py oracle ALE/UpNDown-v5
echo "=== job done ==="
