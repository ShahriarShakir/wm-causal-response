#!/bin/bash
#PBS -P <PROJECT>
#PBS -q <GPU_QUEUE>
#PBS -l ngpus=1,ncpus=12,mem=64GB,walltime=05:00:00
#PBS -l <SITE_STORAGE_DIRECTIVE>
#PBS -l wd
#PBS -N rr_v19obi0
#PBS -o <ROOT>/outputs/rerun_reg/rr_v19obi0.o
#PBS -e <ROOT>/outputs/rerun_reg/rr_v19obi0.e
# Locations are read from the environment (see the README); the script stops when one is not set.
ROOT=${ROOT:?set ROOT to the directory of this package}
STORM_SRC=${STORM_SRC:?set STORM_SRC to the STORM source tree}
STORM_PKGS=${STORM_PKGS:?set STORM_PKGS to the directory of the extra packages STORM needs}
set -uo pipefail
cd ${ROOT}
export PYTHONPATH=${ROOT}/.tools:${STORM_SRC}:${STORM_PKGS}
# registration-fix rerun 2026-09-12: IRIS action slice acts[CONTEXT-1:], STORM decode-first (see ccaus/storm_harness.py)
echo "=== scripts/v19_object_level.py iris ALE/Alien-v5 ==="; env -u PYTHONHOME python3 -u scripts/v19_object_level.py iris ALE/Alien-v5
echo "=== scripts/v19_object_level.py iris ALE/Amidar-v5 ==="; env -u PYTHONHOME python3 -u scripts/v19_object_level.py iris ALE/Amidar-v5
echo "=== scripts/v19_object_level.py iris ALE/Assault-v5 ==="; env -u PYTHONHOME python3 -u scripts/v19_object_level.py iris ALE/Assault-v5
echo "=== scripts/v19_object_level.py iris ALE/Asterix-v5 ==="; env -u PYTHONHOME python3 -u scripts/v19_object_level.py iris ALE/Asterix-v5
echo "=== scripts/v19_object_level.py iris ALE/BankHeist-v5 ==="; env -u PYTHONHOME python3 -u scripts/v19_object_level.py iris ALE/BankHeist-v5
echo "=== scripts/v19_object_level.py iris ALE/BattleZone-v5 ==="; env -u PYTHONHOME python3 -u scripts/v19_object_level.py iris ALE/BattleZone-v5
echo "=== scripts/v19_object_level.py iris ALE/Boxing-v5 ==="; env -u PYTHONHOME python3 -u scripts/v19_object_level.py iris ALE/Boxing-v5
echo "=== scripts/v19_object_level.py iris ALE/Breakout-v5 ==="; env -u PYTHONHOME python3 -u scripts/v19_object_level.py iris ALE/Breakout-v5
echo "=== scripts/v19_object_level.py iris ALE/ChopperCommand-v5 ==="; env -u PYTHONHOME python3 -u scripts/v19_object_level.py iris ALE/ChopperCommand-v5
echo "=== scripts/v19_object_level.py iris ALE/CrazyClimber-v5 ==="; env -u PYTHONHOME python3 -u scripts/v19_object_level.py iris ALE/CrazyClimber-v5
echo "=== scripts/v19_object_level.py iris ALE/DemonAttack-v5 ==="; env -u PYTHONHOME python3 -u scripts/v19_object_level.py iris ALE/DemonAttack-v5
echo "=== scripts/v19_object_level.py iris ALE/Freeway-v5 ==="; env -u PYTHONHOME python3 -u scripts/v19_object_level.py iris ALE/Freeway-v5
echo "=== scripts/v19_object_level.py iris ALE/Frostbite-v5 ==="; env -u PYTHONHOME python3 -u scripts/v19_object_level.py iris ALE/Frostbite-v5
echo "=== scripts/v19_object_level.py iris ALE/Gopher-v5 ==="; env -u PYTHONHOME python3 -u scripts/v19_object_level.py iris ALE/Gopher-v5
echo "=== scripts/v19_object_level.py iris ALE/Hero-v5 ==="; env -u PYTHONHOME python3 -u scripts/v19_object_level.py iris ALE/Hero-v5
echo "=== scripts/v19_object_level.py iris ALE/Jamesbond-v5 ==="; env -u PYTHONHOME python3 -u scripts/v19_object_level.py iris ALE/Jamesbond-v5
echo "=== scripts/v19_object_level.py iris ALE/Kangaroo-v5 ==="; env -u PYTHONHOME python3 -u scripts/v19_object_level.py iris ALE/Kangaroo-v5
echo "=== scripts/v19_object_level.py iris ALE/Krull-v5 ==="; env -u PYTHONHOME python3 -u scripts/v19_object_level.py iris ALE/Krull-v5
echo "=== scripts/v19_object_level.py iris ALE/KungFuMaster-v5 ==="; env -u PYTHONHOME python3 -u scripts/v19_object_level.py iris ALE/KungFuMaster-v5
echo "=== scripts/v19_object_level.py iris ALE/MsPacman-v5 ==="; env -u PYTHONHOME python3 -u scripts/v19_object_level.py iris ALE/MsPacman-v5
echo "=== scripts/v19_object_level.py iris ALE/Pong-v5 ==="; env -u PYTHONHOME python3 -u scripts/v19_object_level.py iris ALE/Pong-v5
echo "=== scripts/v19_object_level.py iris ALE/PrivateEye-v5 ==="; env -u PYTHONHOME python3 -u scripts/v19_object_level.py iris ALE/PrivateEye-v5
echo "=== scripts/v19_object_level.py iris ALE/Qbert-v5 ==="; env -u PYTHONHOME python3 -u scripts/v19_object_level.py iris ALE/Qbert-v5
echo "=== scripts/v19_object_level.py iris ALE/RoadRunner-v5 ==="; env -u PYTHONHOME python3 -u scripts/v19_object_level.py iris ALE/RoadRunner-v5
echo "=== scripts/v19_object_level.py iris ALE/Seaquest-v5 ==="; env -u PYTHONHOME python3 -u scripts/v19_object_level.py iris ALE/Seaquest-v5
echo "=== scripts/v19_object_level.py iris ALE/UpNDown-v5 ==="; env -u PYTHONHOME python3 -u scripts/v19_object_level.py iris ALE/UpNDown-v5
echo "=== job done ==="
