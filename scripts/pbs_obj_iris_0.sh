#!/bin/bash
#PBS -N cObj_i0
#PBS -P <PROJECT>
#PBS -q <GPU_QUEUE>
#PBS -l ngpus=1,ncpus=12,mem=64GB,walltime=06:00:00
#PBS -l <SITE_STORAGE_DIRECTIVE>
#PBS -l wd
#PBS -o <ROOT>/outputs/obj_iris_0.o
#PBS -e <ROOT>/outputs/obj_iris_0.e
# Locations are read from the environment (see the README); the script stops when one is not set.
ROOT=${ROOT:?set ROOT to the directory of this package}
STORM_SRC=${STORM_SRC:?set STORM_SRC to the STORM source tree}
STORM_PKGS=${STORM_PKGS:?set STORM_PKGS to the directory of the extra packages STORM needs}
set -uo pipefail
cd ${ROOT}
export PYTHONPATH=${ROOT}/.tools:${STORM_SRC}:${STORM_PKGS}
  echo "=== Alien ==="; env -u PYTHONHOME python3 -u scripts/v19_object_level.py iris ALE/Alien-v5
echo "=== Amidar ==="; env -u PYTHONHOME python3 -u scripts/v19_object_level.py iris ALE/Amidar-v5
echo "=== Assault ==="; env -u PYTHONHOME python3 -u scripts/v19_object_level.py iris ALE/Assault-v5
echo "=== Asterix ==="; env -u PYTHONHOME python3 -u scripts/v19_object_level.py iris ALE/Asterix-v5
echo "=== BankHeist ==="; env -u PYTHONHOME python3 -u scripts/v19_object_level.py iris ALE/BankHeist-v5
echo "=== BattleZone ==="; env -u PYTHONHOME python3 -u scripts/v19_object_level.py iris ALE/BattleZone-v5
echo "=== Boxing ==="; env -u PYTHONHOME python3 -u scripts/v19_object_level.py iris ALE/Boxing-v5
echo "=== Breakout ==="; env -u PYTHONHOME python3 -u scripts/v19_object_level.py iris ALE/Breakout-v5
echo "=== ChopperCommand ==="; env -u PYTHONHOME python3 -u scripts/v19_object_level.py iris ALE/ChopperCommand-v5
echo "=== CrazyClimber ==="; env -u PYTHONHOME python3 -u scripts/v19_object_level.py iris ALE/CrazyClimber-v5
echo "=== DemonAttack ==="; env -u PYTHONHOME python3 -u scripts/v19_object_level.py iris ALE/DemonAttack-v5
echo "=== Freeway ==="; env -u PYTHONHOME python3 -u scripts/v19_object_level.py iris ALE/Freeway-v5
echo "=== Frostbite ==="; env -u PYTHONHOME python3 -u scripts/v19_object_level.py iris ALE/Frostbite-v5
echo "finished $(date -Is)"
