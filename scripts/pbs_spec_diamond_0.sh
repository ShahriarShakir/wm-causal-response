#!/bin/bash
#PBS -N cSpec_d0
#PBS -P <PROJECT>
#PBS -q <GPU_QUEUE>
#PBS -l ngpus=1,ncpus=12,mem=64GB,walltime=10:00:00
#PBS -l <SITE_STORAGE_DIRECTIVE>
#PBS -l wd
#PBS -o <ROOT>/outputs/spec_diamond_0.o
#PBS -e <ROOT>/outputs/spec_diamond_0.e
# Locations are read from the environment (see the README); the script stops when one is not set.
ROOT=${ROOT:?set ROOT to the directory of this package}
set -uo pipefail
cd ${ROOT}
export PYTHONPATH=${ROOT}/.tools
  echo "=== Alien ==="; env -u PYTHONHOME python3 -u scripts/v15_specificity.py diamond ALE/Alien-v5
echo "=== Amidar ==="; env -u PYTHONHOME python3 -u scripts/v15_specificity.py diamond ALE/Amidar-v5
echo "=== Assault ==="; env -u PYTHONHOME python3 -u scripts/v15_specificity.py diamond ALE/Assault-v5
echo "=== Asterix ==="; env -u PYTHONHOME python3 -u scripts/v15_specificity.py diamond ALE/Asterix-v5
echo "=== BankHeist ==="; env -u PYTHONHOME python3 -u scripts/v15_specificity.py diamond ALE/BankHeist-v5
echo "=== BattleZone ==="; env -u PYTHONHOME python3 -u scripts/v15_specificity.py diamond ALE/BattleZone-v5
echo "=== Boxing ==="; env -u PYTHONHOME python3 -u scripts/v15_specificity.py diamond ALE/Boxing-v5
echo "=== Breakout ==="; env -u PYTHONHOME python3 -u scripts/v15_specificity.py diamond ALE/Breakout-v5
echo "=== ChopperCommand ==="; env -u PYTHONHOME python3 -u scripts/v15_specificity.py diamond ALE/ChopperCommand-v5
echo "=== CrazyClimber ==="; env -u PYTHONHOME python3 -u scripts/v15_specificity.py diamond ALE/CrazyClimber-v5
echo "=== DemonAttack ==="; env -u PYTHONHOME python3 -u scripts/v15_specificity.py diamond ALE/DemonAttack-v5
echo "=== Freeway ==="; env -u PYTHONHOME python3 -u scripts/v15_specificity.py diamond ALE/Freeway-v5
echo "=== Frostbite ==="; env -u PYTHONHOME python3 -u scripts/v15_specificity.py diamond ALE/Frostbite-v5
echo "finished $(date -Is)"
