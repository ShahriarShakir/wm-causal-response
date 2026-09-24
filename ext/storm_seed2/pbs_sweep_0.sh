#!/bin/bash
#PBS -N cS2sw0
#PBS -P <PROJECT>
#PBS -q <GPU_QUEUE>
#PBS -l ngpus=1,ncpus=12,mem=64GB,walltime=10:00:00
#PBS -l <SITE_STORAGE_DIRECTIVE>
#PBS -l wd
#PBS -o <ROOT>/ext/outputs/s2_sweep_0.o
#PBS -e <ROOT>/ext/outputs/s2_sweep_0.e
# Locations are read from the environment (see the README); the script stops when one is not set.
ROOT=${ROOT:?set ROOT to the directory of this package}
STORM_SRC=${STORM_SRC:?set STORM_SRC to the STORM source tree}
STORM_PKGS=${STORM_PKGS:?set STORM_PKGS to the directory of the extra packages STORM needs}
set -uo pipefail
cd ${ROOT}
export PYTHONPATH=${ROOT}/.tools:${STORM_SRC}:${STORM_PKGS}
  echo "=== Alien ==="; STORM_OUT=${ROOT}/ext/outputs STORM_CKPT=${STORM_SRC}/ckpt/Alien-life_done-wm_2L512D8H-100k-seed2/world_model_100000.pth env -u PYTHONHOME python3 -u scripts/v16_storm_sweep.py ALE/Alien-v5 60 8 14 60
echo "=== Amidar ==="; STORM_OUT=${ROOT}/ext/outputs STORM_CKPT=${STORM_SRC}/ckpt/Amidar-life_done-wm_2L512D8H-100k-seed2/world_model_100000.pth env -u PYTHONHOME python3 -u scripts/v16_storm_sweep.py ALE/Amidar-v5 60 8 1 60
echo "=== Assault ==="; STORM_OUT=${ROOT}/ext/outputs STORM_CKPT=${STORM_SRC}/ckpt/Assault-life_done-wm_2L512D8H-100k-seed2/world_model_100000.pth env -u PYTHONHOME python3 -u scripts/v16_storm_sweep.py ALE/Assault-v5 60 8 2 0
echo "=== Asterix ==="; STORM_OUT=${ROOT}/ext/outputs STORM_CKPT=${STORM_SRC}/ckpt/Asterix-life_done-wm_2L512D8H-100k-seed2/world_model_100000.pth env -u PYTHONHOME python3 -u scripts/v16_storm_sweep.py ALE/Asterix-v5 60 8 3 0
echo "=== BankHeist ==="; STORM_OUT=${ROOT}/ext/outputs STORM_CKPT=${STORM_SRC}/ckpt/BankHeist-life_done-wm_2L512D8H-100k-seed2/world_model_100000.pth env -u PYTHONHOME python3 -u scripts/v16_storm_sweep.py ALE/BankHeist-v5 60 8 12 0
echo "=== BattleZone ==="; STORM_OUT=${ROOT}/ext/outputs STORM_CKPT=${STORM_SRC}/ckpt/BattleZone-life_done-wm_2L512D8H-100k-seed2/world_model_100000.pth env -u PYTHONHOME python3 -u scripts/v16_storm_sweep.py ALE/BattleZone-v5 60 8 1 30
echo "=== Boxing ==="; STORM_OUT=${ROOT}/ext/outputs STORM_CKPT=${STORM_SRC}/ckpt/Boxing-life_done-wm_2L512D8H-100k-seed2/world_model_100000.pth env -u PYTHONHOME python3 -u scripts/v16_storm_sweep.py ALE/Boxing-v5 60 8 1 0
echo "=== Breakout ==="; STORM_OUT=${ROOT}/ext/outputs STORM_CKPT=${STORM_SRC}/ckpt/Breakout-life_done-wm_2L512D8H-100k-seed2/world_model_100000.pth env -u PYTHONHOME python3 -u scripts/v16_storm_sweep.py ALE/Breakout-v5 60 8 3 0
echo "=== ChopperCommand ==="; STORM_OUT=${ROOT}/ext/outputs STORM_CKPT=${STORM_SRC}/ckpt/ChopperCommand-life_done-wm_2L512D8H-100k-seed2/world_model_100000.pth env -u PYTHONHOME python3 -u scripts/v16_storm_sweep.py ALE/ChopperCommand-v5 60 8 1 120
echo "=== CrazyClimber ==="; STORM_OUT=${ROOT}/ext/outputs STORM_CKPT=${STORM_SRC}/ckpt/CrazyClimber-life_done-wm_2L512D8H-100k-seed2/world_model_100000.pth env -u PYTHONHOME python3 -u scripts/v16_storm_sweep.py ALE/CrazyClimber-v5 60 8 1 120
echo "=== DemonAttack ==="; STORM_OUT=${ROOT}/ext/outputs STORM_CKPT=${STORM_SRC}/ckpt/DemonAttack-life_done-wm_2L512D8H-100k-seed2/world_model_100000.pth env -u PYTHONHOME python3 -u scripts/v16_storm_sweep.py ALE/DemonAttack-v5 60 8 3 30
echo "=== Freeway ==="; STORM_OUT=${ROOT}/ext/outputs STORM_CKPT=${STORM_SRC}/ckpt/Freeway-life_done-wm_2L512D8H-100k-seed2/world_model_100000.pth env -u PYTHONHOME python3 -u scripts/v16_storm_sweep.py ALE/Freeway-v5 60 8 1 120
echo "=== Frostbite ==="; STORM_OUT=${ROOT}/ext/outputs STORM_CKPT=${STORM_SRC}/ckpt/Frostbite-life_done-wm_2L512D8H-100k-seed2/world_model_100000.pth env -u PYTHONHOME python3 -u scripts/v16_storm_sweep.py ALE/Frostbite-v5 60 8 1 30
echo "finished $(date -Is)"
