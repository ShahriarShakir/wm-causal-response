#!/bin/bash
#PBS -N cFS2_B2
#PBS -P <PROJECT>
#PBS -q <GPU_QUEUE>
#PBS -l ngpus=1,ncpus=12,mem=64GB,walltime=10:00:00
#PBS -l <SITE_STORAGE_DIRECTIVE>
#PBS -l wd
#PBS -o <ROOT>/outputs/fsrank2_iris_b.o
#PBS -e <ROOT>/outputs/fsrank2_iris_b.e
# Locations are read from the environment (see the README); the script stops when one is not set.
ROOT=${ROOT:?set ROOT to the directory of this package}
STORM_SRC=${STORM_SRC:?set STORM_SRC to the STORM source tree}
STORM_PKGS=${STORM_PKGS:?set STORM_PKGS to the directory of the extra packages STORM needs}
set -uo pipefail
cd ${ROOT}
export PYTHONPATH=${ROOT}/.tools:${STORM_SRC}:${STORM_PKGS}
  for g in Alien Amidar Assault Asterix Boxing Breakout ChopperCommand CrazyClimber DemonAttack Freeway Frostbite MsPacman Qbert Seaquest UpNDown; do if [ -f outputs/fsrank2_iris_${g}_dense.json ]; then echo "=== $g done ==="; continue; fi; echo "=== $g ==="; env -u PYTHONHOME python3 -u scripts/v14_fixed_scorer_ranking.py iris ALE/${g}-v5 --densedata; done
echo "finished $(date -Is)"
