#!/bin/bash
#PBS -P <PROJECT>
#PBS -q <CPU_QUEUE>
#PBS -l ncpus=12,mem=48GB,walltime=01:00:00
#PBS -l <SITE_STORAGE_DIRECTIVE>
#PBS -l wd
#PBS -N gR_d_Assaul
#PBS -o <ROOT>/ext/goal/pbs/gR_d_Assaul.o
#PBS -e <ROOT>/ext/goal/pbs/gR_d_Assaul.e
# Locations are read from the environment (see the README); the script stops when one is not set.
ROOT=${ROOT:?set ROOT to the directory of this package}
STORM_SRC=${STORM_SRC:?set STORM_SRC to the STORM source tree}
STORM_PKGS=${STORM_PKGS:?set STORM_PKGS to the directory of the extra packages STORM needs}
set -uo pipefail
cd ${ROOT}
export PYTHONPATH=${ROOT}/.tools:${STORM_SRC}:${STORM_PKGS}
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12 MKL_NUM_THREADS=12
echo "=== diamond ALE/Assault-v5 ==="; env -u PYTHONHOME python3 -u ext/goal/v25_goal_reaching.py diamond ALE/Assault-v5
echo "=== diamond ALE/Asterix-v5 ==="; env -u PYTHONHOME python3 -u ext/goal/v25_goal_reaching.py diamond ALE/Asterix-v5
echo "=== job done ==="
