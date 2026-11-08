#!/bin/bash
#PBS -P <PROJECT>
#PBS -q <GPU_QUEUE>
#PBS -l ngpus=1,ncpus=12,mem=32GB,walltime=00:50:00
#PBS -l <SITE_STORAGE_DIRECTIVE>
#PBS -l wd
#PBS -N vG_d_Breakout
#PBS -o <ROOT>/ext/visited/pbs_gpu/vG_d_Breakout.o
#PBS -e <ROOT>/ext/visited/pbs_gpu/vG_d_Breakout.e
# Locations are read from the environment (see the README); the script stops when one is not set.
ROOT=${ROOT:?set ROOT to the directory of this package}
STORM_SRC=${STORM_SRC:?set STORM_SRC to the STORM source tree}
STORM_PKGS=${STORM_PKGS:?set STORM_PKGS to the directory of the extra packages STORM needs}
set -uo pipefail
cd ${ROOT}
export PYTHONPATH=${ROOT}/.tools:${STORM_SRC}:${STORM_PKGS}
export VISITED_OUTDIR=outputs_gpu
echo "=== diamond ALE/Breakout-v5 ==="; env -u PYTHONHOME python3 -u ext/visited/v24_visited_states.py diamond ALE/Breakout-v5
echo "=== job done ==="
