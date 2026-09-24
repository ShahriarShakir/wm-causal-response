#!/bin/bash
#PBS -P <PROJECT>
#PBS -q <GPU_QUEUE>
#PBS -l ngpus=1,ncpus=12,mem=64GB,walltime=01:30:00
#PBS -l <SITE_STORAGE_DIRECTIVE>
#PBS -l wd
#PBS -N cD2smoke
#PBS -o <ROOT>/ext/diamond_seed2/smoke_Pong.o
#PBS -e <ROOT>/ext/diamond_seed2/smoke_Pong.e
# Locations are read from the environment (see the README); the script stops when one is not set.
DIAMOND_SRC=${DIAMOND_SRC:?set DIAMOND_SRC to the DIAMOND source tree}
ROOT=${ROOT:?set ROOT to the directory of this package}
set -uo pipefail
cd ${DIAMOND_SRC}
export PYTHONPATH=${ROOT}/.tools:${DIAMOND_SRC}/src
export WANDB_MODE=disabled
mkdir -p ${ROOT}/ext/diamond_seed2/runs/Pong_smoke
echo "start $(date -Is)"
env -u PYTHONHOME python3 -u src/main.py env.train.id=PongNoFrameskip-v4 common.seed=2 wandb.mode=disabled hydra.run.dir=${ROOT}/ext/diamond_seed2/runs/Pong_smoke 2>&1 | while IFS= read -r l; do echo "$(date +%T) $l"; done
echo "end $(date -Is)"
