#!/bin/bash
#PBS -P <PROJECT>
#PBS -q <GPU_QUEUE>
#PBS -l ngpus=1,ncpus=12,mem=64GB,walltime=48:00:00
#PBS -l <SITE_STORAGE_DIRECTIVE>
#PBS -l wd
#PBS -N cD2_Pong_5
#PBS -o <ROOT>/ext/diamond_seed2/train_Pong_5.o
#PBS -e <ROOT>/ext/diamond_seed2/train_Pong_5.e
# Locations are read from the environment (see the README); the script stops when one is not set.
DIAMOND_SRC=${DIAMOND_SRC:?set DIAMOND_SRC to the DIAMOND source tree}
ROOT=${ROOT:?set ROOT to the directory of this package}
set -uo pipefail
cd ${DIAMOND_SRC}
export PYTHONPATH=${ROOT}/.tools:${DIAMOND_SRC}/src
export WANDB_MODE=disabled
RUN=${ROOT}/ext/diamond_seed2/runs/Pong_seed2
mkdir -p $RUN; echo "start $(date -Is) chain 5"
if [ -f $RUN/checkpoints/agent_versions/../state.pt ] && [ -d $RUN/dataset ] && [ 5 -gt 1 ]; then
  env -u PYTHONHOME python3 -u src/main.py common.resume=True env.train.id=PongNoFrameskip-v4 common.seed=2 wandb.mode=disabled hydra.run.dir=$RUN 2>&1 | grep --line-buffered -E "Epoch [0-9]+ /|Error|Traceback|rror" | while IFS= read -r l; do echo "$(date +%T) $l"; done
else
  env -u PYTHONHOME python3 -u src/main.py env.train.id=PongNoFrameskip-v4 common.seed=2 wandb.mode=disabled hydra.run.dir=$RUN 2>&1 | grep --line-buffered -E "Epoch [0-9]+ /|Error|Traceback|rror" | while IFS= read -r l; do echo "$(date +%T) $l"; done
fi
echo "end $(date -Is) chain 5"
