#!/bin/bash
#PBS -P <PROJECT>
#PBS -q <GPU_QUEUE>
#PBS -l ngpus=1,ncpus=12,mem=64GB,walltime=48:00:00
#PBS -l <SITE_STORAGE_DIRECTIVE>
#PBS -l wd
#PBS -N cI2_Pong_1
#PBS -o <ROOT>/ext/iris_seed2/train_Pong_1.o
#PBS -e <ROOT>/ext/iris_seed2/train_Pong_1.e
# Locations are read from the environment (see the README); the script stops when one is not set.
PYTHON_MODULE=${PYTHON_MODULE:?set PYTHON_MODULE to the name of the Python 3.10 environment module on your system}
IRIS_SRC=${IRIS_SRC:?set IRIS_SRC to the IRIS source tree}
ROOT=${ROOT:?set ROOT to the directory of this package}
set -uo pipefail
module load ${PYTHON_MODULE}
cd ${IRIS_SRC}
export WANDB_MODE=disabled
RUN=${ROOT}/ext/iris_seed2/runs/Pong_seed2; mkdir -p $RUN
echo "start $(date -Is) chain 1"
if [ -f $RUN/checkpoints/last.pt ] && [ -f $RUN/checkpoints/epoch.pt ] && [ 1 -gt 1 ]; then
  env -u PYTHONHOME -u PYTHONPATH python3 -u src/main.py common.resume=True hydra.run.dir=$RUN 2>&1 | grep --line-buffered -E "Epoch [0-9]+ /|rror|Traceback" | while IFS= read -r l; do echo "$(date +%T) $l"; done
else
  env -u PYTHONHOME -u PYTHONPATH python3 -u src/main.py env.train.id=PongNoFrameskip-v4 common.device=cuda:0 common.seed=2 wandb.mode=disabled hydra.run.dir=$RUN 2>&1 | grep --line-buffered -E "Epoch [0-9]+ /|rror|Traceback" | while IFS= read -r l; do echo "$(date +%T) $l"; done
fi
echo "end $(date -Is) chain 1"
