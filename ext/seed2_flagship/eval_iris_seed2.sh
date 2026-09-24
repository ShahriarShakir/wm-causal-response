#!/bin/bash
#PBS -P <PROJECT>
#PBS -q <GPU_QUEUE>
#PBS -l ngpus=1,ncpus=12,mem=64GB,walltime=07:00:00
#PBS -l <SITE_STORAGE_DIRECTIVE>
#PBS -l wd
#PBS -N cS2evalI
#PBS -o <ROOT>/ext/seed2_flagship/eval_iris_seed2.o
#PBS -e <ROOT>/ext/seed2_flagship/eval_iris_seed2.e
# Locations are read from the environment (see the README); the script stops when one is not set.
ROOT=${ROOT:?set ROOT to the directory of this package}
STORM_SRC=${STORM_SRC:?set STORM_SRC to the STORM source tree}
STORM_PKGS=${STORM_PKGS:?set STORM_PKGS to the directory of the extra packages STORM needs}
# Seed-2 IRIS flagship evaluation (arm F, README in ext/iris_seed2): runs only for a game whose training reached epoch 600.
set -uo pipefail
cd ${ROOT}
export PYTHONPATH=${ROOT}/.tools:${STORM_SRC}:${STORM_PKGS}
PY="env -u PYTHONHOME python3 -u"
for G in Alien Pong; do
  RUN=ext/iris_seed2/runs/${G}_seed2
  EP=$($PY -c "import torch; print(int(torch.load('$RUN/checkpoints/epoch.pt', map_location='cpu')))" 2>/dev/null || echo 0)
  if [ "$EP" -lt 600 ]; then echo "SKIP $G: epoch $EP of 600 (training incomplete; resubmit after it lands)"; continue; fi
  echo "=== IRIS seed 2 $G: epoch $EP, checkpoint $RUN/checkpoints/last.pt ==="
  if [ "$G" = Alien ]; then K=14; W=60; else K=2; W=0; fi
  echo "--- chosen intervention k=$K w=$W ---"; IRIS_CKPT=$RUN/checkpoints/last.pt IRIS_OUT=ext/outputs $PY scripts/v5_iris_sweep.py ALE/${G}-v5 60 8 $K $W
  echo "--- fixed intervention k=3 w=0 ---";      IRIS_CKPT=$RUN/checkpoints/last.pt IRIS_OUT=ext/outputs $PY scripts/v5_iris_sweep.py ALE/${G}-v5 60 8 3 0
done
echo "=== job done ==="
