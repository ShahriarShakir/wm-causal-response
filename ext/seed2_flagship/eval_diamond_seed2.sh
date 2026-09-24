#!/bin/bash
#PBS -P <PROJECT>
#PBS -q <GPU_QUEUE>
#PBS -l ngpus=1,ncpus=12,mem=64GB,walltime=02:00:00,<LOCAL_DISK>=2GB
#PBS -l <SITE_STORAGE_DIRECTIVE>
#PBS -l wd
#PBS -N cS2evalD
#PBS -o <ROOT>/ext/seed2_flagship/eval_diamond_seed2.o
#PBS -e <ROOT>/ext/seed2_flagship/eval_diamond_seed2.e
# Locations are read from the environment (see the README); the script stops when one is not set.
ROOT=${ROOT:?set ROOT to the directory of this package}
STORM_SRC=${STORM_SRC:?set STORM_SRC to the STORM source tree}
STORM_PKGS=${STORM_PKGS:?set STORM_PKGS to the directory of the extra packages STORM needs}
# Seed-2 DIAMOND flagship evaluation (arm E, README in ext/diamond_seed2): runs only for a game whose training reached epoch 1000 (950 collection epochs + 50 final, as the trainer computes from 100k steps).
set -uo pipefail
cd ${ROOT}
export PYTHONPATH=${ROOT}/.tools:${STORM_SRC}:${STORM_PKGS}
PY="env -u PYTHONHOME python3 -u"
for G in Alien Pong; do
  RUN=ext/diamond_seed2/runs/${G}_seed2
  LAST=$(ls $RUN/checkpoints/agent_versions/agent_epoch_*.pt 2>/dev/null | sort | tail -1)
  EP=$(basename "$LAST" .pt | sed 's/[^0-9]//g' | sed 's/^0*//'); EP=${EP:-0}
  if [ "$EP" -lt 1000 ]; then echo "SKIP $G: last checkpoint epoch $EP of 1000 (training incomplete; resubmit after it lands)"; continue; fi
  echo "=== DIAMOND seed 2 $G: epoch $EP, checkpoint $LAST ==="
  if [ "$G" = Alien ]; then K=14; W=60; else K=2; W=0; fi
  echo "--- chosen intervention k=$K w=$W ---"; DIAMOND_CKPT=$LAST DIAMOND_OUT=ext/outputs $PY scripts/v4_video_sweep.py ALE/${G}-v5 60 8 $K $W
  echo "--- fixed intervention k=3 w=0 ---";      DIAMOND_CKPT=$LAST DIAMOND_OUT=ext/outputs $PY scripts/v4_video_sweep.py ALE/${G}-v5 60 8 3 0
done
echo "=== job done ==="
