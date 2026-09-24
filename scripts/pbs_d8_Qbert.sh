#!/bin/bash
#PBS -N ccd8_Qbert
#PBS -P <PROJECT>
#PBS -q <GPU_QUEUE>
#PBS -l ngpus=1,ncpus=12,mem=64GB,walltime=04:00:00
#PBS -l <SITE_STORAGE_DIRECTIVE>
#PBS -l wd
#PBS -o <ROOT>/outputs/d8_Qbert.o
#PBS -e <ROOT>/outputs/d8_Qbert.e
# Locations are read from the environment (see the README); the script stops when one is not set.
ROOT=${ROOT:?set ROOT to the directory of this package}
set -uo pipefail
cd ${ROOT}
export PYTHONPATH=${ROOT}/.tools
echo "started $(date -Is)"; nvidia-smi -L 2>/dev/null | head -1
env -u PYTHONHOME python3 -u scripts/v4_video_sweep.py ALE/Qbert-v5 60 8
echo "finished $(date -Is)"
