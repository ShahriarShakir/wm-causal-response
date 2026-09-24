#!/bin/bash
#PBS -N cAbl_UpNDown
#PBS -P <PROJECT>
#PBS -q <GPU_QUEUE>
#PBS -l ngpus=1,ncpus=12,mem=64GB,walltime=06:00:00
#PBS -l <SITE_STORAGE_DIRECTIVE>
#PBS -l wd
#PBS -o <ROOT>/outputs/abl_UpNDown.o
#PBS -e <ROOT>/outputs/abl_UpNDown.e
# Locations are read from the environment (see the README); the script stops when one is not set.
ROOT=${ROOT:?set ROOT to the directory of this package}
set -uo pipefail
cd ${ROOT}
export PYTHONPATH=${ROOT}/.tools
echo "started $(date -Is)"
for D in 1 4 8 16; do
  echo "=== UpNDown duration $D ==="
  env -u PYTHONHOME python3 -u \
      scripts/v4_video_sweep.py ALE/UpNDown-v5 60 $D
done
echo "finished $(date -Is)"
