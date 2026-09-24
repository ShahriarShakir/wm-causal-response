#!/bin/bash
#PBS -N cSynth
#PBS -P <PROJECT>
#PBS -q <CPU_QUEUE>
#PBS -l ncpus=4,mem=16GB,walltime=01:30:00
#PBS -l <SITE_STORAGE_DIRECTIVE>
#PBS -l wd
#PBS -o <ROOT>/outputs/synth.o
#PBS -e <ROOT>/outputs/synth.e
# Locations are read from the environment (see the README); the script stops when one is not set.
ROOT=${ROOT:?set ROOT to the directory of this package}
set -uo pipefail
cd ${ROOT}
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
export PYTHONPATH=${ROOT}/.tools
export MKL_NUM_THREADS=4 OMP_NUM_THREADS=4
echo "started $(date -Is)"
env -u PYTHONHOME python3 -u     scripts/synthetic_failures.py
echo "finished $(date -Is)"
