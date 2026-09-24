#!/bin/bash
#PBS -N cIntSearch
#PBS -P <PROJECT>
#PBS -q <CPU_QUEUE>
#PBS -l ncpus=4,mem=16GB,walltime=04:00:00
#PBS -l <SITE_STORAGE_DIRECTIVE>
#PBS -l wd
#PBS -o <ROOT>/outputs/intervention_search.o
#PBS -e <ROOT>/outputs/intervention_search.e
# Locations are read from the environment (see the README); the script stops when one is not set.
ROOT=${ROOT:?set ROOT to the directory of this package}
set -uo pipefail
cd ${ROOT}
export PYTHONPATH=${ROOT}/.tools
env -u PYTHONHOME python3 -u scripts/emulator_intervention_search.py
echo "finished $(date -Is)"
