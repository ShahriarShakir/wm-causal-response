#!/bin/bash
#PBS -P <PROJECT>
#PBS -q <GPU_QUEUE>
#PBS -l ngpus=1,ncpus=12,mem=64GB,walltime=05:00:00
#PBS -l <SITE_STORAGE_DIRECTIVE>
#PBS -l wd
#PBS -N cPC_d_Alien
#PBS -o <ROOT>/ext/policy_ctx/cPC_d_Alien.o
#PBS -e <ROOT>/ext/policy_ctx/cPC_d_Alien.e
# Locations are read from the environment (see the README); the script stops when one is not set.
ROOT=${ROOT:?set ROOT to the directory of this package}
set -uo pipefail
cd ${ROOT}
export PYTHONPATH=${ROOT}/.tools
echo "=== diamond ALE/Alien-v5 30 8 14 60 0.0 ==="; env -u PYTHONHOME python3 -u scripts/v22_policy_context.py diamond ALE/Alien-v5 30 8 14 60 0.0
echo "=== diamond ALE/Alien-v5 30 8 14 60 0.1 ==="; env -u PYTHONHOME python3 -u scripts/v22_policy_context.py diamond ALE/Alien-v5 30 8 14 60 0.1
