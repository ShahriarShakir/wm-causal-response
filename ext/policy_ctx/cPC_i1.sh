#!/bin/bash
#PBS -P <PROJECT>
#PBS -q <GPU_QUEUE>
#PBS -l ngpus=1,ncpus=12,mem=64GB,walltime=05:00:00
#PBS -l <SITE_STORAGE_DIRECTIVE>
#PBS -l wd
#PBS -N cPC_i1
#PBS -o <ROOT>/ext/policy_ctx/cPC_i1.o
#PBS -e <ROOT>/ext/policy_ctx/cPC_i1.e
# Locations are read from the environment (see the README); the script stops when one is not set.
ROOT=${ROOT:?set ROOT to the directory of this package}
set -uo pipefail
cd ${ROOT}
export PYTHONPATH=${ROOT}/.tools
echo "=== iris ALE/Freeway-v5 30 8 1 120 0.0 ==="; env -u PYTHONHOME python3 -u scripts/v22_policy_context.py iris ALE/Freeway-v5 30 8 1 120 0.0
echo "=== iris ALE/Breakout-v5 30 8 3 0 0.0 ==="; env -u PYTHONHOME python3 -u scripts/v22_policy_context.py iris ALE/Breakout-v5 30 8 3 0 0.0
echo "=== iris ALE/Boxing-v5 30 8 1 0 0.0 ==="; env -u PYTHONHOME python3 -u scripts/v22_policy_context.py iris ALE/Boxing-v5 30 8 1 0 0.0
