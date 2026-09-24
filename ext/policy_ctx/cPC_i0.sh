#!/bin/bash
#PBS -P <PROJECT>
#PBS -q <GPU_QUEUE>
#PBS -l ngpus=1,ncpus=12,mem=64GB,walltime=05:00:00
#PBS -l <SITE_STORAGE_DIRECTIVE>
#PBS -l wd
#PBS -N cPC_i0
#PBS -o <ROOT>/ext/policy_ctx/cPC_i0.o
#PBS -e <ROOT>/ext/policy_ctx/cPC_i0.e
# Locations are read from the environment (see the README); the script stops when one is not set.
ROOT=${ROOT:?set ROOT to the directory of this package}
set -uo pipefail
cd ${ROOT}
export PYTHONPATH=${ROOT}/.tools
echo "=== iris ALE/Alien-v5 30 8 14 60 0.0 ==="; env -u PYTHONHOME python3 -u scripts/v22_policy_context.py iris ALE/Alien-v5 30 8 14 60 0.0
echo "=== iris ALE/Pong-v5 30 8 2 0 0.0 ==="; env -u PYTHONHOME python3 -u scripts/v22_policy_context.py iris ALE/Pong-v5 30 8 2 0 0.0
echo "=== iris ALE/BankHeist-v5 30 8 12 0 0.0 ==="; env -u PYTHONHOME python3 -u scripts/v22_policy_context.py iris ALE/BankHeist-v5 30 8 12 0 0.0
echo "=== iris ALE/Alien-v5 30 8 14 60 0.1 ==="; env -u PYTHONHOME python3 -u scripts/v22_policy_context.py iris ALE/Alien-v5 30 8 14 60 0.1
echo "=== iris ALE/Pong-v5 30 8 2 0 0.1 ==="; env -u PYTHONHOME python3 -u scripts/v22_policy_context.py iris ALE/Pong-v5 30 8 2 0 0.1
