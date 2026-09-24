#!/bin/bash
#PBS -N cSus_d
#PBS -P <PROJECT>
#PBS -q <GPU_QUEUE>
#PBS -l ngpus=1,ncpus=12,mem=64GB,walltime=08:00:00
#PBS -l <SITE_STORAGE_DIRECTIVE>
#PBS -l wd
#PBS -o <ROOT>/outputs/sus_d.o
#PBS -e <ROOT>/outputs/sus_d.e
# Locations are read from the environment (see the README); the script stops when one is not set.
ROOT=${ROOT:?set ROOT to the directory of this package}
set -uo pipefail
cd ${ROOT}
export PYTHONPATH=${ROOT}/.tools
  echo "=== Alien sustained ==="; HORIZON=48 env -u PYTHONHOME python3 -u scripts/v4_video_sweep.py ALE/Alien-v5 60 40 14 60
echo "=== Pong sustained ==="; HORIZON=48 env -u PYTHONHOME python3 -u scripts/v4_video_sweep.py ALE/Pong-v5 60 40 2 0
echo "=== BankHeist sustained ==="; HORIZON=48 env -u PYTHONHOME python3 -u scripts/v4_video_sweep.py ALE/BankHeist-v5 60 40 12 0
echo "=== Boxing sustained ==="; HORIZON=48 env -u PYTHONHOME python3 -u scripts/v4_video_sweep.py ALE/Boxing-v5 60 40 1 0
echo "=== Breakout sustained ==="; HORIZON=48 env -u PYTHONHOME python3 -u scripts/v4_video_sweep.py ALE/Breakout-v5 60 40 3 0
echo "=== Freeway sustained ==="; HORIZON=48 env -u PYTHONHOME python3 -u scripts/v4_video_sweep.py ALE/Freeway-v5 60 40 1 120
echo "=== Gopher sustained ==="; HORIZON=48 env -u PYTHONHOME python3 -u scripts/v4_video_sweep.py ALE/Gopher-v5 60 40 1 120
echo "=== Seaquest sustained ==="; HORIZON=48 env -u PYTHONHOME python3 -u scripts/v4_video_sweep.py ALE/Seaquest-v5 60 40 1 60
echo "=== MsPacman sustained ==="; HORIZON=48 env -u PYTHONHOME python3 -u scripts/v4_video_sweep.py ALE/MsPacman-v5 60 40 1 60
echo "=== Frostbite sustained ==="; HORIZON=48 env -u PYTHONHOME python3 -u scripts/v4_video_sweep.py ALE/Frostbite-v5 60 40 1 30
echo "finished $(date -Is)"
