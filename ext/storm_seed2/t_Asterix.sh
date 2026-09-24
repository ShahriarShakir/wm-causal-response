#!/bin/bash
#PBS -N cS2_Asterix
#PBS -P <PROJECT>
#PBS -q <GPU_QUEUE>
#PBS -l ngpus=1,ncpus=12,mem=64GB,walltime=09:00:00
#PBS -l <SITE_STORAGE_DIRECTIVE>
#PBS -l wd
#PBS -o <ROOT>/ext/storm_seed2/t_Asterix.o
#PBS -e <ROOT>/ext/storm_seed2/t_Asterix.e
# Locations are read from the environment (see the README); the script stops when one is not set.
STORM_SRC=${STORM_SRC:?set STORM_SRC to the STORM source tree}
STORM_PKGS=${STORM_PKGS:?set STORM_PKGS to the directory of the extra packages STORM needs}
ROOT=${ROOT:?set ROOT to the directory of this package}
set -uo pipefail
cd ${STORM_SRC}
export PYTHONPATH=${STORM_PKGS}:${ROOT}/.tools
echo "started $(date -Is)"
env -u PYTHONHOME python3 -u train.py -n "Asterix-life_done-wm_2L512D8H-100k-seed2" -seed 2 -config_path config_files/STORM.yaml -env_name "ALE/Asterix-v5" -trajectory_path "D_TRAJ/Asterix.pkl"
echo "finished $(date -Is)"
