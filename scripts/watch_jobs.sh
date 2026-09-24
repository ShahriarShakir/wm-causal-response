#!/bin/bash
# Locations are read from the environment (see the README); the script stops when one is not set.
ROOT=${ROOT:?set ROOT to the directory of this package}
# robust watch: reports each landed output and exits 0 when none of the tracked jobs remain
cd ${ROOT}
seen=""
for i in $(seq 1 120); do
  running=$(qstat -u ${USER} 2>/dev/null | grep -E "cRank|cCtrl|cHid|cSynth" | awk '{print $4":"$10}' | tr '\n' ' ')
  for f in $(ls outputs/irishidden_*.json outputs/rssm_controls.json outputs/ranking_*_*.json outputs/irishiddenctl_*.json 2>/dev/null | grep -v DRY); do
    case " $seen " in *" $f "*) ;; *) echo "$(date +%H:%M) LANDED $f"; seen="$seen $f";; esac
  done
  echo "$(date +%H:%M) running: ${running:-none}"
  [ -z "$running" ] && { echo "all tracked jobs finished"; exit 0; }
  sleep 300
done
echo "watch timed out after 10h"; exit 0
