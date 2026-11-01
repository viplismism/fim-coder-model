#!/usr/bin/env bash
# Sample system memory every 2 s while training processes run; kill them if available memory drops below a floor.
# Usage: checks/memwatch.sh <out.tsv> <floor_gb>
OUT=$1; FLOOR_KB=$(( ${2:-12} * 1024 * 1024 ))
echo -e "time\tmem_available_gb\tmem_used_gb\tswap_used_gb\ttrain_procs" > "$OUT"
while true; do
  P=$(pgrep -f "fimexp.train --data_version" | tr '\n' ' ')
  A=$(awk '/MemAvailable/ {print $2}' /proc/meminfo); T=$(awk '/MemTotal/ {print $2}' /proc/meminfo)
  S=$(awk '/SwapTotal/ {t=$2} /SwapFree/ {f=$2} END {print t-f}' /proc/meminfo)
  echo -e "$(date +%T)\t$(awk "BEGIN{printf \"%.1f\", $A/1048576}")\t$(awk "BEGIN{printf \"%.1f\", ($T-$A)/1048576}")\t$(awk "BEGIN{printf \"%.1f\", $S/1048576}")\t$P" >> "$OUT"
  if [ -n "$P" ] && [ "$A" -lt "$FLOOR_KB" ]; then echo "MEMORY PRESSURE: killing $P" >> "$OUT"; kill $P; fi
  [ -f "$OUT.stop" ] && break
  sleep 2
done
