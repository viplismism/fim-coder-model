#!/usr/bin/env bash
# Hardware log every 10 s, synced to disk after each line so it survives a hard reset:
# GPU temp/power/SM clock/util/throttle flags, max ACPI thermal zone, fan speeds, NVMe temp, memory available.
OUT=$1
echo -e "time\tgpu_c\tpower_w\tsm_mhz\tutil\tthrottle\thw_thermal\tsw_thermal\tacpi_max_c\tfans_rpm\tnvme_c\tmem_avail_gb" >> "$OUT"
while true; do
  G=$(nvidia-smi --query-gpu=temperature.gpu,power.draw,clocks.sm,utilization.gpu,clocks_throttle_reasons.active,clocks_throttle_reasons.hw_thermal_slowdown,clocks_throttle_reasons.sw_thermal_slowdown --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' | tr ',' '\t')
  A=$(cat /sys/class/thermal/thermal_zone*/temp | sort -n | tail -1 | awk '{printf "%.1f", $1/1000}')
  F=$(cat /sys/class/hwmon/hwmon*/fan*_input 2>/dev/null | tr '\n' '/')
  N=$(cat /sys/class/hwmon/hwmon3/temp1_input 2>/dev/null | awk '{printf "%.1f", $1/1000}')
  M=$(awk '/MemAvailable/ {printf "%.1f", $2/1048576}' /proc/meminfo)
  echo -e "$(date '+%F %T')\t$G\t$A\t$F\t$N\t$M" >> "$OUT"; sync "$OUT"
  sleep 10
done
