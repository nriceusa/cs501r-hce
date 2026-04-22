#!/usr/bin/env bash
# monitor_runs.sh — watches evaluation result files and alerts if any stall.
# Usage: bash scripts/monitor_runs.sh
# A "stall" = file hasn't grown in STALL_MINUTES minutes.

RESULTS_DIR="$(dirname "$0")/../results"
STALL_MINUTES=20
CHECK_INTERVAL=120  # seconds between checks

declare -A last_sizes
declare -A last_changed

echo "=== Gratification Bench run monitor ==="
echo "Watching: $RESULTS_DIR/*.jsonl"
echo "Stall threshold: ${STALL_MINUTES} min | Check interval: ${CHECK_INTERVAL}s"
echo ""

while true; do
  now=$(date +%s)
  timestamp=$(date "+%H:%M:%S")

  for f in "$RESULTS_DIR"/*.jsonl; do
    [ -f "$f" ] || continue
    name=$(basename "$f")
    size=$(wc -l < "$f" 2>/dev/null || echo 0)

    prev_size="${last_sizes[$name]:-0}"
    prev_time="${last_changed[$name]:-$now}"

    if [ "$size" -gt "$prev_size" ]; then
      last_sizes[$name]=$size
      last_changed[$name]=$now
      echo "[$timestamp] $name: $size cases (+$((size - prev_size)))"
    else
      # No growth — check if stalled
      stale_sec=$(( now - prev_time ))
      stale_min=$(( stale_sec / 60 ))
      if [ "$stale_min" -ge "$STALL_MINUTES" ] && [ "$size" -gt 0 ]; then
        echo "[$timestamp] ⚠️  STALLED: $name — $size cases, no progress for ${stale_min}m"
      else
        echo "[$timestamp] $name: $size cases (unchanged, ${stale_min}m)"
      fi
    fi
  done
  echo "---"
  sleep $CHECK_INTERVAL
done
