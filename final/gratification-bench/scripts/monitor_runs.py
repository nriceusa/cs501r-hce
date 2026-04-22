#!/usr/bin/env python3
"""monitor_runs.py — watches evaluation result files and alerts on stalls.

A "stall" = a file has been non-empty but hasn't grown in STALL_MINUTES.
Prints a timestamped status line every CHECK_INTERVAL seconds.

Usage:
    python scripts/monitor_runs.py
"""

import time
from datetime import datetime
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results"
STALL_MINUTES = 20
CHECK_INTERVAL = 120  # seconds

last_sizes: dict[str, int] = {}
last_changed: dict[str, float] = {}

print("=== Gratification Bench run monitor ===")
print(f"Watching: {RESULTS_DIR}/*.jsonl")
print(f"Stall threshold: {STALL_MINUTES} min | Check interval: {CHECK_INTERVAL}s\n")

while True:
    now = time.time()
    ts = datetime.now().strftime("%H:%M:%S")
    files = sorted(RESULTS_DIR.glob("*.jsonl"))

    if not files:
        print(f"[{ts}] No .jsonl files in results/ yet — waiting…")
    else:
        for f in files:
            name = f.name
            size = sum(1 for line in f.open() if line.strip()) if f.stat().st_size > 0 else 0
            prev_size = last_sizes.get(name, 0)
            prev_time = last_changed.get(name, now)

            if size > prev_size:
                delta = size - prev_size
                last_sizes[name] = size
                last_changed[name] = now
                print(f"[{ts}] ✓ {name}: {size} cases (+{delta})")
            else:
                stale_min = int((now - prev_time) / 60)
                if prev_size == 0 and size == 0:
                    print(f"[{ts}]   {name}: 0 cases (not started yet)")
                elif stale_min >= STALL_MINUTES:
                    print(f"[{ts}] ⚠️  STALLED: {name} — {size} cases, no progress for {stale_min}m")
                else:
                    print(f"[{ts}]   {name}: {size} cases (unchanged, {stale_min}m since last write)")

    print("---")
    time.sleep(CHECK_INTERVAL)
