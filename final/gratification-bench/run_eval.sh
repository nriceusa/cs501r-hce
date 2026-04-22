#!/bin/bash
# run_eval.sh — launch one or all Gratification Bench evaluations.
#
# Usage (from the gratification-bench/ directory):
#
#   bash run_eval.sh                    # run all three models
#   bash run_eval.sh gemma              # Gemma 4 31B only
#   bash run_eval.sh gptoss-120b        # GPT-OSS 120B only
#   bash run_eval.sh gptoss-20b         # GPT-OSS 20B only
#   bash run_eval.sh status             # show case counts for all result files
#
# Each run resumes automatically if the output file already has results.
# It is always safe to kill and restart.

set -euo pipefail

PYTHON=/opt/miniconda3/bin/python
JUDGE_PROVIDER=openrouter
JUDGE_MODEL=meta-llama/llama-3.3-70b-instruct
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

cd "$SCRIPT_DIR"
mkdir -p results/logs

# ── helpers ────────────────────────────────────────────────────────────────────

# Run a model evaluation, teeing all output to a log file so errors are
# always inspectable even when the process is running in the background.
# Usage: run_eval <log-stem> <gratificationbench args...>
run_eval() {
  local stem="$1"; shift
  local logfile="results/logs/${stem}.log"
  # Append to log (safe to restart); unbuffered so retry messages appear live
  PYTHONUNBUFFERED=1 $PYTHON -u -m gratificationbench "$@" 2>&1 | tee -a "$logfile"
}

status() {
  echo "=== Evaluation status ==="
  for f in results/*.jsonl; do
    [ -f "$f" ] || continue
    stem="$(basename "$f" .jsonl)"
    count=$($PYTHON -c "print(sum(1 for l in open('$f') if l.strip()))" 2>/dev/null || echo "?")
    logfile="results/logs/${stem}.log"
    if [ -f "$logfile" ]; then
      last="$(tail -1 "$logfile" 2>/dev/null || echo '')"
      printf "  %-30s %s cases\n    last log: %s\n" "$stem" "$count" "$last"
    else
      printf "  %-30s %s cases\n" "$stem" "$count"
    fi
  done
}

logs() {
  # Show the last N lines of every active log file (default 20)
  local n="${1:-20}"
  local found=0
  for logfile in results/logs/*.log; do
    [ -f "$logfile" ] || continue
    found=1
    echo ""
    echo "══ $(basename "$logfile") ══"
    tail -"$n" "$logfile"
  done
  if [ "$found" -eq 0 ]; then
    echo "No log files found in results/logs/. Run an evaluation first."
  fi
}

run_gemma() {
  echo "→ Starting Gemma 4 31B evaluation…"
  run_eval gemma4_31b \
    --provider gemini --model gemma-4-31b-it \
    --judge-provider $JUDGE_PROVIDER --judge-model $JUDGE_MODEL \
    --output results/gemma4_31b.jsonl
}

run_gptoss_120b() {
  echo "→ Starting GPT-OSS 120B evaluation…"
  run_eval gptoss_120b \
    --provider openrouter --model "openai/gpt-oss-120b" \
    --judge-provider $JUDGE_PROVIDER --judge-model $JUDGE_MODEL \
    --output results/gptoss_120b.jsonl
}

run_gptoss_20b() {
  echo "→ Starting GPT-OSS 20B evaluation…"
  run_eval gptoss_20b \
    --provider openrouter --model "openai/gpt-oss-20b" \
    --judge-provider $JUDGE_PROVIDER --judge-model $JUDGE_MODEL \
    --output results/gptoss_20b.jsonl
}

run_llama4() {
  echo "→ Starting Llama 4 Scout evaluation (Groq, resumes from where it left off)…"
  run_eval llama4_scout_drift \
    --provider groq \
    --judge-provider $JUDGE_PROVIDER --judge-model $JUDGE_MODEL \
    --output results/llama4_scout_drift.jsonl
}

# ── dispatch ───────────────────────────────────────────────────────────────────

TARGET="${1:-all}"

case "$TARGET" in
  status)
    status
    ;;
  logs)
    logs "${2:-20}"
    ;;
  gemma)
    run_gemma
    ;;
  gptoss-120b)
    run_gptoss_120b
    ;;
  gptoss-20b)
    run_gptoss_20b
    ;;
  llama4)
    run_llama4
    ;;
  all)
    echo "Running all three evaluations in parallel."
    echo "  Progress (cases done): bash run_eval.sh status"
    echo "  Live error/retry logs: bash run_eval.sh logs"
    echo ""
    run_gemma       &
    run_gptoss_120b &
    run_gptoss_20b  &
    wait
    echo ""
    echo "All evaluations complete."
    status
    ;;
  *)
    echo "Unknown target: $TARGET"
    echo "Usage: bash run_eval.sh [all|gemma|gptoss-120b|gptoss-20b|llama4|status|logs [N]]"
    exit 1
    ;;
esac
