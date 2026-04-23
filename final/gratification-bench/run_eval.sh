#!/bin/bash
# run_eval.sh — launch one or all Gratification Bench evaluations.
#
# All models are accessed via OpenRouter. Add OPENROUTER_API_KEY to .env.
#
# Usage (from the gratification-bench/ directory):
#
#   bash run_eval.sh                    # run all four models in parallel
#   bash run_eval.sh gpt5               # GPT-5.3 only
#   bash run_eval.sh gemini3            # Gemini 3 Flash only
#   bash run_eval.sh claude             # Claude Sonnet 4.6 only
#   bash run_eval.sh grok               # Grok 4.1 Fast only
#   bash run_eval.sh status             # show case counts for all result files
#   bash run_eval.sh logs [N]           # show last N lines of each log (default 20)
#
# Each run resumes automatically if the output file already has results.
# It is always safe to kill and restart.

set -euo pipefail

PYTHON=/opt/miniconda3/bin/python
PROVIDER=openrouter
JUDGE_PROVIDER=openrouter
JUDGE_MODEL=google/gemma-4-26b-a4b-it
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

run_gpt5() {
  echo "→ Starting GPT-5.3 evaluation…"
  run_eval gpt5 \
    --provider $PROVIDER --model "openai/gpt-5.3-chat" \
    --judge-provider $JUDGE_PROVIDER --judge-model $JUDGE_MODEL \
    --output results/gpt5.jsonl
}

run_gemini3() {
  echo "→ Starting Gemini 3 Flash evaluation…"
  run_eval gemini3_flash \
    --provider $PROVIDER --model "google/gemini-3-flash-preview" \
    --judge-provider $JUDGE_PROVIDER --judge-model $JUDGE_MODEL \
    --output results/gemini3_flash.jsonl
}

run_claude() {
  echo "→ Starting Claude Sonnet 4.6 evaluation…"
  run_eval claude_sonnet_46 \
    --provider $PROVIDER --model "anthropic/claude-sonnet-4.6" \
    --judge-provider $JUDGE_PROVIDER --judge-model $JUDGE_MODEL \
    --output results/claude_sonnet_46.jsonl
}

run_grok() {
  echo "→ Starting Grok 4.1 Fast evaluation…"
  run_eval grok_41_fast \
    --provider $PROVIDER --model "x-ai/grok-4.1-fast" \
    --judge-provider $JUDGE_PROVIDER --judge-model $JUDGE_MODEL \
    --output results/grok_41_fast.jsonl
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
  gpt5)
    run_gpt5
    ;;
  gemini3)
    run_gemini3
    ;;
  claude)
    run_claude
    ;;
  grok)
    run_grok
    ;;
  all)
    echo "Running all four evaluations in parallel."
    echo "  Progress (cases done): bash run_eval.sh status"
    echo "  Live error/retry logs: bash run_eval.sh logs"
    echo ""
    run_gpt5    &
    run_gemini3 &
    run_claude  &
    run_grok    &
    wait
    echo ""
    echo "All evaluations complete."
    status
    ;;
  *)
    echo "Unknown target: $TARGET"
    echo "Usage: bash run_eval.sh [all|gpt5|gemini3|claude|grok|status|logs [N]]"
    exit 1
    ;;
esac
