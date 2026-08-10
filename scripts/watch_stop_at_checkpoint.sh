#!/usr/bin/env bash
# stop mixkd after checkpoint-N is fully written (model.safetensors stable)
set -euo pipefail
CKPT="${1:?usage: watch_stop_at_checkpoint.sh outputs/phase2_mixkd/checkpoint-12000}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
WEIGHTS="$CKPT/model.safetensors"
LOG="outputs/logs/phase2_mixkd.log"
echo "[$(date -Is)] watching for $WEIGHTS"

while true; do
  if [[ -f "$WEIGHTS" ]]; then
    s1=$(stat -c%s "$WEIGHTS" 2>/dev/null || echo 0)
    sleep 15
    s2=$(stat -c%s "$WEIGHTS" 2>/dev/null || echo 0)
    if [[ "$s1" -gt 1000000 && "$s1" -eq "$s2" ]]; then
      echo "[$(date -Is)] checkpoint stable ($s2 bytes), stopping train"
      pkill -f 'scripts/train_phase2_distill.py --config configs/phase2_distill_mixkd.toml' || true
      pkill -f 'run_phase2_mixkd.sh' || true
      sleep 3
      pgrep -af 'train_phase2_distill.py.*mixkd' && echo "warn: process still alive" || echo "[$(date -Is)] train stopped"
      echo "[$(date -Is)] stopped after $CKPT" >> "$LOG"
      exit 0
    fi
  fi
  if rg -q "Saving model checkpoint to ${CKPT}\$" "$LOG" 2>/dev/null; then
    echo "[$(date -Is)] save started for $CKPT, waiting for weights..."
  fi
  sleep 30
done
