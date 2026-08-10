#!/usr/bin/env bash
# retrain italian colbert with document_length=256 (keeps outputs/final intact)
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p outputs/logs

echo "[$(date -Is)] phase 1 contrastive doc256 start"
uv run python scripts/train_phase1_contrastive.py --config configs/phase1_contrastive_doc256.toml \
  2>&1 | tee outputs/logs/phase1_doc256.log
echo "[$(date -Is)] phase 1 doc256 done"

echo "[$(date -Is)] phase 2 distillation doc256 start"
uv run python scripts/train_phase2_distill.py --config configs/phase2_distill_doc256.toml \
  2>&1 | tee outputs/logs/phase2_doc256.log
echo "[$(date -Is)] phase 2 doc256 done"

echo "[$(date -Is)] copy final_doc256"
mkdir -p outputs/final_doc256
cp -a outputs/phase2_doc256/final/. outputs/final_doc256/

echo "[$(date -Is)] evaluate triplets"
uv run python scripts/evaluate.py --model outputs/final_doc256 --config configs/eval.toml \
  2>&1 | tee outputs/logs/eval_doc256.log || true

echo "[$(date -Is)] bench only ItColBERT doc256"
unset HF_HUB_ENABLE_HF_TRANSFER || true
export HF_XET_HIGH_PERFORMANCE=1
uv run python scripts/run_benchmark.py \
  --benchmarks mldr-it mmarco-it \
  --mmarco-max-corpus-docs 100000 \
  --only "ItColBERT (doc256)" \
  --top-k 100 \
  --output-dir outputs/benchmark \
  2>&1 | tee outputs/logs/bench_doc256.log

echo "[$(date -Is)] all done doc256"
