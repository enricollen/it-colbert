#!/usr/bin/env bash
# run full two-phase italian colbert training
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p outputs/logs

echo "[$(date -Is)] phase 1 contrastive start"
uv run python scripts/train_phase1_contrastive.py --config configs/phase1_contrastive.toml \
  2>&1 | tee outputs/logs/phase1.log
echo "[$(date -Is)] phase 1 done"

echo "[$(date -Is)] phase 2 distillation start"
uv run python scripts/train_phase2_distill.py --config configs/phase2_distill.toml \
  2>&1 | tee outputs/logs/phase2.log
echo "[$(date -Is)] phase 2 done"

echo "[$(date -Is)] evaluate"
uv run python scripts/evaluate.py --model outputs/phase2/final \
  2>&1 | tee outputs/logs/eval.log

echo "[$(date -Is)] copy final model card artifacts"
mkdir -p outputs/final
cp -a outputs/phase2/final/. outputs/final/
uv run python scripts/infer_demo.py --model outputs/final --query "Qual è la capitale d'Italia?" \
  2>&1 | tee outputs/logs/infer.log

echo "[$(date -Is)] all done"
