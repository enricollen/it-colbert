#!/usr/bin/env bash
# v2 recipe end to end: phase 1 contrastive -> phase 2 kd -> length-matched bench.
# see TODO.md for what changed and why, and for the ablation runs to do after this.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p outputs/logs

export PYTHONUNBUFFERED=1

# 1) warm the dataset cache while the network is available
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE HF_DATASETS_OFFLINE || true
echo "[$(date -Is)] caching datasets"
uv run python -u - <<'PY'
from it_colbert.data import (
    ITALIAN_CONTRASTIVE_SOURCES,
    load_kd_italian,
    load_mmarco_hn_contrastive_italian,
    load_mmarco_hn_kd_italian,
    load_tevatron_style_contrastive_italian,
)

load_kd_italian(max_samples=64, seed=0, split_sampling="sequential")
load_mmarco_hn_kd_italian(max_samples=64, seed=0)
load_mmarco_hn_contrastive_italian(max_samples=64, seed=0)
for dataset_id, split in ITALIAN_CONTRASTIVE_SOURCES:
    load_tevatron_style_contrastive_italian(dataset_id, split=split, max_samples=64)
print("datasets cached", flush=True)
PY

# 2) train offline so hub lookups on local checkpoints cannot hang the run
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1

echo "[$(date -Is)] phase 1 v2 start"
uv run python -u scripts/train_phase1_contrastive.py \
  --config configs/phase1_contrastive.toml \
  2>&1 | tee -a outputs/logs/phase1.log

# benchmark phase 1 on its own before distilling. ColBERT-Zero reports that a
# supervised contrastive stage plus KD reaches ~99.4% of full multi-vector
# pretraining, but the marginal value of the KD stage was never measured here
# against a phase 1 that was not broken. this costs ~30 minutes and can redirect
# where the rest of the GPU budget goes.
echo "[$(date -Is)] publish + benchmark phase1-only ablation"
mkdir -p outputs/final_phase1
cp -a outputs/phase1/final/. outputs/final_phase1/
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE HF_DATASETS_OFFLINE || true
uv run python -u scripts/run_benchmark.py \
  --benchmarks mldr-it mmarco-it \
  --mmarco-max-corpus-docs 100000 \
  --colbert-doc-length 512 \
  --only "ItColBERT (phase1-only)" \
  --top-k 100 \
  --output-dir outputs/benchmark \
  2>&1 | tee outputs/logs/bench_phase1.log
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

echo "[$(date -Is)] phase 2 v2 start"
# the IR evaluator needs the benchmark corpora, which were cached in step 1
unset HF_DATASETS_OFFLINE || true
RESUME_ARGS=()
if [[ -n "${RESUME_FROM:-}" ]]; then
  RESUME_ARGS=(--resume "$RESUME_FROM")
fi
uv run python -u scripts/train_phase2_distill.py \
  --config configs/phase2_distill.toml \
  "${RESUME_ARGS[@]}" \
  2>&1 | tee -a outputs/logs/phase2.log

echo "[$(date -Is)] publish outputs/final"
mkdir -p outputs/final
cp -a outputs/phase2/final/. outputs/final/

echo "[$(date -Is)] benchmark (length-matched at 512, full italian suite)"
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE HF_DATASETS_OFFLINE || true
export HF_XET_HIGH_PERFORMANCE=1
uv run python -u scripts/run_benchmark.py \
  --benchmarks mldr-it mmarco-it miracl-ita squad-ita \
  --mmarco-max-corpus-docs 100000 \
  --extra-max-corpus-docs 50000 \
  --colbert-doc-length 512 \
  --only "ItColBERT" \
  --top-k 100 \
  --output-dir outputs/benchmark \
  2>&1 | tee outputs/logs/bench.log

echo "[$(date -Is)] v2 done — next: baselines (TODO.md step 6), then mining (step 8)"
