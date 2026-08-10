#!/usr/bin/env bash
# phase2 mix kd: lighton italian + mmarco-it ce-scored hard negatives, then bench
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p outputs/logs

export PYTHONUNBUFFERED=1

# 1) cache datasets online (needed once for mmarco hn shards)
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE HF_DATASETS_OFFLINE || true
echo "[$(date -Is)] ensuring mix kd datasets are cached"
uv run python -u - <<'PY'
from it_colbert.data import load_kd_italian, load_mmarco_hn_kd_italian
print("cache lighton sample...", flush=True)
load_kd_italian(max_samples=8, seed=0)
print("cache mmarco hn...", flush=True)
ds = load_mmarco_hn_kd_italian(max_samples=None, seed=0)
print("mmarco hn kd rows:", len(ds), flush=True)
print(ds[0].keys(), "n_docs", len(ds[0]["documents"]), flush=True)
PY

# 2) train offline to avoid hub hangs on local phase1/final model-card lookup
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

RESUME_ARGS=()
if [[ -n "${RESUME_FROM:-}" ]]; then
  RESUME_ARGS=(--resume "$RESUME_FROM")
elif [[ -d outputs/phase2_mixkd ]]; then
  latest="$(ls -d outputs/phase2_mixkd/checkpoint-[0-9]* 2>/dev/null | sort -V | tail -1 || true)"
  if [[ -n "${latest}" ]]; then
    RESUME_ARGS=(--resume "$latest")
    echo "[$(date -Is)] auto-resume from $latest"
  fi
fi

echo "[$(date -Is)] phase 2 mix kd start"
uv run python -u scripts/train_phase2_distill.py --config configs/phase2_distill_mixkd.toml \
  "${RESUME_ARGS[@]}" \
  2>&1 | tee -a outputs/logs/phase2_mixkd.log
echo "[$(date -Is)] phase 2 mix kd done"

echo "[$(date -Is)] copy final_mixkd"
mkdir -p outputs/final_mixkd
cp -a outputs/phase2_mixkd/final/. outputs/final_mixkd/

echo "[$(date -Is)] bench only ItColBERT mixkd"
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE HF_DATASETS_OFFLINE || true
unset HF_HUB_ENABLE_HF_TRANSFER || true
export HF_XET_HIGH_PERFORMANCE=1
uv run python -u scripts/run_benchmark.py \
  --benchmarks mldr-it mmarco-it \
  --mmarco-max-corpus-docs 100000 \
  --only "ItColBERT (mixkd)" \
  --top-k 100 \
  --output-dir outputs/benchmark \
  2>&1 | tee outputs/logs/bench_mixkd.log

echo "[$(date -Is)] all done mixkd"
