#!/usr/bin/env bash
# step 1: scale kd from phase1 checkpoint (doc180), then bench only the new model
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p outputs/logs

# line-buffered logs under nohup/tee (otherwise loss lines appear minutes late)
export PYTHONUNBUFFERED=1
# saving checkpoints triggers a model-card hub lookup on local path "phase1/final"
# which can hang on http; keep training offline (bench re-enables network below)
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

RESUME_ARGS=()
if [[ -n "${RESUME_FROM:-}" ]]; then
  RESUME_ARGS=(--resume "$RESUME_FROM")
elif [[ -d outputs/phase2_fullkd ]]; then
  latest="$(ls -d outputs/phase2_fullkd/checkpoint-* 2>/dev/null | sort -V | tail -1 || true)"
  if [[ -n "${latest}" ]]; then
    RESUME_ARGS=(--resume "$latest")
    echo "[$(date -Is)] auto-resume from $latest"
  fi
fi

echo "[$(date -Is)] phase 2 full kd start"
uv run python -u scripts/train_phase2_distill.py --config configs/phase2_distill_fullkd.toml \
  "${RESUME_ARGS[@]}" \
  2>&1 | tee -a outputs/logs/phase2_fullkd.log
echo "[$(date -Is)] phase 2 full kd done"

echo "[$(date -Is)] copy final_fullkd"
mkdir -p outputs/final_fullkd
cp -a outputs/phase2_fullkd/final/. outputs/final_fullkd/

echo "[$(date -Is)] bench only ItColBERT fullkd"
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE HF_DATASETS_OFFLINE || true
unset HF_HUB_ENABLE_HF_TRANSFER || true
export HF_XET_HIGH_PERFORMANCE=1
uv run python -u scripts/run_benchmark.py \
  --benchmarks mldr-it mmarco-it \
  --mmarco-max-corpus-docs 100000 \
  --only "ItColBERT (fullkd)" \
  --top-k 100 \
  --output-dir outputs/benchmark \
  2>&1 | tee outputs/logs/bench_fullkd.log

echo "[$(date -Is)] all done fullkd"
