#!/usr/bin/env bash
# end to end: phase 1 -> phase1-only benchmark -> phase 2 -> benchmark.
#
# SAFE TO INTERRUPT. Ctrl-C or kill it at any point and re-run the same command:
#   - finished stages are detected via their `final/` model and skipped
#   - an interrupted stage resumes from its newest COMPLETE checkpoint
#     (a checkpoint truncated by an interrupt mid-save is ignored)
#   - the benchmark skips models already present in results.json
#   - the built phase-1 dataset is cached, so restarts do not re-index the
#     8.8M-passage mmarco collection
#
# run one stage only:  STAGE=phase1 bash scripts/run_train.sh
# force a fresh start:  rm -rf outputs/phase1   (or outputs/phase2)
#
# see TODO.md for the reasoning behind each stage and the follow-up runs.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p outputs/logs

export PYTHONUNBUFFERED=1
STAGE="${STAGE:-all}"

log() { echo "[$(date -Is)] $*"; }

stage_done() {
  # a stage is finished once it wrote its final model
  [[ -f "$1/final/model.safetensors" || -f "$1/final/pytorch_model.bin" ]]
}

want_stage() { [[ "$STAGE" == "all" || "$STAGE" == "$1" ]]; }

# a typo in STAGE would otherwise run nothing and exit 0, looking like success
case "$STAGE" in
  all|cache|phase1|phase1_bench|phase2|bench) ;;
  *)
    echo "unknown STAGE=$STAGE (expected: all|cache|phase1|phase1_bench|phase2|bench)" >&2
    exit 2
    ;;
esac

# ---------------------------------------------------------------- cache -----
# warm every dataset while the network is up. cheap to repeat: everything below
# is a cache hit after the first run.
if want_stage cache; then
  unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE HF_DATASETS_OFFLINE || true
  log "caching datasets"
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

# both phases run the IR evaluator, which needs the benchmark corpora
from it_colbert.benchmark.datasets import load_mldr_italian, load_mmarco_italian_dev

load_mldr_italian(split="test")
load_mmarco_italian_dev(max_corpus_docs=50_000)
print("datasets cached", flush=True)
PY
fi

# train with the hub offline so model-card lookups on local checkpoint paths
# cannot hang the run. datasets stay ONLINE: both phases run the IR evaluator,
# which loads the benchmark corpora (already cached, so this costs nothing).
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

# -------------------------------------------------------------- phase 1 -----
if want_stage phase1; then
  if stage_done outputs/phase1; then
    log "phase 1 already finished (outputs/phase1/final exists) — skipping"
  else
    log "phase 1 start (auto-resumes if a checkpoint exists)"
    uv run python -u scripts/train_phase1_contrastive.py \
      --config configs/phase1_contrastive.toml \
      2>&1 | tee -a outputs/logs/phase1.log
    log "phase 1 done"
  fi
fi

# ------------------------------------------------- phase1-only ablation -----
# benchmark phase 1 alone before distilling. ColBERT-Zero reports contrastive +
# KD reaching ~99.4% of full multi-vector pretraining, but the marginal value of
# the KD stage was never measured here against a phase 1 that was not broken.
# ~30 minutes, and it can redirect where the rest of the GPU budget goes.
if want_stage phase1_bench; then
  if ! stage_done outputs/phase1; then
    log "skipping phase1-only benchmark: outputs/phase1/final does not exist yet"
  else
  log "publish + benchmark phase1-only ablation"
  mkdir -p outputs/final_phase1
  cp -a outputs/phase1/final/. outputs/final_phase1/
  unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE || true
  uv run python -u scripts/run_benchmark.py \
    --benchmarks mldr-it mmarco-it \
    --only "ItColBERT (phase1-only)" \
    --output-dir outputs/benchmark \
    2>&1 | tee -a outputs/logs/bench_phase1.log
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
  fi
fi

# -------------------------------------------------------------- phase 2 -----
if want_stage phase2; then
  if stage_done outputs/phase2; then
    log "phase 2 already finished (outputs/phase2/final exists) — skipping"
  else
    log "phase 2 start (auto-resumes if a checkpoint exists)"
    uv run python -u scripts/train_phase2_distill.py \
      --config configs/phase2_distill.toml \
      2>&1 | tee -a outputs/logs/phase2.log
    log "phase 2 done"
  fi
fi

# ------------------------------------------------------------ benchmark -----
if want_stage bench; then
  if ! stage_done outputs/phase2; then
    log "skipping final benchmark: outputs/phase2/final does not exist yet"
  else
  log "publish outputs/final"
  mkdir -p outputs/final
  cp -a outputs/phase2/final/. outputs/final/

  log "benchmark (all four suites, length-matched at 512)"
  unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE || true
  export HF_XET_HIGH_PERFORMANCE=1
  uv run python -u scripts/run_benchmark.py \
    --only "ItColBERT" \
    --output-dir outputs/benchmark \
    2>&1 | tee -a outputs/logs/bench.log
  fi
fi

log "done — next: TODO.md 6 (statistics), then 8 (mining)"
