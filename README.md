# ItColBERT

Monolingual **Italian late-interaction (ColBERT)** retriever for RAG — same naming idea as [JaColBERT](https://huggingface.co/bclavie/JaColBERT) for Japanese. Trained with [PyLate](https://github.com/lightonai/pylate) on Italian IR data.

There was no dedicated Italian-only ColBERT at project start (only multilingual ColBERTs that include Italian, plus dense Italian embedders). This repo fills that gap.

## Model recipe

1. **Backbone:** `DeepMount00/Italian-ModernBERT-base` → PyLate `ColBERT` (token vectors, dim 128, MaxSim)
2. **Phase 1 — supervised contrastive:** `CachedContrastive` on `unicamp-dl/mmarco` (italian) + `nickprock/it-wiki-retrieval-synthetic-hn`
3. **Phase 2 — knowledge distillation:** `Distillation` on soft labels from a cross-encoder teacher
   - default / prior runs: `lightonai/embeddings-fine-tuning-filtered-it` (`mxbai-rerank-large-v2` scores)
   - **mixkd (current):** LightOn KD **+** `hotchpotch/mmarco-hard-negatives-reranker-filtered` (`italian-hard-negatives`, scores from `BAAI/bge-reranker-v2-m3`) so the student stays strong on both long-doc (MLDR) and short-passage (mMARCO) distributions

This follows the ColBERT-Zero / GTE-ModernColBERT sweet spot: contrastive multi-vector training, then KD (skip expensive unsupervised mega-pretrain).

## Requirements

- Linux + NVIDIA GPU (tuned for **24GB** VRAM, e.g. RTX 3090)
- [`uv`](https://docs.astral.sh/uv/)
- Hugging Face access for dataset downloads

## Setup

```bash
cd PROJECTS/it-colbert
uv sync
```

## Repository layout (JaColBERT-style naming)

| Name | Role |
|---|---|
| **ItColBERT** | Model family (like JaColBERT for Japanese) |
| `it-colbert` | Git / PyPI project name |
| `it_colbert` | Python import (`from it_colbert...`) |
| `src/it_colbert/` | Package source (standard uv layout) |
| `configs/`, `scripts/` | Training and benchmark entrypoints |

HF weights (when published) can live in a separate model repo (JaColBERT is model-card + weights only); this repo is the **training code**.

## Train

Smoke-check (short GPU validation):

```bash
uv run python scripts/train_phase1_contrastive.py --smoke
uv run python scripts/train_phase2_distill.py --smoke
```

Full train (phase 1 → phase 2 → eval → infer demo):

```bash
mkdir -p outputs/logs
bash scripts/run_full_train.sh
# or: nohup bash scripts/run_full_train.sh > outputs/logs/full_train_nohup.log 2>&1 &
# monitor: tail -f outputs/logs/phase1.log
```

Manual steps:

```bash
uv run python scripts/train_phase1_contrastive.py --config configs/phase1_contrastive.toml
uv run python scripts/train_phase2_distill.py --config configs/phase2_distill.toml

# mix kd (lighton + mmarco-it ce-scored hard negatives) → bench
bash scripts/run_phase2_mixkd.sh
# or: nohup bash scripts/run_phase2_mixkd.sh > outputs/logs/mixkd_nohup.log 2>&1 &
```

### Phase 2 checkpoint selection

Two signals are available; **prefer the IR one**.

- `ir_eval_enabled = true` runs nDCG@10 on a pooled MLDR-it slice and MRR@10 on a
  pooled mMARCO-it slice every `eval_steps`, and selects on their mean
  (`it-ir_score`, higher is better). This is the metric that penalises trading
  long-doc quality for short-passage quality.
- Hold-out KL (below) only measures how closely the student copies the teacher on
  the teacher's own distribution. fullkd selected its best-KL step while pooled
  mMARCO MRR@10 fell 0.491 → 0.341, so KL alone is not a safe selection signal.

Both can run together: KL stays in the logs for overfitting diagnosis while
`it-ir_score` drives early stopping and `load_best_model_at_end`.

### Phase 2 validation (overfitting check)

Phase 2 holds out **`kd_eval_samples`** rows from the KD train mixture (never used in train on fresh runs) and runs PyLate `ColBERTDistillationEvaluator`: **KL(student ‖ teacher)** every `eval_steps` (default 2000). This is the right signal for KD overfitting — not mMARCO train triplets alone (those overlap phase 1).

- If val KL stops improving for **`early_stopping_patience`** evals (default 3), training **stops early**
- With **`load_best_model_at_end`**, `final/` is the **best val-KL** checkpoint (lower KL is better), not necessarily the last step
- Overhead: a few minutes per eval on ~512×11 docs → roughly **+5–10%** on a long KD run
- Config keys: `kd_eval_samples`, `eval_steps`, `early_stopping_patience`, `load_best_model_at_end`, `metric_for_best_model`
- Disable val: `kd_eval_samples = 0`. Disable stop only: `early_stopping_patience = 0`

### Phase 2 mixkd (option B — mMARCO without building a new KD set)

mMARCO triplets have no soft scores, so “KD on mMARCO” needs a teacher-scored set. We reuse a **different** mMARCO-derived dataset than phase 1 (see below).

#### Datasets in the mix (shuffled together, not sequential)

| Source | Hub id | Teacher scores | Scale as stored | What we do |
|---|---|---|---|---|
| **LightOn KD** | `lightonai/embeddings-fine-tuning-filtered-it` | `mxbai-rerank-large-v2` (`rerank_scores`) | already **logits-like** (~5–9) | used as-is (same as fullkd) |
| **mMARCO HN KD** | `hotchpotch/mmarco-hard-negatives-reranker-filtered` / `italian-hard-negatives` | `BAAI/bge-reranker-v2-m3` (`pos_score`, `negs_score`) | **probs** in ~[0,1] | convert with `logit(p)=log(p/(1-p))` before KD |

Both become PyLate rows: `(query, documents[11], scores[11])`. Batches are a **random mix** (~400k LightOn + ~374k mMARCO-HN).

**Why convert only mMARCO:** ColBERTv2 / PyLate `Distillation` do `KL(log_softmax(student) ‖ log_softmax(teacher_labels))` and expect **logits**. Feeding [0,1] probs as logits flattens the teacher; LightOn needs no change.

#### Phase 1 mMARCO vs mixkd mMARCO HN — not the same dataset

| | Phase 1 contrastive | Mixkd mMARCO branch |
|---|---|---|
| Dataset | `unicamp-dl/mmarco` Italian **train triples** (tsv join) | `hotchpotch/mmarco-hard-negatives-reranker-filtered` (`italian-hard-negatives`) |
| Form | `(query, positive, negative)` — 1 neg | `(query, pos_text, negs_text[], pos_score, negs_score[])` — many HN |
| Labels | hard 0/1 (contrastive CE) | soft CE scores → logits → KL |
| Teacher | none | `bge-reranker-v2-m3` (precomputed) |

Same **domain** (Italian mMARCO train), different **resource**: phase 1 uses official triples; mixkd uses a third-party HN pack mined from mMARCO and filtered/scored by a reranker. Benchmark still uses **dev** — no test leak; train-domain overlap with phase 1 is intentional.

#### Run knobs / artifacts

- **Mix:** `include_mmarco_hn = true`; `max_train_samples` caps **LightOn only**; omit `mmarco_hn_max_samples` → all ~370k HN rows
- **Config / script:** `configs/phase2_distill_mixkd.toml`, `scripts/run_phase2_mixkd.sh`
- **Code:** `load_mmarco_hn_kd_italian()` in `data.py` (pad to 11-ways + logit convert)
- **Outputs:** `outputs/phase2_mixkd/final` → `outputs/final_mixkd`, bench name `ItColBERT (mixkd)`

Why mix: full LightOn-only KD lifted MLDR but hurt pooled mMARCO; adding CE-scored mMARCO HN targets both distributions.

The final IR quality check remains the benchmark (MLDR-it nDCG@10 / mMARCO-it **dev** MRR@10).

## Evaluate / demo

```bash
uv run python scripts/evaluate.py --model outputs/phase2/final
uv run python scripts/infer_demo.py --model outputs/phase2/final --query "Qual è la capitale d'Italia?"
```

## IR benchmark (MLDR-it + mMARCO-it)

```bash
# resume-friendly comparison (skips models already in results.json)
uv run python scripts/run_benchmark.py \
  --benchmarks mldr-it mmarco-it \
  --mmarco-max-corpus-docs 100000 \
  --output-dir outputs/benchmark

# if large baseline weights are still downloading, leave this overnight:
nohup bash scripts/wget_remaining_weights.sh > outputs/logs/wget_remaining.log 2>&1 &
# then: tail -f outputs/logs/wget_remaining.log
```

Protocol notes:

- **MLDR-it:** full Italian corpus (~10k docs), official test split; primary metric **nDCG@10**
- **mMARCO-it (this repo):** pooled **100k** docs (qrel positives + reservoir sample) for relative ranking. Absolute MRR is inflated vs published full 8.8M-corpus numbers
- **MIRACL:** no Italian language split; not used
- **Length matching:** pass `--colbert-doc-length 512` so every late-interaction model is indexed at the same length. Earlier runs indexed `jina-colbert-v2` at 180 tokens and ours at 512, which handicapped the strongest baseline on long documents; the per-model `effective_length` is now recorded in `results.json`
- **Long-doc mode:** `--chunk-chars 2000 --chunk-overlap-chars 200` indexes chunks and max-pools per document, so text past the encoder's truncation point still counts
- **BM25:** Italian analyzer (lowercase, punctuation strip, stopwords, Snowball stem). The old `text.lower().split()` understated the lexical baseline
- Pending comparative ColBERTs: `lightonai/mLateOn`, `antoinelouis/colbert-xm` (wired in code)

### Results so far (`outputs/benchmark/results.json`)

MLDR-it (nDCG@10): bge-m3 **0.453** · e5-large **0.431** · jina-colbert-v2 **0.369** · **fullkd 0.352** · BM25 **0.332** · **ours@512 0.277** · doc256 **0.227**

mMARCO-it pooled 100k (MRR@10): jina **0.849** · e5-large **0.824** · **ours 0.491** · doc256 **0.411** · **fullkd 0.341** (full LightOn KD over-specialized away from short passages)

Published full-corpus mMARCO-it MRR@10 (literature): jina-colbert-v2 0.337 · mColBERT 0.292 · mE5-base 0.280 · BM25 0.153

> ⚠️ **These numbers came from a buggy KD split budget — see below.** They are kept
> as the record of what was actually run, but the conclusion drawn from them
> ("KD scale lifts MLDR") was wrong.

### KD split budget — the bug behind the fullkd/80k trade-off

`load_kd_italian` used to drain the LightOn splits **in order** and stop when
`max_train_samples` ran out. `msmarco_it` is first and holds ~522k of the ~1.56M
rows, so:

| run | budget | splits it actually saw |
|---|---|---|
| ours (80k KD) | 80,000 | **msmarco_it only** |
| mixkd | 400,000 | **msmarco_it only** + mMARCO HN |
| fullkd | uncapped | all 8 splits |

So the MLDR gain 0.277 → 0.352 was **not** KD scale. It was data composition:
fullkd is the only run that ever saw `trivia_it`, `nq_it`, `hotpotqa_it`,
`fever_it`, `squadv2_it`. (`mldr_it` itself is only ~1.4k rows, 0.09% of the set,
so it contributed almost nothing either way.) And mixkd was never a mixture at
all — both of its branches were short-passage MS MARCO.

Fixed: `kd_split_sampling = "proportional"` (now the default) spends the budget
across every split by row count, with a `kd_split_min_share` floor so the small
on-domain splits survive. The legacy configs pin `"sequential"` so the runs above
stay reproducible.

**Next:** the v2 recipe — see [TODO.md](TODO.md).

## Citation / related work

- ColBERT / ColBERTv2 (Stanford)
- PyLate (LightOn)
- mMARCO (unicamp-dl)
- Italian-ModernBERT (DeepMount00)