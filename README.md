# ItColBERT

Monolingual **Italian late-interaction (ColBERT)** retriever for search and RAG,
trained with [PyLate](https://github.com/lightonai/pylate).

Goal, stated plainly: the best Italian-specialized late-interaction retriever at
base size, with **reproducible Italian evaluation**. Not "the first Italian
ColBERT" — [`SauerkrautLM-Multi-ModernColBERT`](https://huggingface.co/VAGOsolutions/SauerkrautLM-Multi-ModernColBERT)
already covers Italian, and it is in the benchmark as the model to beat.

This repo is the **training and evaluation code**. Weights live in a separate
model repo.

---

## Recipe

1. **Backbone:** [`nickprock/Italian-ModernBERT-base-embed-mmarco-mnrl`](https://huggingface.co/nickprock/Italian-ModernBERT-base-embed-mmarco-mnrl)
   → PyLate `ColBERT` (token vectors, dim 128, MaxSim).
   Starting from a checkpoint that already retrieves, rather than a raw MLM, is
   the [ColBERT-Zero](https://huggingface.co/blog/lightonai/colbert-zero)
   efficiency result: supervised contrastive + distillation from a dense init
   reaches ~99.4% of full multi-vector pretraining at roughly 10× lower cost.
2. **Phase 1 — supervised contrastive** (`CachedContrastive`) on mMARCO-it
   triples, reranker-mined mMARCO hard negatives, Italian wiki retrieval pairs,
   MIRACL-ita and SQuAD-ita.
3. **Phase 2 — knowledge distillation** (`Distillation`, KL) from a single
   cross-encoder teacher: [`lightonai/embeddings-fine-tuning-filtered-it`](https://huggingface.co/datasets/lightonai/embeddings-fine-tuning-filtered-it)
   (`mxbai-rerank-large-v2` scores), spread proportionally across all 8 splits.
4. **Checkpoint selection on retrieval metrics** (nDCG@10 / MRR@10), not on
   hold-out KD KL divergence.

See [TODO.md](TODO.md) for the runbook and the reasoning behind each choice.

---

## Requirements

- Linux + NVIDIA GPU (sized for **24 GB**, e.g. RTX 3090)
- [`uv`](https://docs.astral.sh/uv/)
- Hugging Face access for dataset downloads

```bash
uv sync
mkdir -p outputs/logs
```

---

## Layout

| Path | Role |
|---|---|
| `src/it_colbert/` | package: data loaders, training phases, IR evaluator |
| `src/it_colbert/benchmark/` | retrieval benchmark: datasets, retrievers, stats |
| `configs/` | training configs (TOML) |
| `scripts/` | entrypoints and diagnostics |
| `TODO.md` | execution plan, in order, with the reasoning |

### Configs

| File | What it is |
|---|---|
| `phase1_contrastive.toml` | phase 1, dim 128 — the main recipe |
| `phase2_distill.toml` | phase 2, dim 128 — single-teacher KD |
| `phase1_contrastive_dim64.toml` / `phase2_distill_dim64.toml` | half-size index variant |
| `phase2_distill_mmarco_hn.toml` | ablation: adds a second teacher (see below) |
| `smoke.toml` | tiny overrides for a fast pipeline check |

### Scripts

| Script | Purpose |
|---|---|
| `run_train.sh` | phase 1 → phase1-only benchmark → phase 2 → benchmark |
| `train_phase1_contrastive.py` / `train_phase2_distill.py` | individual phases |
| `inspect_kd_scores.py` | teacher sharpness + real documents-per-row (sets `kd_n_ways`) |
| `run_benchmark.py` | Italian IR benchmark across all models |
| `compare_models.py` | paired bootstrap: is a gap real? |
| `mine_hard_negatives.py` | round-2 self-mined negatives |
| `run_mteb_italian.py` | MMTEB Italian task slice |
| `infer_demo.py` / `push_to_hub.py` | demo and publishing |

---

## Train

Smoke-check first — it exercises every changed path in a couple of minutes:

```bash
uv run python scripts/train_phase1_contrastive.py --config configs/phase1_contrastive.toml --smoke
uv run python scripts/train_phase2_distill.py     --config configs/phase2_distill.toml     --smoke
```

Full run:

```bash
nohup bash scripts/run_train.sh > outputs/logs/train_nohup.log 2>&1 &
tail -f outputs/logs/phase1.log
```

### Validation

Both phases run `ItIREvaluator` (nDCG@10 / MRR@10 on small pooled slices)
alongside their cheap proxy metric.

**Phase 1** additionally keeps `ColBERTTripletEvaluator` as a divergence check,
and has early stopping **off**: one epoch over ~1.7M unique triplets cannot
overfit, and stopping early truncates the warmup+decay schedule, leaving the model
at a high learning rate. If the IR curve is still climbing at the end, train
longer — do not stop sooner.

**Phase 2** selects checkpoints. Two signals are available; **prefer the retrieval
one**.

- `ir_eval_enabled = true` runs nDCG@10 on a pooled MLDR-it slice and MRR@10 on a
  pooled mMARCO-it slice every `eval_steps`, selecting on their mean
  (`it-ir_score`, higher is better). This is the metric that penalises trading
  long-document quality for short-passage quality.
- Hold-out KL only measures how closely the student copies the teacher on the
  teacher's own distribution. An earlier run selected its best-KL step while
  pooled mMARCO MRR@10 fell 0.491 → 0.341 — KL alone is not a safe signal.

Both can run together: KL stays in the logs for overfitting diagnosis while
`it-ir_score` drives early stopping and `load_best_model_at_end`.

### KD split budget

`max_train_samples` is spent **proportionally** across the 8 LightOn splits, with
a `kd_split_min_share` floor so small on-domain splits survive. An earlier version
drained splits in order; since `msmarco_it` holds ~522k of ~1.56M rows, every
capped run was 100% mMARCO. That produced the mistaken conclusion that "more KD
improves long-document retrieval" when the real variable was data composition.

`kd_split_sampling = "sequential"` restores the old behaviour if you ever need it.

### Why one teacher

`include_mmarco_hn = false` by default. The alternative mixes
`mxbai-rerank-large-v2` (LightOn) with `bge-reranker-v2-m3` (mMARCO hard
negatives) in one KL loss, so the student imitates an average of two different
opinions about relevance — and it is redundant, because LightOn already contains
an `msmarco_it` split scored by the same teacher. `phase2_distill_mmarco_hn.toml`
keeps it available as an ablation.

---

## Benchmark

```bash
uv run python scripts/run_benchmark.py --output-dir outputs/benchmark
```

Defaults: all four benchmarks, every ColBERT indexed at 512, mMARCO pooled to
100k, bootstrap intervals on. Override with `--benchmarks`, `--colbert-doc-length`
and `--mmarco-max-corpus-docs`.

| Benchmark | Source | Corpus | Primary metric |
|---|---|---|---|
| `mldr-it` | `Shitao/MLDR` italian test | full (~10k long docs) | nDCG@10 |
| `mmarco-it` | `unicamp-dl/mmarco` italian dev | pooled | MRR@10 (**rank only**) |
| `miracl-ita` | `yuri-no/miracl-ita-argos` dev | pooled | nDCG@10 |
| `squad-ita` | `yuri-no/squad-ita` test | pooled | nDCG@10 |

MIRACL-ita and SQuAD-ita are **community machine translations**, not official
resources. Label them as such anywhere you publish.

### Protocol

- **Length matching is the default** (`--colbert-doc-length 512`), not an opt-in.
  Earlier runs indexed `jina-colbert-v2` at 180 tokens and ours at 512,
  handicapping the strongest baseline on long documents, and nothing in the output
  revealed it. The per-model `effective_length` is now recorded in `results.json`.
- **Long-document mode.** `--chunk-chars 2000 --chunk-overlap-chars 200` indexes
  chunks and max-pools per document, so text past the encoder's truncation point
  still counts.
- **BM25** uses an Italian analyzer (lowercase, punctuation strip, stopwords,
  Snowball stem). A whitespace tokenizer badly understates the lexical baseline in
  an inflected language.
- **Pooled corpora are rank-only.** On a 100k pooled mMARCO, `jina-colbert-v2`
  scores 0.849 MRR@10 against a published full-corpus 0.337. `results.json` marks
  these splits `comparable_to_literature: false`.
- **MIRACL** has no official Italian split; the community translation is used
  instead.

### Significance

**MLDR-it has 200 queries.** The standard error on nDCG@10 there is roughly
±0.02–0.03, so differences under ~0.03 are not established by a single run. The
benchmark writes 95% bootstrap intervals into `results.json` and per-query scores
into `outputs/benchmark/per_query/`. To test a specific gap:

```bash
uv run python scripts/compare_models.py --benchmark-dir outputs/benchmark \
  --benchmark mldr-it --a "ItColBERT" --b "jina-colbert-v2"

# everything against one reference, on every benchmark:
uv run python scripts/compare_models.py --benchmark-dir outputs/benchmark \
  --baseline "ItColBERT" --all
```

---

## Demo

```bash
uv run python scripts/infer_demo.py --model outputs/final --query "Qual è la capitale d'Italia?"
```

---

## Related work

- ColBERT / ColBERTv2 (Stanford)
- [PyLate](https://github.com/lightonai/pylate) and [ColBERT-Zero](https://huggingface.co/blog/lightonai/colbert-zero) (LightOn)
- [mxbai-edge-colbert-v0](https://arxiv.org/pdf/2510.14880) — hard-negative mining and data composition as the primary quality drivers
- mMARCO (unicamp-dl), MLDR (BAAI)
- Italian-ModernBERT (DeepMount00), Italian embedding models (nickprock)
