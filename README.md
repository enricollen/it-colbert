# ItColBERT

![ItColBERT banner](assets/banner.jpg)

Monolingual **Italian late-interaction (ColBERT)** retriever for search and RAG,
trained with [PyLate](https://github.com/lightonai/pylate).

**[Model on Hugging Face →](https://huggingface.co/enricollen/ItColBERT)**
· License: Apache-2.0 · This repo: training + evaluation code and the full
development history. Weights and the polished model card live in the HF repo
above.

## What this is

Most retrieval embedding models compress a whole passage into a single
vector. ColBERT-style **late interaction** models keep one small vector per
*token* instead, and score a document by matching each query token against
its best document token (MaxSim) rather than comparing two averaged blobs.
That keeps fine-grained detail — exact names, numbers, phrasing — that gets
blurred away by single-vector compression, at the cost of a larger index.

Before building this, the Italian options were: multilingual late-interaction
models that *include* Italian among many languages
([`jina-colbert-v2`](https://huggingface.co/jinaai/jina-colbert-v2),
[`mLateOn`](https://huggingface.co/lightonai/mLateOn),
[`ColBERT-XM`](https://huggingface.co/antoinelouis/colbertxm),
[`SauerkrautLM-Multi-ModernColBERT`](https://huggingface.co/VAGOsolutions/SauerkrautLM-Multi-ModernColBERT)),
or strong Italian dense embedders that drop late interaction entirely.
Nothing combined the two — an Italian-*specialized* model that keeps
token-level matching. That gap is what this project fills. Not "the first
Italian ColBERT" (the models above already cover Italian); the first one
specialized on it.

## Results at a glance

Real numbers, paired-bootstrap significance tested. **†** = not statistically
distinguishable from ItColBERT (p > .05); read those as ties regardless of
which number is higher. Full protocol, caveats, and confidence intervals in
the [model card](https://huggingface.co/enricollen/ItColBERT).

| Model | MLDR-it (nDCG@10) | mMARCO-it (MRR@10) | MIRACL-ita (nDCG@10) | SQuAD-ita (nDCG@10) |
|---|---|---|---|---|
| **ItColBERT** | **0.4008** (0.4610 chunked) | **0.7196** | **0.7194** | **0.9026** |
| mLateOn | 0.4623 | 0.8207 | 0.7880 | 0.9480 |
| jina-colbert-v2 | 0.3858 † | 0.8389 | 0.7755 | 0.8849 |
| bge-m3 (dense) | 0.4531 | 0.7812 | 0.7566 | 0.8247 |
| multilingual-e5-large (dense) | 0.4310 † | 0.8239 | 0.7653 | 0.8513 |
| SauerkrautLM-Multi-ModernColBERT | 0.3122 | 0.5342 | 0.5996 | 0.8338 |
| ColBERT-XM | 0.2734 | 0.6654 | 0.6260 | 0.8558 |
| BM25 | 0.4850 (vs. 0.4610 chunked: †) | 0.5715 | 0.5516 | 0.8262 |

The strongest **Italian-specialized** late-interaction model tested here,
beating every general-purpose late-interaction alternative except one
(mLateOn, the strongest multilingual late-interaction model found). Behind
large multilingual dense embedders on most benchmarks — matching those was
never the goal, they're a different model class.

## Try it

```bash
pip install -U pylate
```

```python
from pylate import rank, models

model = models.ColBERT(model_name_or_path="enricollen/ItColBERT")

queries = ["Qual è la capitale d'Italia?"]
documents = [[
    "Roma è la capitale d'Italia.",
    "Milano è la capitale economica del Paese.",
]]
documents_ids = [[1, 2]]

queries_embeddings = model.encode(queries, is_query=True)
documents_embeddings = model.encode(documents, is_query=False)

reranked = rank.rerank(
    documents_ids=documents_ids,
    queries_embeddings=queries_embeddings,
    documents_embeddings=documents_embeddings,
)
print(reranked)
# [[{'id': 1, 'score': 31.682}, {'id': 2, 'score': 31.552}, {'id': 3, 'score': 31.454}]]
# one list per query, sorted highest score first — "Roma" wins, as expected.
```

Indexing a full corpus, RAG integration notes, and the long-document chunking
recipe: see the [model card](https://huggingface.co/enricollen/ItColBERT).

## The journey, briefly

The short version of three rounds of work, including what didn't work —
full detail and numbers in [TODO.md](TODO.md):

1. **Start from a model that already retrieves**, not a raw language model —
   the [ColBERT-Zero](https://huggingface.co/blog/lightonai/colbert-zero)
   efficiency result: supervised contrastive + distillation from a retrieval
   -capable init reaches ~99% of full multi-vector pretraining at ~10× lower
   cost.
2. **Round 1 — broaden the data, then distil.** The starting checkpoint only
   knew machine-translated mMARCO. Added more varied Italian sources, then a
   distillation stage from a cross-encoder teacher. This is the model above —
   strongest Italian-specialized late-interaction model tested, weakest on
   long documents.
3. **Found the long-document weakness was mostly mechanical.** The hardest
   benchmark's documents run a few thousand words; they were being truncated
   at 512 tokens, discarding ~80% of the average document. Chunking documents
   at query time — no retraining — recovered most of the gap (+0.06 nDCG@10,
   the largest single gain in the project).
4. **Round 2 — tried mining harder training examples** from the model's own
   predictions. No improvement, and a small real drop in generalization.
   Rejected.
5. **Round 3 — tried training the model to read twice as much text per
   document** natively, instead of relying on chunking. Statistically no
   better than chunking a normally-trained model, once compared fairly on
   held-out data. Rejected.
6. **What shipped:** since neither training round beat "train normally, then
   chunk long documents at query time," that's what's here.

<details>
<summary><strong>The full timeline, with numbers (click to expand)</strong></summary>

<br>

**Round 1 — the model that shipped.** The backbone
([`nickprock/Italian-ModernBERT-base-embed-mmarco-mnrl`](https://huggingface.co/nickprock/Italian-ModernBERT-base-embed-mmarco-mnrl))
already retrieved on its own (0.302 MLDR-it nDCG@10 alone), but its training
data was mMARCO only — machine-translated, and the axis it was already
strongest on. Phase 1 broadened that with reranker-mined hard negatives plus
Italian wiki, MIRACL-ita and SQuAD-ita, ~2.4M triplets, one epoch, lr 1e-5,
batch 512 (~12.6h). That alone reached 0.3484 MLDR-it nDCG@10. Phase 2
distilled from a single cross-encoder teacher (`mxbai-rerank-large-v2` via
LightOn's dataset), sample budget spread proportionally across all 8 splits
so the mixture couldn't collapse to one dominant split (an earlier bug had
done exactly that — see TODO.md §0.1). Final result: MLDR-it 0.4008, mMARCO-it
0.7196 MRR@10, MIRACL-ita 0.7194, SQuAD-ita 0.9026 — 3 of 4 up over phase-1
alone, with the sole regression (mMARCO, −0.028) landing on the one axis that
was already in-domain. Ranked ahead of every other late-interaction model
tested except one (mLateOn).

**The chunking discovery — the single largest gain in the project.**
MLDR-it's documents run a median 2,666 tokens against the 512-token index —
every document truncates, and only ~20% of the corpus's tokens were ever
encoded. Splitting documents into 2,000-character overlapping chunks and
max-pooling scores at query time, with the *same, already-trained*
checkpoint — no retraining at all — moved MLDR-it nDCG@10 from 0.4008 to
0.4610 (+0.0602, p=0.0225, the largest single measured effect in the whole
project). It also reframed an earlier, wrong headline: "loses to BM25 on the
only clean out-of-domain benchmark" turned out to be a protocol artifact —
BM25 reads the whole document, the truncated model didn't. Read at equal
document access, it's a statistical tie (p=0.249), not a loss. Cost: ~7×
wall-clock, ~26GB host RAM — right at this machine's ceiling, as later rounds
found out the hard way.

**Round 2 — mining harder negatives, rejected.** Hypothesis: the model's own
retrieval mistakes on mMARCO could mine better negatives than the original
BM25-sampled ones. Mined 46,583 rows (200k-document pool, `--skip-top 5` to
avoid unlabelled true positives) and retrained phase 1 with them added.
Result: MLDR-it −0.0473 (not significant alone, n=200), but mMARCO-it MRR@10
−0.0277 (n=6980, *significant* — the mining made the axis it targeted worse,
not better). Phase 2's fixes (a replay stream, lower learning rate) partially
recovered mMARCO (net +0.0100 over round 1) but MIRACL-ita and SQuAD-ita both
moved down significantly, and MLDR-it never moved outside noise (0.4008 →
0.3779, p=0.082). Root cause, diagnosed after the fact: both correctives —
the mined negatives and the phase-2 replay stream — drew from mMARCO only,
so the model got sharper on the axis that was already strongest and never
touched the one benchmark that mattered. Round 1 stayed the release
candidate.

**Round 3 (this README calls it "Track B") — training at length, rejected.**
Given chunking's result, the next hypothesis was obvious: what if the model
were trained to natively read past 512 tokens, instead of relying on
inference-time chunking? Vetted three long-document Italian sources before
spending any GPU time; two turned out to be pre-chunked and short (median
117–439 tokens — no better than what the model already saw). The one that
worked: `hotchpotch/wikipedia-multilingual-synthetic-ir-query`'s
`it-long_doc` split — 650,885 rows, median ~1,033 tokens, and critically, the
query's supporting evidence starts past token 512 in 30.7% of rows (past
1,024 in 14.2%). Checked and removed the 7.8% of articles that overlapped
MLDR-it's own test corpus before using it.

Ran a controlled A/B: two phase-1-only models on an identical 500k-triplet
mixture (250k mMARCO + 250k `it-long_doc`), differing in exactly one thing —
document length, 512 vs 1024 tokens. Gate: does the 1024 arm beat the 512 arm
on held-out MLDR-it by more than the measured noise floor (0.0030), without
regressing mMARCO/MIRACL? Result: unchunked, +0.0293 (not significant, p=0.35,
n=104 held-out queries). Chunked — arguably the fairer comparison, since it's
each arm's *stronger* protocol — essentially a dead tie: −0.0032, p=0.85.
Best-of-arm (each arm's better of chunked/unchunked) nominally favored the
*512* arm, not the one trained at length. Guardrails on mMARCO/MIRACL were
clean, but moot — the primary comparison never cleared the bar either way.

(One real infrastructure bug surfaced running this: the documented chunked-
benchmark command was missing a required flag, which silently routed the run
through a broken ANN fallback and OOM-killed both arms on a 27GB-RAM
machine. Fixed and reran cleanly at ~999s/arm once the flag was restored —
worth knowing if you're reproducing any chunked benchmark from this repo.)

**Conclusion:** two independent training-based attempts (round 2, round 3)
both failed to beat a training-free trick (chunking) discovered by
measurement, not by guessing. That's the actual argument for why chunking,
not further training, is the right lever here — and why `outputs/final_round1`
plus the chunking recipe is what shipped as v1.

</details>

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

Everything in this repo — every training run, every benchmark, all significance
testing — was done on **one consumer machine**, not a cluster: a single RTX 3090
(24GB), Intel Core i7-14700K, 32GB RAM (27GB usable — WSL2 caps it below the
physical total, Ubuntu 22.04). No multi-GPU, no cloud compute. That ceiling
is also why some decisions in [TODO.md](TODO.md) look the way they do
(mini-batch sizes, chunking's ~26GB host-RAM cost, the OOM kills documented
in §13.4 — WSL2's 27GB cap, not the physical 32GB, is what got hit).

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
| `run_train.sh` | phase 1 → phase1-only benchmark → phase 2 → benchmark; interruptible and resumable |
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

### Interrupt and resume

The pipeline is built to be stopped overnight. Kill it at any point and re-run the
same command: finished stages are skipped via their `final/` model, an interrupted
stage resumes from its newest **complete** checkpoint (one truncated by an
interrupt mid-save is detected and ignored), the benchmark skips models already in
`results.json`, and the built phase-1 dataset is cached under
`outputs/dataset_cache` so restarts do not re-index the 8.8M-passage mMARCO
collection.

```bash
STAGE=phase1 bash scripts/run_train.sh    # one stage only
rm -rf outputs/phase1                     # genuinely start that stage over
uv run python scripts/train_phase1_contrastive.py --no-auto-resume   # same, per-phase
```

Stages: `cache`, `phase1`, `phase1_bench`, `phase2`, `bench`.

### Validation

Both phases run `ItIREvaluator` (nDCG@10 / MRR@10 on small pooled slices)
alongside their cheap proxy metric.

**Phase 1** additionally keeps `ColBERTTripletEvaluator` as a divergence check,
and has early stopping **off**: one epoch over ~2.4M triplets cannot
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

To score a checkpoint that is not in `DEFAULT_MODELS` — an intermediate
`checkpoint-N`, a previous round's `final/` — pass it inline instead of editing the
model list, and give it its own output directory:

```bash
uv run python scripts/run_benchmark.py --benchmarks mldr-it \
  --output-dir outputs/benchmark_scratch --models-only-extra \
  --extra-colbert "round1=outputs/final_round1" "p1 ckpt-5000=outputs/phase1/checkpoint-5000"
```

Completed `(benchmark, model name)` pairs are skipped on re-run, which is what makes
the pipeline resumable — but it keys off the display *name*, not the weights. Reusing
a name for new weights silently skips the model. Use a fresh name or a fresh
`--output-dir`.

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
  still counts. This matters more than it sounds on MLDR-it, where the median
  document is 2666 tokens against a 512-token cap: every document truncates and
  only 20% of the corpus's tokens are otherwise encoded. Chunking is worth +.060
  nDCG@10 (p=.02) to an unchanged checkpoint. Add
  `--colbert-brute-force-limit 70000` so the chunked corpus stays on the exact
  MaxSim path, and expect ~7x the wall clock and ~26 GB of host RAM.
  **Chunked and unchunked numbers are different protocols** — keep them in separate
  `--output-dir`s, and note that chunking currently applies to the ColBERT path
  only, so a chunked ColBERT against a truncated dense baseline is not a fair row.
- **Document truncation is measurable before you spend a GPU on it.**
  `scripts/inspect_lengths.py` reports query/document token percentiles against the
  configured limits, plus how many chunks and how much memory a chunked run would
  need.
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
uv run python scripts/infer_demo.py --model outputs/final_round1 --query "Qual è la capitale d'Italia?"
# or, without training anything locally:
uv run python scripts/infer_demo.py --model enricollen/ItColBERT --query "Qual è la capitale d'Italia?"
```

---

## Related work

- ColBERT / ColBERTv2 (Stanford)
- [PyLate](https://github.com/lightonai/pylate) and [ColBERT-Zero](https://huggingface.co/blog/lightonai/colbert-zero) (LightOn)
- [mxbai-edge-colbert-v0](https://arxiv.org/pdf/2510.14880) — hard-negative mining and data composition as the primary quality drivers
- mMARCO (unicamp-dl), MLDR (BAAI)
- Italian-ModernBERT (DeepMount00), Italian embedding models (nickprock)
