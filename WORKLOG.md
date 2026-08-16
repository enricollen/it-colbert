# Italian ColBERT — work log

Chronological record of training, benchmarking, and ablations for the monolingual Italian late-interaction retriever (`DeepMount00/Italian-ModernBERT-base` + [PyLate](https://github.com/lightonai/pylate)).

---

## 1. Project setup

- Built two-phase ColBERT training pipeline (ColBERT-Zero / GTE-ModernColBERT style, **without** unsupervised mega-pretrain).
- **Phase 1:** `CachedContrastive` on `unicamp-dl/mmarco` (Italian) + `nickprock/it-wiki-retrieval-synthetic-hn`.
- **Phase 2:** `Distillation` (KL) on `lightonai/embeddings-fine-tuning-filtered-it` using cross-encoder `rerank_scores`.
- Stack: PyLate, Sentence Transformers, FAISS / Voyager, ranx metrics.
- Hardware: NVIDIA RTX 3090 24GB (WSL2).

---

## 2. Initial training (doc_len=180)

| Phase | Config highlights | Runtime | Output |
|---|---|---|---|
| Phase 1 contrastive | 400k mmarco + wiki HN (≤2), batch 128, lr 3e-6, `document_length=180` | ~1h40 | `outputs/phase1/final` |
| Phase 2 KD | **80k** LightOn IT samples, batch 16, lr 1e-5, `document_length=180` | ~1h01 | `outputs/phase2/final` → `outputs/final` |

This `outputs/final` checkpoint remained the **best** model after later ablations.

---

## 3. IR benchmark framework

Implemented comparable eval on:

- **MLDR-it** — full Italian corpus (~10k docs, 200 test queries); primary metric **nDCG@10**.
- **mMARCO-it** — pooled **100k** docs (qrel positives + reservoir sample), 6980 queries; primary metric **MRR@10**.

**Protocol notes**

- Pooled mMARCO MRR is **inflated** vs published full 8.8M-corpus numbers → use for **relative** ranking only.
- MIRACL has **no Italian split** → not used.
- Metrics: MRR@10, nDCG@{1,5,10,100}, MAP@100, Recall@{1,5,10,100}, Precision@10, Hits@{1,5,10}.
- Artifacts: `outputs/benchmark/results.json`, canvas report.

**Baselines evaluated**

| Model | Kind |
|---|---|
| BM25 | lexical |
| Italian-ModernBERT-base (mean-pool) | dense (no IR FT) |
| Italian-ModernBERT-mmarco-mnrl | dense |
| multilingual-e5-base / e5-large | dense |
| bge-m3 | dense |
| Ita-Search | dense |
| jina-colbert-v2 | ColBERT |
| mLateOn | ColBERT (`lightonai/mLateOn`) |
| ColBERT-XM | ColBERT (`antoinelouis/colbert-xm`, lang=`it_IT`) |
| italian-colbert (ours) | ColBERT |

**Infra fixes along the way**

- HF downloads flaky on WSL → resumable `wget` weight fetch + hub linking.
- jina load: `trust_remote_code=True` + install `einops`.
- FastPLAID ABI mismatch → Voyager CPU ANN fallback for large corpora.
- CUDA teardown hang after bench → `os._exit(0)` + stronger `clear_cuda`.
- MLDR ColBERT: exact MaxSim bruteforce for ≤25k docs (Voyager segfault on long multi-vectors).

---

## 4. Final benchmark scores (pooled protocol)

### MLDR-it (nDCG@10)

| Model | nDCG@10 | MRR@10 | R@100 |
|---|---:|---:|---:|
| **bge-m3** | **0.453** | 0.422 | 0.710 |
| multilingual-e5-large | 0.431 | 0.397 | 0.715 |
| multilingual-e5-base | 0.429 | 0.400 | 0.655 |
| jina-colbert-v2 | 0.369 | 0.330 | 0.640 |
| BM25 | 0.332 | 0.294 | 0.590 |
| Italian-ModernBERT-mmarco-mnrl | 0.302 | 0.283 | 0.515 |
| **italian-colbert (ours, train180 infer512)** | **0.277** | 0.247 | 0.550 |
| italian-colbert (doc256 train+infer) | 0.227 | 0.200 | 0.495 |
| italian-colbert (train180 infer180) | 0.216 | 0.188 | 0.465 |
| Ita-Search | 0.189 | 0.166 | 0.425 |
| Italian-ModernBERT-base mean-pool | ~0.004 | ~0.002 | 0.125 |

### mMARCO-it pooled 100k (MRR@10)

| Model | MRR@10 | nDCG@10 | R@100 |
|---|---:|---:|---:|
| **jina-colbert-v2** | **0.849** | 0.871 | 0.981 |
| multilingual-e5-large | 0.824 | 0.850 | 0.978 |
| multilingual-e5-base | 0.791 | 0.820 | 0.971 |
| bge-m3 | 0.781 | 0.811 | 0.969 |
| Ita-Search | 0.565 | 0.602 | 0.885 |
| Italian-ModernBERT-mmarco-mnrl | 0.508 | 0.541 | 0.802 |
| **italian-colbert (ours)** | **0.491** | 0.517 | 0.699 |
| italian-colbert (doc256) | 0.411 | 0.432 | 0.599 |
| Italian-ModernBERT-base mean-pool | 0.017 | 0.020 | 0.086 |

Published full-corpus mMARCO-it MRR@10 (literature, not comparable to pooled run): jina 0.337, mono-mT5 0.303, mColBERT 0.292, mE5-base 0.280, ColBERT-XM 0.265, BM25 0.153.

---

## 5. Ablation: retrain at `document_length=256`

**Hypothesis:** train/infer length mismatch was limiting MLDR.

**Experiment:** same data budget (400k contrastive + 80k KD), only `document_length` 180→256. Separate outputs (`outputs/phase1_doc256`, `outputs/phase2_doc256`, `outputs/final_doc256`). Old `outputs/final` kept.

**Result: regression**

| Checkpoint | MLDR nDCG@10 | mMARCO MRR@10 |
|---|---:|---:|
| ours train180 + infer512 | **0.277** | **0.491** |
| doc256 train+infer | 0.227 | 0.411 |
| train180 + infer180 | 0.216 | — |

**Conclusion:** length alone is not the lever. Keep **train@180**; use longer infer only when useful for long docs.

---

## 6. Diagnosis — why scores lag jina / dense SOTA

1. **KD under-scaled:** used 80k of ~**1.56M** available LightOn Italian KD rows (~5%).
2. **Init from LM**, not a strong Italian dense embedder (GTE-ModernColBERT-style path).
3. **No ColBERTv2 hard-neg refresh** (self-mine → strong CE soft labels → distill).
4. Backbone/data scale vs jina (560M, huge pair pretrain, multilingual CE teacher).

---

## 7. Improvement roadmap

| # | Step | Status |
|---|---|---|
| 1 | **Scale KD** from `outputs/phase1/final`, doc180, full LightOn IT (~1.56M) | done (fullkd) |
| 1b | **Mix KD** LightOn + mMARCO HN (option B) | **paused @ ckpt-2000; resume next** |
| 1c | **Gemma single-teacher rescore** of KD pairs → new IT KD dataset + retrain | **after mixkd results** |
| 2 | Stronger init from Italian dense embedder, then contrastive → KD | pending |
| 3 | ColBERTv2 hard-neg loop with strong multilingual CE teacher | pending |
| 4 | More / harder contrastive negatives (wiki + self-mined) | pending |
| 5 | System: hybrid BM25+ColBERT and/or CE rerank@100 | pending |
| 6 | Bench add **mLateOn** + **ColBERT-XM** (after fullkd) | pending |
| 7 | Checkpoint merge mixkd ↔ fullkd (cheap dual-bench fix) | **running: 80k⊕fullkd first** |

---

## 8. Step 1 details (current run)

- **Start:** `outputs/phase1/final` (pre-KD contrastive checkpoint).
- **Train:** phase2 Distillation only, `document_length=180`, **all** KD splits (~1.56M examples).
- **Batch:** 16 (same as successful 80k run).
- **Outputs:** `outputs/phase2_fullkd/final` → `outputs/final_fullkd`.
- **Then:** bench only `italian-colbert (fullkd)` on MLDR-it + pooled mMARCO; baselines unchanged.
- **ETA:** ~20h wall time at ~1.3–1.4 it/s (≈97k steps), then ~30–40 min bench.

Configs / script:

- `configs/phase2_distill_fullkd.toml`
- `scripts/run_phase2_fullkd.sh`
- Logs: `outputs/logs/phase2_fullkd.log`, `outputs/logs/fullkd_nohup.log`

---

## 9. Phase 2 validation (KD overfitting check)

**Why:** KD can overfit teacher soft labels / KD-set quirks even while train KL drops. Need a validation signal without full retrieval cost.

**Chosen approach (option 1):** hold out a slice of **LightOn Italian KD** never used in train; monitor **KL(student ‖ teacher)** with PyLate `ColBERTDistillationEvaluator`.

Rejected: mMARCO-train triplet accuracy — overlaps phase 1 and LightOn `msmarco_it`, so it is not a clean held-out.

**What we added:**

- `load_kd_italian_train_eval()` splits train / hold-out before training.
- Default: **512** KD hold-out examples every **2000** steps.
- **Early stopping:** patience **3** evals on `kd-holdout_kl_divergence` (lower is better).
- **`load_best_model_at_end`:** `final/` is the best val-KL weights, not always the last step.
- Config: `kd_eval_samples`, `eval_steps`, `early_stopping_patience`, `load_best_model_at_end`; set `kd_eval_samples = 0` to disable.
- Code: `data.py`, `train_phase2.py`, `Phase2Config`, `EarlyStoppingCallback`.

**Mid-run enable (2026-08-09):** resumed from `checkpoint-80000` with KD KL val + early stopping + load-best. Because resume must keep the original train size/`max_steps`, eval is a **512-example monitor set that may overlap train** for the remainder of this job. Fresh from-scratch phase2 runs still use a true hold-out (`exclude_from_train=True`).

**Fullkd outcome:** best val-KL @ step **92000** → MLDR nDCG@10 **0.352** (↑ from 0.277) but pooled mMARCO MRR@10 **0.341** (↓ from 0.491). LightOn-only full KD helps long-doc, hurts short-passage.

---

## 11. Mix KD (option B) — mMARCO HN already scored

**Problem:** no native mMARCO-it KD set with soft labels; building one means mining + CE scoring.

**Option B:** reuse `hotchpotch/mmarco-hard-negatives-reranker-filtered` / `italian-hard-negatives` (CE = `bge-reranker-v2-m3`, fields `pos_score`/`negs_score`). Convert to PyLate KD rows and **mix** (shuffle) with LightOn (400k cap) so phase2 sees both distributions in the same batches.

### Score scales (important)

- **LightOn** `rerank_scores` (mxbai): already logits-like → **no conversion**.
- **mMARCO HN** `pos_score`/`negs_score` (bge): sigmoid-like probs in [0,1] → convert with `logit(p)=log(p/(1-p))` in `load_mmarco_hn_kd_italian`, because PyLate/ColBERTv2 KD do `log_softmax` on teacher labels and expect logits.
- After fix, mixkd train loss starts ~**1.3** (same ballpark as fullkd ~1.16); with raw probs it wrongly started ~0.43.

Also pad/truncate each HN row to **11** documents (n_ways) so the collator can stack batches with LightOn.

### Not the same dataset as phase 1 mMARCO

- Phase 1: `unicamp-dl/mmarco` Italian **official train triples** `(query, positive, negative)` → contrastive.
- Mixkd: **hotchpotch** HN pack derived from mMARCO, multi-negative + CE soft scores → distillation.
- Same train domain, different artifact / supervision. Dev benchmark unchanged.

### Run

- Start: `outputs/phase1/final` (fresh; do not resume pre-logit checkpoints)
- Config: `configs/phase2_distill_mixkd.toml`
- Script: `scripts/run_phase2_mixkd.sh` (auto-resume only `checkpoint-[0-9]*`)
- Out: `outputs/final_mixkd` + bench `italian-colbert (mixkd)`
- Code: `load_mmarco_hn_kd_italian`, `include_mmarco_hn` on `Phase2Config` / `load_kd_italian_train_eval`
- Docs: README “Phase 2 mixkd”

**ETA:** ~770k examples @ batch 16 ≈ 48k steps ≈ **8–12h** + bench, with early stop on hold-out KL.

**Paused 2026-08-09:** stopped cleanly at `outputs/phase2_mixkd/checkpoint-2000` for overnight. Resume: `bash scripts/run_phase2_mixkd.sh` (auto-resume).

---

## 11c. Checkpoint merge (80k ⊕ fullkd) — no mixkd required

**Idea (JaColBERTv2.5):** average parameters of complementary specialists to reduce forgetting without retraining.

- Inputs: `outputs/final` (80k KD, strong mMARCO) + `outputs/final_fullkd` (strong MLDR), weights **0.5 / 0.5**
- Script: `scripts/merge_checkpoints.py` (averages backbone + `1_Dense` safetensors)
- Out: `outputs/final_merge_80k_fullkd`
- Bench name: `italian-colbert (merge-80k-fullkd)` (infer `document_length=512`)

```bash
uv run python scripts/merge_checkpoints.py \
  --inputs outputs/final outputs/final_fullkd \
  --output outputs/final_merge_80k_fullkd --weights 0.5 0.5
uv run python -u scripts/run_benchmark.py \
  --benchmarks mldr-it mmarco-it --mmarco-max-corpus-docs 100000 \
  --only "italian-colbert (merge-80k-fullkd)" --top-k 100 \
  --output-dir outputs/benchmark
```

**Pooled results (2026-08-10):**

| Model | MLDR nDCG@10 | mMARCO MRR@10 |
|---|---:|---:|
| `final` (80k KD) | 0.277 | **0.491** |
| `final_fullkd` | **0.352** | 0.341 |
| **merge 0.5/0.5** | 0.302 | 0.442 |

Merge sits **between** the two specialists on both benches — no Pareto win. Softens fullkd’s mMARCO collapse and lifts 80k’s MLDR, but loses to each parent on its strong axis. Next: try unequal weights (e.g. 0.7×80k + 0.3×fullkd) and/or wait for mixkd; Gemma rescore still deferred.

---

## 11b. Planned after mixkd: Gemma teacher rescore (new IT KD dataset)

**Decision:** wait for mixkd bench; if teacher mismatch / soft-label quality still looks limiting, rescore KD **candidate lists** (not corpora) with one stronger teacher.

- Teacher: `BAAI/bge-reranker-v2-gemma` (better multilingual than jina; slower).
- Scope: existing LightOn-IT + mMARCO-HN `(query, ~11 docs)` pairs only — **not** full mMARCO corpus, **not** MLDR.
- Artifact: publishable Italian KD dataset with unified Gemma scores (~order of 8M CE pair inferences; ~1.5–4 days on RTX 3090).
- Then: phase2 from `phase1/final` on the new scores + bench.

Cheaper alternatives if mixkd is already strong: checkpoint-merge mixkd↔fullkd; or gemma-rescore a subset only.

---

## 12. Key paths

```
outputs/final/                 # trained model (phase 2) — target for bench
outputs/final_phase1/          # phase1-only ablation copy for bench
outputs/phase1/                # contrastive training checkpoints
outputs/phase2/                # kd training checkpoints
outputs/benchmark/results.json # all IR scores
outputs/logs/                  # train / bench logs
outputs_v1_archive/            # deleted 2026-08-11 after archive step (was old v1 runs)
```

Canvas: `~/.cursor/projects/home-enricollen/canvases/italian-colbert-benchmark.canvas.tsx`

---

## 13. Rewrite runbook (TODO.md) — 2026-08-11

**Setup (done)**

- §0: archived old `outputs/` → `outputs_v1_archive/` (later deleted to reclaim disk).
- §1 smoke: both phases log `it-ir_score`; evaluator wiring OK.
- §2 `inspect_kd_scores`: all LightOn splits store **11 docs/query** → `kd_n_ways = 11` confirmed.
- §3 baselines: four suites, 512-token colbert, italian BM25, bootstrap CI.

**Baseline benchmark (§3, completed 2026-08-11 ~09:47)**

- 44/52 runs OK; 8 expected errors (`ItColBERT` ×2 on each suite — no checkpoint yet).
- ENOSPC crash overnight: truncated `results.json`; repaired from completed entries;
  deleted finished colbert indexes (~54G); added atomic `results.json` writes + index
  cleanup after each colbert model in `benchmark/run.py`.
- Headline MLDR nDCG@10 (full ~10k corpus): BM25 0.485, mLateOn 0.462, bge-m3 0.453,
  jina 0.386, SauerkrautLM 0.312, mnrl init 0.302.
- Headline mMARCO pooled MRR@10 (100k docs, rank-only): jina 0.839, mLateOn 0.821,
  e5-large 0.824, e5-base 0.791, bge-m3 0.781, SauerkrautLM 0.534, BM25 0.572.
- MIRACL-ita / squad-ita: jina and mLateOn lead dense/colbert rows; see `results.json`.

**Training (§4–5, started 2026-08-11)**

```bash
nohup bash scripts/run_train.sh > outputs/logs/train_nohup.log 2>&1 &
tail -f outputs/logs/phase1.log
```

Recipe: mnrl init, phase1 contrastive (512 batch / 32 mini, lr 1e-5, ~2.4M samples),
phase1-only bench, phase2 single-teacher KD (proportional splits), final ItColBERT bench.
Interrupt-safe: re-run the same command to resume.

**Round 1 training complete (2026-08-14)**

| stage | notes |
|---|---|
| Phase 1 | ~4858 steps, 1 epoch; resumed from ckpt-2500; `outputs/phase1/final` |
| Phase 1-only bench | MLDR nDCG@10 **0.348**, mMARCO pooled MRR@10 **0.748** |
| Phase 2 KD | batch 8 OOM → batch 4; early stopped step 10000; best ckpt **step 2000** (`it-ir_score` 0.689) |
| Final | `outputs/phase2/checkpoint-2000` → `outputs/final` |

**Final ItColBERT benchmark (512-token ColBERT, four suites)**

| model | MLDR nDCG@10 | mMARCO MRR@10 | miracl nDCG@10 | squad nDCG@10 |
|---|---:|---:|---:|---:|
| **ItColBERT (final)** | **0.401** | 0.720 | — | — |
| ItColBERT (phase1-only) | 0.348 | **0.748** | — | — |
| mLateOn | 0.462 | 0.821 | — | — |
| BM25 | 0.485 | 0.572 | — | — |
| bge-m3 | 0.453 | 0.781 | — | — |
| SauerkrautLM | 0.312 | 0.534 | — | — |

MLDR target (>0.40) met. KD lifted MLDR vs phase1-only (+0.053, p=0) but hurt pooled mMARCO (−0.029 MRR, p=0) — §7 tradeoff.

**§6 significance (baseline ItColBERT, `compare_models.py --all`)**

Publishable claims (all p=0 unless noted):

- Beats **SauerkrautLM** on all four suites (MLDR +0.089 nDCG, mMARCO +0.186 MRR, miracl +0.120, squad +0.069).
- Beats **mnrl init** everywhere; beats **mean-pool base** everywhere by large margins.
- **Behind** mLateOn, bge-m3, jina on MLDR and mMARCO (all significant).
- **Ahead of** BM25 on mMARCO/miracl/squad; **behind** BM25 on MLDR (−0.084 nDCG, p=0.006).
- vs **jina** on MLDR: +0.015 nDCG — **not significant** (p≈0.51).
- vs **multilingual-e5-base** on miracl: −0.018 nDCG — **not significant** (p≈0.06).
- Final vs phase1-only: MLDR +0.052 (sig), mMARCO −0.029 (sig), squad ≈ tie (not sig).

Full log: `outputs/logs/compare_models.log`.

**§8 mining (2026-08-14 → 2026-08-14 evening)**

Target: mine HN from round-1 `outputs/final` before retrain (inference-only; safe
to run ahead of §7).

```bash
uv run python scripts/mine_hard_negatives.py \
  --model outputs/final --output outputs/mined_hn \
  --queries 50000 --corpus-docs 200000 --negatives 8 --skip-top 5 \
  --query-chunk-size 500 --retrieve-batch-size 5
```

OOM on 27GB WSL during Voyager retrieval (16GB `index.voyager` + model + rerank
spike). Fixes landed in `scripts/mine_hard_negatives.py` + `retrievers.py`:

- Persist corpus/manifest under `outputs/mining_state/`; Voyager index under
  `outputs/mining_index/` (reuse without re-encode).
- Per-chunk checkpoints `chunks/chunk_NNNNN.json`; resume from last finished chunk.
- Each retrieval chunk runs in a **subprocess** so RAM is freed between chunks.
- `--retrieve-batch-size 5` (pylate default 50 OOMs).

Result: **46,583** rows → `outputs/mined_hn` (avg 8.0 negatives/query). 94/94
chunks. Disk cleanup earlier: deleted unused phase1/phase2 mid-checkpoints and
`phase2_feasibility/` (~13GB); kept `final`, `final_phase1`, `checkpoint-2000`.

**§7 phase-2 fixes + smoke (2026-08-15)**

Config (`configs/phase2_distill.toml`) before retrain:

- `learning_rate = 2e-6`, `warmup_ratio = 0.01`
- `eval_steps` / `save_steps` = 500
- MIRACL in ir_eval; `ir_eval_weights = [0.35, 0.35, 0.30]`
- `contrastive_anchor_enabled` left false until smoke

§7.1 smoke (`configs/smoke.toml` already had `contrastive_anchor_enabled = true`):

```bash
uv run python scripts/train_phase1_contrastive.py --smoke
uv run python scripts/train_phase2_distill.py --smoke
```

Passed: log shows `phase2 contrastive anchor on: 512 replay triplets alongside 253
kd rows`; MIRACL metrics in ir_eval; no OOM. Then set
`contrastive_anchor_enabled = true` in `phase2_distill.toml`.

**§8 round 2 retrain (started 2026-08-15 ~13:04)**

```bash
mv outputs/final outputs/final_round1
rm -rf outputs/phase1 outputs/phase2
# mined_negatives_path = "outputs/mined_hn" uncommented in phase1_contrastive.toml
nohup bash scripts/run_train.sh > outputs/logs/round2_train.log 2>&1 &
```

- Mined HN loaded: **186,332** triplets (4 per query from 46k mined rows).
- Phase 1 running (~5.2k steps/epoch). @step 250: MLDR nDCG@10 **0.397**,
  mMARCO MRR@10 **0.808**, `it-ir_score` **0.603**; loss ~40 → ~0.4 early.
- Round-1 best kept at `outputs/final_round1` for comparison after round 2.
