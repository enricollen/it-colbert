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
- Phase 1 dataset: **2,673,467** train / 2000 eval → **5222 steps** at batch 512
  (round 1: 2,487,135 → 4858 steps). Composition: mmarco 1.5M, mmarco-hn 783,591,
  squad-ita 99,955, wiki-hn 100,002, mined 186,332, miracl-ita 5,587.
  Cached at `outputs/dataset_cache/phase1-64432ff02d8bdf14`.
- Round-1 best kept at `outputs/final_round1` for comparison after round 2.

**Round 2 phase 1 timeline**

| when | what |
|---|---|
| 2026-08-15 13:04 | dataset cache + phase 1 start |
| 2026-08-15 22:30 | `stop_at_2230` timer fired, SIGTERM, clean stop |
| 2026-08-16 09:45 | resumed, auto-resume from `outputs/phase1/checkpoint-1750` |
| 2026-08-16 10:14 | at step ~1800 of 5222 (epoch 0.347), loss ~0.19, lr 6.88e-6 |

Throughput ~80 min per 250 steps including IR eval (same as round 1) → ~28h wall
for the full epoch, ~18h left at the 10:14 mark.

**Round 2 phase 1 IR eval, matched against round 1**

In-training `ItIREvaluator` only (150 MLDR queries / 2k docs, 300 mMARCO queries /
3k docs) — not the benchmark, and not comparable to full-corpus numbers.

| step | MLDR r1 → r2 | mMARCO MRR r1 → r2 | `it-ir_score` r1 → r2 |
|---|---|---|---|
| 250 | .369 → .397 | .776 → .808 | .573 → .603 |
| 500 | .453 → .425 | .848 → .868 | .650 → .647 |
| 750 | .450 → .435 | .872 → .884 | .661 → .660 |
| 1000 | .484 → .442 | .886 → .903 | .685 → .673 |
| 1250 | .485 → .430 | .899 → .907 | .692 → .668 |
| 1500 | .481 → .431 | .906 → .913 | .693 → .672 |
| 1750 | .462 → .447 | .907 → .924 | .685 → .685 |

mMARCO up at 7/7 evals, MLDR down at 6/7, composite flat. Interpretation and the
decision gate are in TODO.md §8.2 / §8.3.

**Correction to the §8 mining record:** the command actually run used
`--corpus-docs 200000`, not the 500000 in the runbook (reduced after the WSL OOM).
Also noted 2026-08-16: `mine_hard_negatives.py:83-106` sources both queries and
mining corpus from `load_mmarco_italian`, so the mined set is entirely mMARCO.

---

## 15. Methodology audit + selection/report query split (2026-08-16)

Code audit while round-2 phase 1 was running; eight findings written up as
TODO.md §11. Two touched the numbers already published in §13, so they were
checked rather than assumed.

**Finding: training-time eval read the benchmark test queries.** `ir_eval.py`
builds its MLDR split from `load_mldr_italian(split="test")` and its MIRACL split
from the benchmark's dev split, and `configs/phase2_distill.toml` asked for all
200 MLDR queries. Phase 2 selects on that metric (`load_best_model_at_end` +
early stopping), so round 1 chose its checkpoint on the same queries §13 reports.
Phase 1 never selects, so it was never exposed.

**Checked before fixing — round 1 is clean.** Splitting the 52 existing per-query
files by the new partition, `mldr-it` nDCG@10:

| model | selection half | report half | delta |
|---|---:|---:|---:|
| ItColBERT (the selected one) | .4136 | .3890 | +.0246 |
| field mean over 13 models | — | — | +.0308 |
| Italian-ModernBERT-mmarco-mnrl | .3534 | .2545 | +.0989 |
| BM25 | .4638 | .5047 | −.0409 |

The selection half is easier for nearly every model, selected or not, so the delta
is query difficulty. ItColBERT sits below the field average — no winner's curse.
MIRACL-ita is the control: it was not in round-1's `ir_eval`, so no bias is
possible, and none shows (ItColBERT −.0468 vs field mean −.0199). Round 1 drew
from 5 checkpoints and took the first, so selection pressure was almost nil.
**No published round-1 number changes.** The exposure is prospective: round 2 runs
`eval_steps = 500`, which turns 5 candidates into dozens.

**Implemented.**

- `benchmark/stats.py`: `split_query_ids()` partitions by keyed blake2b hash *of
  the query id*. First cut sliced a shuffled list and failed its own test — a
  subset of a query set split differently from the whole set, which would have let
  a selection query reappear as a reporting query in exactly the tools meant to
  keep them apart. Per-id hashing fixes it; halves land at MLDR 96/104, MIRACL
  379/420, mMARCO 3451/3529, SQuAD 3895/3714.
- `ir_eval.py`: `ItIREvaluator(query_half=…)`, applied before subsampling.
- `config.py` + `configs/phase2_distill.toml`: `ir_eval_query_half`, `"selection"`
  for phase 2, `"all"` for phase 1 (it does not select, and keeping the full set
  leaves round-2 phase-1 curves comparable with round 1). Evaluator warns on
  `"all"`.
- `scripts/report_query_half.py` (new): held-out means + bootstrap CI from the
  per-query files, no rerun needed.
- `compare_models.py --query-half`.
- `save_per_query` now stores `scored_query_ids()` — `per_query_metrics` skips
  queries with no qrels, so the unfiltered id list would misalign. All 52 existing
  files verified aligned; this is a guard, not a repair.

Verified on the dev box: syntax across all 8 edited files, and the partition's
disjointness / coverage / order-preservation / subset-nesting properties against
the real benchmark query ids. `report_query_half.py` end-to-end and the smoke
runs need the GPU box — no torch/numpy env here.

---

## 16. Round 2 completion — phase 1, phase 2, and both decision gates (2026-08-23)

**Phase 1** finished at step 5222 (epoch 1.0), train loss 0.0312, 2.343e4 s
(~6.5h from the last resume). Saved to `outputs/phase1/final` → `outputs/final_phase1`.

**`phase1_bench` ran but produced nothing — caught and fixed manually.** The stage
executed and exited 0, but `results.json` still held round-1's
`ItColBERT (phase1-only)` entries from the earlier run, so the skip-completed check
short-circuited every benchmark before the new checkpoint was ever scored. Not a
crash, not a log warning — just a silent no-op. Fixed by clearing those four
entries and re-running `run_benchmark.py --only "ItColBERT (phase1-only)"` by hand.
Same trap would have hit `phase2_bench`/`bench` for `outputs/final` too, so it was
also handled by hand rather than trusting `run_train.sh`'s automatic re-bench.
Worth a real fix before round 3: the skip-completed check needs to key off which
checkpoint's weights were scored, not just whether the model *name* has an entry.

**§8.3 decision gate (phase-1-only, round 2 vs round 1), full 4-benchmark bench,
GPU:**

| benchmark | round-1 phase1-only | round-2 phase1-only | Δ | significant? |
|---|---|---|---|---|
| MLDR-it nDCG@10 | .3484 | .3011 | −.0473 | no (n=200, round-1 value inside round-2's 95% CI) |
| mMARCO-it MRR@10 | .7784 | .7507 | −.0277 | **yes** (n=6980, round-1 value outside CI) |
| MIRACL-ita nDCG@10 | .6903 | .6867 | −.0036 | no |
| SQuAD-ita nDCG@10 | .9041 | .9037 | −.0004 | no |

Worse than the pre-registered prediction in TODO.md §8.3 (mMARCO up / MLDR down).
Instead mMARCO — the axis the mined negatives are 100% drawn from (§8.1) — moved
significantly *worse*, and nothing else moved outside noise. Mining did not
generalize and mildly hurt its own source distribution.

**Phase 2** (§7 fixes: `contrastive_anchor_enabled=true`, lr 2e-6, `warmup_ratio`
0.01, `eval_steps=500`, MIRACL added to `ir_eval`) ran to completion in one shot:
early-stopped at step 4000, restored best checkpoint from step 2000
(`it-ir_score`=0.7710, patience 4/4). Contrast with round 1's step-2000-was-the-
first-and-best-eval collapse (§5) — the lr/warmup/eval_steps fixes did remove that
specific failure mode; there's a real curve here, not an accidental first pick.

**§8.4 (full round-2-vs-round-1 comparison, `outputs/final` vs `outputs/final_round1`,
paired bootstrap via `compare_models.py`):**

| benchmark | round-1 final | round-2 final | Δ | p | verdict |
|---|---|---|---|---|---|
| MLDR-it nDCG@10 | .4008 | .3779 | −.0229 | .082 | not significant |
| mMARCO-it MRR@10 | .7196 | .7297 | +.0100 | <.0001 | **significant, up** |
| MIRACL-ita nDCG@10 | .7194 | .7091 | −.0103 | .026 | **significant, down** |
| SQuAD-ita nDCG@10 | .9026 | .8987 | −.0039 | .011 | **significant, down** |

Phase 2's contrastive anchor (also 100% mMARCO, §11.5) pulled the final model's
mMARCO score back above round 1's, overcorrecting past phase 1's own mMARCO loss —
but MLDR, the actual target, still doesn't move outside noise, and now misses the
§6 `>0.40` target that round 1 had just cleared. MIRACL and SQuAD both moved down
with p<.05 (SQuAD's −.004 is real but tiny given n=7609).

**Decision: round 2 does not replace round 1.** `outputs/final_round1` stays the
release candidate. Confirms §11.5's prediction, made before round 2 finished phase
1, that both round-2 levers (mined HN, phase-2 anchor) draw from mMARCO only and
would move that axis without touching the out-of-domain one. Next round should not
repeat either lever on the same source — see TODO.md §10.5 (cheap, do first) and
§10.6 (a real long-document/wiki Italian source, promoted to next-in-line by this
result).

**Process note:** round-1's per-query files (`ItColBERT`, `ItColBERT (phase1-only)`)
would have been silently overwritten by round 2's benchmark runs before this
comparison could happen. Backed them up to `*_round1.json` in
`outputs/benchmark/per_query/` first, then reconstructed a `ItColBERT (round1)`
entry in `results.json` (mean + bootstrap CI recomputed straight from the raw
per-query values) so the comparisons above are genuinely paired, not a diff of two
independently-computed point estimates. Do this backup step *before* clearing any
`results.json` entries in future rounds — there's no other copy of round 1's
per-query data.

---

## 17. The no-training audit — the length ceiling was the whole story (2026-08-24)

After round 2 came back negative, the question was what to try next given mined hard
negatives and the phase-2 fixes had both failed to move MLDR-it. Instead of picking
another training lever, ran the three cheap never-executed items in TODO.md §10.5,
§11.2 and §11.4 first. All three resolved in one evening, no training, and the
answer to "what should round 3 do" changed completely.

### Noise floor first (§11.4)

Benchmarked the round-2 phase-1 decay tail — `checkpoint-4750`, `-5000`, `-5222`,
which were still on disk — on the full MLDR-it suite: **.2982 / .2992 / .3011,
range .0030, monotone.** The .4403-.4507 spread this section worried about came from
the 150-query in-training slice and was slice noise, not run instability. Gives a
floor to judge everything else against: round 2's −.0473 phase-1 regression is 16x
it, so those regressions were real.

Needed a way to score arbitrary checkpoints, so `run_benchmark.py` gained
`--extra-colbert "NAME=PATH"` and `--models-only-extra`. Beats editing
`DEFAULT_MODELS` for every one-off comparison, which is how round 2's stale-entry
trap happened in the first place (§16).

### Query length: a real truncation not worth fixing (§10.5)

New `scripts/inspect_lengths.py` (tokenizer only, ~20s, no gpu). MLDR-it truncates
**12.5% of queries** at `query_length=32`, p95 46 tokens, max 445; the other three
benchmarks are clean (≤0.6%). So the suspicion was right.

But raising it loses, monotonically: MLDR nDCG@10 .4008 → .3764 (q64) → .3053 (q96)
on `outputs/final_round1`. Splitting by query separated the two effects — the 25
truncated queries gain (+.031 at q64), the 175 untouched ones lose (−.032) — because
ColBERT pads queries to `query_length` with `[MASK]` and the count is trained in.
So the fix can only happen in training, and the ceiling is +.031 on 12.5% of
queries ≈ **+.004 overall, inside the noise floor**, resting on 2 queries that
actually changed rank. Closed as a non-lever.

### The document side, which was the real problem (§11.2)

Same tokenizer pass on documents: MLDR-it median **2666 tokens** against a 512 cap.
**100% of documents truncate and 20.2% of the corpus's tokens are all any neural
model has ever seen.** mMARCO 0%, MIRACL 1.4%, SQuAD 0.7% — MLDR-it is the only
benchmark with this problem, which is why two rounds of negative-mining never
touched it.

Chunked `outputs/final_round1` at 2000 chars / 200 overlap, max-pooled per document
(code that had existed unused since the harness was written, `chunk_chars` default 0):

| protocol | doc chars | BM25 | trunc-512 | chunked |
|---|---|---|---|---|
| as reported in §6 | 12000 vs ~2200 | .4850 | .4008 | — |
| length-symmetric | 4000 all | .4487 | .4008 | .4384 |
| full coverage | 12000 all | .4850 | .4008 | **.4610** |

**+.0602 nDCG@10, p=.0225, paired — 20x the noise floor, and larger than every
training lever from both rounds combined.** MRR@10 +.0558 (p=.040), recall@100
.6850 → .7550, tying BM25 exactly. Against BM25 the lead becomes −.0241 (p=.249) at
full coverage and −.0103 (p=.628) at symmetric 4000, so **"loses to BM25 on the only
clean out-of-domain benchmark" is a protocol artifact** — corrected in TODO.md §6.

Sanity check that fell out for free: dropping the corpus from 12000 to 4000 chars
costs BM25 .0363 and costs the truncated ColBERT exactly nothing (.4008 either way,
since it never read past ~2200 chars).

### What it took to run

First full-coverage attempt died. 64780 chunks → past the 25000 brute-force
threshold → ANN path → `fast_plaid`'s native extension fails to load against the
installed torch (`undefined symbol: _ZN3c106detail14torchCheckFail...`), silent
fallback to Voyager, which holds a second full copy: killed it at 24.1 GB RSS on a
27 GB box. **That fallback has been silently in effect for every 100k-doc mmarco run
to date** — worth its own fix.

Two changes made it fit:

- `maxsim_topk(..., consume=True)` frees each source embedding as it copies it into
  the padded tensors, so the brute-force path holds one copy instead of two. Guarded
  to lists only (a stacked ndarray can't be freed piecewise) and off by default, so
  `ir_eval`'s training-time calls are untouched. Verified numerically inert: the
  unchunked run reproduces **.400773 to six decimals** with and without it.
- `--colbert-brute-force-limit` to keep a chunked corpus on the exact path rather
  than the heavier, broken ANN one.

Also added `--mldr-max-doc-chars`, which is what makes the length-symmetric row
possible: `load_mldr_italian`'s 12000-char cap applies to BM25 too, so it was the
only knob that could equalize document access. It had been a silent loader default.

Even so, full coverage peaks at **26 of 27 GB** and takes 999s vs 148s truncated
(6.7x) on a 10000-document corpus — first real efficiency data in the project, now
recorded in TODO.md §11.7. Chunked mmarco at 100k docs is not runnable on this box.

### Conclusion

Round 3 is a document-length problem, not a data problem: raise `document_length`
past 512 (ModernBERT's window is 8192, no run has used more than 512), cheapest
decisive step being phase 2 alone at 1024 on the existing `final_phase1` (~1.8h).
Chunking proves the information in tokens 512-3000 is worth +.06 to a model never
trained to use it, and training at length avoids chunking's 6.7x inference cost.
Full plan in TODO.md §12, including what not to do — no re-mining, no query-length
change, no more mMARCO.

Caveat that gates publishing any of it: `chunk_chars` applies to the ColBERT path
only, so the other four late-interaction models and every dense baseline are still
truncated. The old asymmetry favoured BM25; this one would favour us. The field has
to be re-run chunked before .4610 appears next to anyone else's number (TODO.md
§12.1).

---

## 18. Finding a long-document Italian source, and building the ablation (2026-08-24)

§17 ended with "round 3 is a document-length problem". This is the follow-on: find
data that actually has long documents, then build an experiment that can tell
whether training at length helps. No GPU time spent yet. Full spec in TODO.md §13.

### ReDiX was a dead end, and not for the reason expected

`ReDiX/wikipediaQA-ita` had been sitting in TODO.md §10.4/§10.6 as "gated, schema
unverified". It looked blocked: the card demands an email to `redix.ai@redix.it`,
and an earlier `load_dataset` raised `DatasetNotFoundError: is a gated dataset`.

Both assumptions were wrong. The repo is `gated: auto`, and the token already in
`~/.cache/huggingface/token` reads it — `hf_hub_download` of `README.md` succeeded,
then a ranged GET pulled the data file directly. No access request was ever needed.

What killed it was the content. Sampling 4,724 rows across ten windows of the full
237 MB file (`usedStorage` reports 788 MB; the live file is 237 MB) and tokenizing
with the round-1 tokenizer: schema is `question` / `context` / `answer` with no
negatives, and `context` is median **439 tokens**, p90 489, max 1,085 — only 4.6%
over 512. 1,613 unique contexts across those 4,724 rows, so it is ~3 questions per
pre-chunked passage. It exercises nothing past the current cap. Ruled out.

### `it-long_doc`, and why it is the right source

Searching the Hub for Italian retrieval sets surfaced
`hotchpotch/wikipedia-multilingual-synthetic-ir-query`, which ships a per-language
`long_doc` config. Verified over all 9 Italian shards (650,885 rows):

| quantity | value |
|---|---|
| document length | median 4,133 chars (~1,033 tok), p75 ~1,779 tok, p90 ~3,257 tok |
| over 512 / 1,024 tokens | **98.2%** / 50.5% |
| evidence span starts past 512 / 1,024 tok | **30.7%** / 14.2% |

The last row is the one that matters and it comes from the dataset's own
`query_source_text_start` column, which records the span the synthetic query was
generated from. At `document_length = 512`, 30.7% of these rows train the model to
match a query against a document whose evidence is not in the window. At 1024,
14.2%. That gap is a measurable lever, not a hypothesis.

Ungated, Parquet, CC-BY-SA-4.0/GFDL, revision pinned. Crucially **not** MLDR, so
MLDR-it survives as the held-out benchmark — unlike `Shitao/MLDR` it-train, the
other long candidate, which is clean and long (median 5,785 tok, zero query overlap
with test) but in-domain against the benchmark and only 2,151 queries.

Two honest caveats, both going in the model card: the dataset card warns its queries
are "relatively easy" and recommends mixing rather than sole use, and both it and
MLDR-it come from Wikipedia, so article overlap is plausible and has to be measured.

### The overlap check

`scripts/check_longdoc_overlap.py` does content-defined word-shingle containment:
13-word n-grams, kept only where the first word hashes to 0 mod 16. The mod-p
selection is the point — it depends on content, not offset, so the same passage
yields the same shingles on both sides even when the surrounding article is cut
differently. Naive fixed-offset sampling would have missed almost everything.

Cost control matters here: 650k rows at ~690 words each is ~450M positions. Two
things make it tractable — articles are deduplicated by a text-derived key first
(rows repeat articles across query styles), and per-word crc32 is memoized, so the
inner loop is dict lookups and only 1/16 of positions build an n-gram string.

Validated on a synthetic pair before running: the same article written with
markdown headings and different punctuation scored 5 hits, an unrelated article 0.

`longdoc_normalize` / `longdoc_doc_key` live in `data.py` and are imported by the
script, so the keys it writes are exactly the keys the loader filters on. Getting
that wrong would silently exclude nothing.

### The ablation, and why both arms get trained

`configs/phase1_longdoc_512.toml` and `configs/phase1_longdoc_1024.toml` differ in
exactly two values: `document_length` (512/1024) and `mini_batch_size` (32/16).
`per_device_train_batch_size` stays 512 in both so the in-batch negative count is
identical — GradCache is what allows peak VRAM to move without touching the loss.

Mixture is deliberately small and exactly two sources of known size: 250k mMARCO +
250k `it-long_doc` = 500k triplets, ~980 steps. mMARCO hard negatives, MIRACL/SQuAD
and wiki-hn are all off. The question is not how good phase 1 can get; it is whether
reading past 512 tokens helps at all, and both arms pay the same price to answer it.

Training both arms rather than reusing a checkpoint is forced by §12.4: round 2
overwrote `outputs/final_phase1`, so there is no phase-1-only 512 control on disk.

Gate: arm B must beat arm A on the MLDR-it **report half** by more than the .0030
noise floor, with mMARCO-it and MIRACL-ita not dropping by more than the same. Both
arms are phase-1-only on a fifth of round 1's data, so neither is comparable to
.4008 or .4610 — the comparison is strictly A-vs-B.

### Two operational notes

`hf download` stalls on Xet: seven blob files sat at 0 bytes indefinitely, no error.
`HF_HUB_DISABLE_XET=1` fixes it and transfers at ~7 MB/s. Worth remembering for
every Hub download on this box.

`hf datasets sql` needs DuckDB, which is not installed. PyArrow over `HfFileSystem`
reads Parquet footers and single columns with range requests instead, which is how
the 650k-row distribution was measured without downloading 62 GB — though at ~13
minutes for two int columns it is slow enough that bulk work should download first.
