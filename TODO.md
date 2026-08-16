# Runbook

Goal: an Italian-specialized late-interaction retriever at base size, with
reproducible Italian evaluation. Not "the first Italian ColBERT" (that claim no
longer holds, see §0.5), and not "beats mE5-large everywhere", which is not
realistic on one 3090.

Hardware: one RTX 3090 (24 GB), Linux, `uv`.

Order matters. §1 and §2 are cheap and make the later numbers trustworthy. §3
populates the comparison table before any GPU-days are spent. §4 to §6 produce and
measure the model. §7 to §9 cover the follow-up work.

This file is self-contained: start at the top and work down.

---

## Start here — first pull on the training machine

```bash
cd ~/projects/it-colbert
git pull
```

**1. Archive the old `outputs/` before anything else.** Config and output paths
were renamed (the `_v2` suffix is gone now that there is no v1), so
`outputs/phase1`, `outputs/phase2` and `outputs/final` now refer to the new
recipe. On this machine those same paths still hold the old checkpoints.
`outputs/` is gitignored, so nothing warns about it, and `ModelSpec("ItColBERT")`
points at `outputs/final`. Left in place, the benchmark will index a superseded v1
checkpoint and label it as the new model.

```bash
mv outputs outputs_v1_archive     # keep until the new run has numbers, then delete
mkdir -p outputs/logs
```

**2. Sync dependencies.** `pyproject.toml` gained `snowballstemmer` (Italian BM25
analyzer) and an optional `mteb` extra. If the lockfile is stale, `uv sync`
regenerates it; commit the result.

```bash
uv sync
```

**3. Confirm every new module arrived.** These are imported at module load, so a
missing one fails at run time, not at pull time:

```bash
ls src/it_colbert/checkpoints.py src/it_colbert/callbacks.py \
   src/it_colbert/ir_eval.py src/it_colbert/benchmark/stats.py \
   scripts/compare_models.py scripts/mine_hard_negatives.py scripts/run_mteb_italian.py
```

All seven must exist. `checkpoints.py` and `callbacks.py` are imported by both
training phases, `stats.py` by the benchmark. Then read "Stopping and resuming"
below, and go to §1.

### Riskiest untested pieces

Nothing in this repo had been executed since the rewrite; it was validated by
parsing and config loading only. In likely-to-break order:

1. `ItIREvaluator` metric-name plumbing through sentence-transformers.
   `it-ir_score` has to reach `metric_for_best_model`. §1 checks this.
2. The `miracl-ita` / `squad-ita` loaders against the live Hub schemas. §3 checks
   this.
3. `mteb` multi-vector support, which may not cover ColBERT in the installed
   version. §9 has the fallback.

---

## Stopping and resuming

The pipeline is built to be stopped overnight. Kill it at any point and re-run the
identical command the next morning:

```bash
pkill -f run_train.sh; pkill -f train_phase                              # stop
nohup bash scripts/run_train.sh > outputs/logs/train_nohup.log 2>&1 &    # resume
```

| situation | behaviour |
|---|---|
| stage already finished | detected via its `final/` model, skipped |
| stage interrupted | resumes from its newest complete checkpoint |
| killed mid-save | that checkpoint is ignored, falls back to the previous one |
| benchmark interrupted | models already in `results.json` are skipped |
| dataset build | cached under `outputs/dataset_cache`, not rebuilt |

Phase 1 saves every 250 steps and phase 2 every 2000, so a stop costs at most that
much work. Verified on the 2026-08-12 run: stopped 22:33, resumed 10:56 the next
day, zero steps repeated.

```bash
STAGE=phase1 bash scripts/run_train.sh    # one stage: cache|phase1|phase1_bench|phase2|bench
rm -rf outputs/phase1                     # genuinely restart that stage
uv run python scripts/train_phase1_contrastive.py --no-auto-resume   # same, per-phase
```

**The skip logic cuts both ways.** Because a stage with a `final/` model is
skipped, re-running `run_train.sh` after §8 does nothing until the stage
directories are deleted. §8 repeats this where it matters.

`outputs/dataset_cache` holds a few GB of text. It lives inside the gitignored
`outputs/`, so it never ships, but budget the disk.

---

## 0. Context — what was wrong, and what the literature says

### 0.1 The KD split budget bug (fixed)

`load_kd_italian` drained the LightOn splits in `KD_SPLITS` order and stopped when
`max_train_samples` ran out. `msmarco_it` is first and holds ~522k of ~1.56M rows:

| earlier run | budget | splits it actually saw |
|---|---|---|
| 80k KD | 80,000 | msmarco_it only |
| "mixkd" | 400,000 | msmarco_it only, plus mMARCO HN |
| fullkd | uncapped | all 8 splits |

So the reported "full KD lifts MLDR 0.277 to 0.352" was data composition, not KD
scale: fullkd is the only run that saw the wiki-style splits. "mixkd" was never a
mixture. `mldr_it` is ~1.4k rows (0.09%), so it never mattered either way.

Fixed: `kd_split_sampling = "proportional"` is the default, with a
`kd_split_min_share` floor.

### 0.2 Undertraining — the actual quality gap

`DeepMount00/Italian-ModernBERT-base` mean-pooled scores 0.004 nDCG@10 on MLDR: a
pure MLM with no retrieval ability. It was then trained with lr 3e-6 for ~3.1k
steps, `mini_batch_size == per_device_train_batch_size` (so GradCache was a no-op
and each query saw 256 in-batch negatives instead of ~1024), and one BM25-sampled
negative per query.

### 0.3 The KD design was wrong, not just miscalibrated

**Two teachers in one loss.** LightOn rows are scored by `mxbai-rerank-large-v2`,
the mMARCO hard-negative branch by `BAAI/bge-reranker-v2-m3`. Temperature matching
fixes how confident each sounds, not the fact that they can disagree about which
document is better. ColBERTv2, JaColBERT and GTE-ModernColBERT all distil from one
teacher.

**The mMARCO branch was redundant.** LightOn already contains an `msmarco_it`
split of ~522k rows, about a third of the set, scored by the same mxbai teacher.
mMARCO KD was never missing. It only looked missing because the split-budget bug
made every capped run 100% msmarco_it, so "fullkd hurts mMARCO" was really "a
generalist scores lower than a specialist on the specialist's own benchmark".

**Scale mismatch.** `Distillation` teaches `softmax(teacher_scores)`, not their raw
scale. mxbai spans ~5 to 9; `logit(p)` on bge probabilities with `eps=1e-6` spans
~[−13.8, +13.8], so it is near one-hot and roughly 3x sharper. The mMARCO branch
then dominates every mixed batch and the soft-label benefit disappears.

**Decision: `include_mmarco_hn = false`.** One teacher, no temperature to guess, no
padded rows. `configs/phase2_distill_mmarco_hn.toml` keeps the branch as an
ablation.

**`kd_n_ways` stays at 11** until measured. Raising it to 32 helps only if the
source stores that many documents per query; otherwise the extra slots are padding
that costs a full encoder pass each and teaches nothing. §2 prints the real count.

**One structural limit that does not go away:** `Distillation` compares student and
teacher only over the documents in the row. There are no in-batch negatives, so
phase 2 never trains the model to discriminate against a whole corpus. Only phase 1
does. This is why phase 2 can improve KL while making retrieval worse, which is
exactly what happened in round 1 (§5).

### 0.4 Benchmark protocol

- `jina-colbert-v2` was indexed at 180 tokens while ours ran at 512, handicapping
  the strongest baseline on long documents.
- BM25 used `text.lower().split()`: no stemming, no stopwords. BM25 rises once
  this is fixed.
- Pooled mMARCO inflates absolute scores badly: jina scores 0.849 there against a
  published full-corpus 0.337. `results.json` now carries
  `comparable_to_literature: false` for pooled splits.
- No significance testing at all. On 200 MLDR queries the standard error is
  ±0.02 to 0.03, so the headline gaps were never established.

### 0.5 Competitive landscape (checked August 2026)

[`VAGOsolutions/SauerkrautLM-Multi-ModernColBERT`](https://huggingface.co/VAGOsolutions/SauerkrautLM-Multi-ModernColBERT)
is PyLate/ModernBERT late interaction tuned for 7 European languages including
Italian. `mLateOn` and `ColBERT-XM` were wired in this repo and never run. All
three are now in `DEFAULT_MODELS` and are the reference points for any claim.

### 0.6 What the literature says about this recipe

[ColBERT-Zero](https://huggingface.co/blog/lightonai/colbert-zero):

- dense init plus distillation only: 54.09 nDCG@10
- full multi-vector pretraining: 55.43
- dense init plus supervised contrastive plus distillation: 99.4% of full
  pretraining at ~10x lower cost

That third line is this recipe. Their other warning, that whatever prompt setup the
base model was trained with needs preserving downstream, does not apply here: the
mnrl init uses no prefixes (checked its model card).

[mxbai-edge-colbert-v0](https://arxiv.org/pdf/2510.14880) names hard-negative
mining and training-data composition as the primary quality drivers, above any
hyperparameter. That is §8.

---

## 1. Smoke test

Runs the whole pipeline in miniature. Minutes, not hours. Do it before spending
GPU-days.

```bash
uv run python scripts/train_phase1_contrastive.py --config configs/phase1_contrastive.toml --smoke
uv run python scripts/train_phase2_distill.py     --config configs/phase2_distill.toml     --smoke
```

Pass criteria: both phases log a line like
`ir eval @ step N: {'it-ir_mldr_ndcg@10': ..., 'it-ir_mmarco_mrr@10': ..., 'it-ir_score': ...}`
and exit clean. Phase 1 runs the evaluator for diagnosis; phase 2 selects
checkpoints on it. If `it-ir_score` is missing, the evaluator wiring is broken and
that needs fixing before anything else.

The first smoke run downloads MIRACL-ita and SQuAD-ita (a few hundred MB).
Subsequent runs come from cache.

---

## 2. Measure before configuring

```bash
uv run python scripts/inspect_kd_scores.py --samples 2000 --temperatures 1 2 3 4 5
```

`docs(mean/min)` decides `kd_n_ways`. If the LightOn splits store ~11 documents per
query, leave it at 11. Raise it only up to what the data contains.

`entropy(mean/med)` matters only for the ablation. The default config uses one
teacher, so there is no temperature to tune. For
`configs/phase2_distill_mmarco_hn.toml`, pick the `mmarco_hn` temperature whose
mean entropy is closest to the LightOn splits', weighting `msmarco_it` and
`trivia_it` most since they dominate. The committed `3.0` is a guess, not a
measurement. If even the highest temperature leaves it much sharper, also set
`mmarco_hn_score_clip = 6.0`.

---

## 3. Baselines first (~half a day, no training)

```bash
uv run python scripts/run_benchmark.py --output-dir outputs/benchmark
```

All four Italian benchmarks, length-matched at 512, with bootstrap intervals.
Those are now the defaults, so an unmatched comparison cannot happen by accident.
Runs every baseline including the three previously-unrun ColBERTs.

Running this before the long training job means the trained model's number lands in
an already-populated table.

Two failures are expected here and can be ignored. `ItColBERT` and
`ItColBERT (phase1-only)` point at `outputs/final` and `outputs/final_phase1`,
which do not exist yet. They are recorded with an `error` and retried automatically
once training has produced them; the benchmark only skips models that already have
metrics.

Any pre-existing `results.json` is not comparable (180-token jina index, whitespace
BM25, no intervals). The archive step above already produces a clean directory.

---

## 4. Train — phase 1 is the load-bearing part

`run_train.sh` runs every stage in order and is safe to interrupt (see above).

```bash
nohup bash scripts/run_train.sh > outputs/logs/train_nohup.log 2>&1 &
tail -f outputs/logs/phase1.log
```

| knob | before | now | why |
|---|---|---|---|
| init | raw MLM | `nickprock/Italian-ModernBERT-base-embed-mmarco-mnrl` | already retrieves (0.302 MLDR / 0.508 mMARCO alone) |
| lr | 3e-6 | 1e-5 | was far below the ColBERT norm |
| batch / mini | 128 / 128 | 512 / 32 | GradCache was a no-op; 4x negatives, ~4.6k steps |
| negatives | BM25 triples | plus reranker-mined HN | official triples are too easy |
| data | mMARCO only | plus MIRACL-ita, SQuAD-ita | see below |
| samples | 400k | ~2.4M total | ~3.1k steps was not enough |
| doc length | 180 | 512 | trained at 180 but indexed at 512 |

Why the extra Italian sources matter: the init was trained on 39.7M mMARCO samples
and scores 0.302 on MLDR, below BM25. Init, phase-1 data and the KD set were all
mMARCO. More mMARCO reinforces the axis the model is already fine on. That is the
structural reason long-document retrieval lags, and it does not fix itself.

### Validation during phase 1

Two evaluators run together every 250 steps:

- `ColBERTTripletEvaluator`, a cheap divergence check. Against BM25-sampled
  negatives it saturates above ~0.95 within a few hundred steps, so it indicates
  when something is broken, not when something is good.
- `ItIREvaluator`, real nDCG@10 / MRR@10 on small pooled slices. This is the one
  that answers whether the curve has flattened or more training would help.

No early stopping and no `load_best_model_at_end` in phase 1, deliberately. One
epoch over ~2.4M triplets cannot overfit, so there is nothing to protect against,
and the schedule is warmup plus linear decay, so stopping early leaves the model at
a high learning rate, usually worse than the annealed endpoint.
`load_best_model_at_end` would then keep that mid-schedule checkpoint. Both knobs
exist in the config and are off; turning them on logs a warning.

### Watch for

- OOM: lower `mini_batch_size` (32 to 16), not the batch size. The batch controls
  negatives; the mini-batch controls memory.
- Triplet accuracy dropping early means lr is too high. Try 7e-6.
- `it-ir_score` still climbing at the last eval means train longer, not shorter:
  `num_train_epochs = 2`, or batch 1024. This is the decision the IR evaluator
  exists to inform; triplet accuracy saturated hundreds of steps earlier and cannot
  answer it.

### Result — round 1 (`outputs/logs/phase1.log`)

Completed one epoch, 4858 steps, 4.547e+04s (~12.6h), final train loss 0.0754.
Last IR eval at step 4750: MLDR nDCG@10 0.4479, mMARCO nDCG@10 0.9317,
`it-ir_score` 0.6854. Benchmark numbers for this checkpoint are in §6 under
`ItColBERT (phase1-only)`.

---

## 5. Phase 2 — distillation

Runs automatically inside `run_train.sh`. Config: `configs/phase2_distill.toml`.

| knob | before | now |
|---|---|---|
| split budget | msmarco_it only | proportional over all 8 |
| teachers | two (mxbai + bge) | one (mxbai) |
| selection | hold-out KL | `it-ir_score` = mean(MLDR nDCG@10, mMARCO MRR@10) |
| doc length | 180 | 512 |

`run_train.sh` benchmarks `outputs/final_phase1` before distilling. That ablation
costs ~30 minutes and shows what the KD stage is worth once phase 1 is not broken,
which had never been measured here.

### Watch for

- IR eval costs a few minutes per eval. Fine at `eval_steps = 2000`. If it
  dominates, raise `eval_steps`; do not shrink the corpora below ~2k docs, the
  metric gets noisy.
- OOM: drop `per_device_train_batch_size` 8 to 4 before touching `kd_n_ways`.
- mMARCO MRR falling while MLDR nDCG rises is the old forgetting failure. The
  proportional mixture was expected to prevent it. It did not (see below).

### Result — round 1, run 2026-08-13 (`outputs/logs/phase2.log`)

Early-stopped at step 10000, which is 4.4% of one epoch (225k steps total). The
selected checkpoint was step 2000, the first eval. Every later eval was worse.

| step | MLDR nDCG@10 | mMARCO nDCG@10 | `it-ir_score` | hold-out KL |
|---|---|---|---|---|
| phase 1 final | .4479 | .9317 | .6854 | — |
| 2000 (selected) | .4963 | .8952 | .6885 | 1.119 |
| 4000 | .4376 | .8666 | .6445 | 1.106 |
| 6000 | .4838 | .8271 | .6454 | 1.095 |
| 8000 | .4369 | .7946 | .6067 | 1.084 |
| 10000 | .4502 | .7413 | .5848 | 1.075 |

Hold-out KL improved monotonically while IR degraded monotonically: the student
copied the teacher better and retrieved worse. 11-way KL supervises ordering inside
a small candidate list and gives no signal to keep the global embedding space
separable, which is what retrieval over a 100k corpus needs. This is the §0.3
structural limit showing up in practice, and the proportional mixture did not
prevent it.

The learning rate was still climbing at step 10000 (8.888e-6; `warmup_ratio = 0.05`
over 225k steps is 11250 warmup steps, so the `1e-5` peak was never reached).
Degradation tracked the learning rate upward. The whole run cost 6339s (~1.8h), so
it is cheap to redo.

Phase 2 is still worth keeping. The in-training `it-ir_score` (MLDR and mMARCO
only) rated it +0.003, which is noise, but the full 4-benchmark run disagrees:

| benchmark | phase1-only | phase2 final | Δ |
|---|---|---|---|
| MLDR-it | .3484 | .4008 | +.052 |
| MIRACL-ita | .6903 | .7194 | +.029 |
| SQuAD-ita | .9041 | .9026 | −.002 |
| mMARCO-it | .7784 | .7509 | −.028 |

3 of 4 up. The only loss is mMARCO, which is phase 1's own training distribution,
while everything else gains. That is the trade KD is supposed to make. MLDR gained
+.052 despite `mldr_it` being 1386 of 900510 rows (0.15%) of the KD mix, so the
gain comes from the domain breadth of the whole mixture, not from the MLDR rows.

Two conclusions for the rerun: the gain is real but was captured in the first 0.9%
of the KD data and then lost, and `it-ir_score` is too narrow to see it. Both are
addressed in §7.

---

## 6. Read the results — with statistics

### Targets

| benchmark | previous best | target | stretch | round 1 actual |
|---|---|---|---|---|
| MLDR-it nDCG@10 | 0.352 | > 0.40 | > 0.45 (beats bge-m3) | 0.4008, target met but inside noise |
| mMARCO-it pooled MRR@10 | 0.491 | > 0.60 | > 0.80 | 0.7196, target met |
| vs SauerkrautLM-Multi-ModernColBERT | not run | win on ≥3 of 4 | win everywhere | 4 of 4, stretch met |

Both mMARCO and MLDR come from one checkpoint. Winning one and losing the other
means the mixture is still unbalanced, in which case go to §7.

The MLDR row was selected on the MLDR test set (§11.1): phase 2 picked its
checkpoint by that exact metric on those exact 200 queries. Checked 2026-08-16
against the held-out half and against all 12 other models — no inflation is
detectable, because round 1 drew from only 5 checkpoints and took the first. The
number stands. What does not stand is reading .4008 vs a .40 target as a pass: the
standard error there is ~.03, so that margin is noise either way, and the held-out
half reads .3890.

### Where round 1 stands

Ranking of `ItColBERT` among the 13 benchmarked models:

| benchmark | nDCG@10 | rank | ahead of it |
|---|---|---|---|
| SQuAD-ita | .9026 | 2 / 13 | mLateOn (.9480) |
| mMARCO-it | .7509 | 6 / 13 | jina-colbert-v2, e5-large, mLateOn, e5-base, bge-m3 |
| MIRACL-ita | .7194 | 6 / 13 | mLateOn, jina, e5-large, bge-m3, e5-base |
| MLDR-it | .4008 | 5 / 13 | BM25 (.4850), mLateOn, bge-m3, e5-large, e5-base |

Three of these four are in-domain for this model and out-of-domain for every
baseline: the `yuri-no/squad-ita` and `yuri-no/miracl-ita-argos` train splits are
phase-1 training data (different splits, leakage-checked in §10.1, but the same
source and translation), and `msmarco_it` is in the KD mixture. Those three numbers
are inflated relative to the baselines and need stating as such wherever they are
reported.

MLDR-it is the only clean out-of-domain benchmark. It is also the weakest result,
and it loses to BM25 — though that specific comparison is length-asymmetric, since
BM25 indexes the full document text while every neural model is truncated at 512
tokens (§11.2). Re-run MLDR chunked before repeating the BM25 claim.

### Test every claim

```bash
uv run python scripts/compare_models.py --benchmark-dir outputs/benchmark \
  --baseline "ItColBERT" --all
```

Prints, per benchmark and metric, the delta, its 95% interval, a p-value and a
SIGNIFICANT / not-significant verdict. A ranking claim that comes back "not
significant" should not be reported as a ranking claim. On 200 MLDR queries a 0.017
gap lands around p≈0.9.

---

## 7. Fixing phase 2 — do this before round 2

**Status: applied 2026-08-15 in `configs/phase2_distill.toml`, never run at
scale.** All five items below are in the config, and §7.1 smoke passed. The round-2
phase-2 run is the first real test of any of them. This is the part of round 2 most
likely to move the numbers — see §8.2, where the phase-1 half is coming out flat.

Round 1 (§5) confirmed the forgetting failure and also settled the older open
question of whether phase 2 earns its place: it does, 3 of 4 benchmarks up. The
remaining problem is that it degrades after step 2000. In priority order:

1. **Turn on the replay stream: `contrastive_anchor_enabled = true`.** This is the
   largest lever and it was off for the whole of round 1. Already implemented
   (`src/it_colbert/train_phase2.py:86-107`): it runs a mMARCO contrastive stream
   next to the KD stream, supplying the global-separability signal that KL alone
   does not provide. Never executed, so smoke-test it first (§7.1).
   sentence-transformers uses one batch size for all datasets, so the contrastive
   stream runs at the KD batch size and acts as a regulariser, not a phase-1
   replacement.
2. **`learning_rate` 1e-5 to 2e-6, `warmup_ratio` 0.05 to 0.01.** Degradation
   tracked the learning rate upward through the whole run. This is a refinement
   stage on an already-trained phase-1 model, and 1e-5 is a from-scratch rate.
3. **`eval_steps` 2000 to 500.** The selected checkpoint was the first eval, so the
   real optimum lies somewhere in steps 0 to 2000 and was never measured. An IR
   eval takes ~72s.
4. **Add MIRACL to `ir_eval`, or rebalance `ir_eval_weights`.** The current
   0.5 MLDR / 0.5 mMARCO composite scored phase 2 at +0.003 when the actual gains
   were +.052 MLDR and +.029 MIRACL. The selection metric is blind to where phase 2
   helps, and it is what early stopping fires on.
5. **Rebalance if still needed:** raise `kd_split_min_share` so the wiki-style
   splits get more room. Only after 1 to 4, and not together with another change.

### 7.1 Smoke-test the replay stream first

`contrastive_anchor_enabled = true` calls `build_phase1_dataset()` from inside
phase 2 (`train_phase2.py:87`) and hands sentence-transformers a two-dataset
`DatasetDict` with a per-dataset loss dict. That path has never run.

There is no CLI flag for it; the entrypoints are TOML-driven (`--config`,
`--smoke`, `--resume`, `--no-auto-resume`, see `src/it_colbert/cli.py`). Add one
line to the `[phase2]` block of `configs/smoke.toml`, which already sets
`contrastive_anchor_samples = 512` but leaves the switch off:

```toml
contrastive_anchor_enabled = true
```

Then:

```bash
# smoke phase 2 reads outputs/smoke_phase1/final, so build that first
uv run python scripts/train_phase1_contrastive.py --smoke
uv run python scripts/train_phase2_distill.py --smoke
```

Check that the log contains `phase2 contrastive anchor on: N replay triplets
alongside M kd rows`, that both losses appear, and that it reaches a save without
OOM. Then set the same switch in `configs/phase2_distill.toml`.

Expect it to be slower per step, since it trains on two streams. Watch VRAM: round
1 already had to drop `per_device_train_batch_size` from 8 to 4 to fit 11-way KD at
length 512 on the 3090, and the contrastive stream runs at that same batch size.

---

## 8. Round 2 — self-mined hard negatives

**Status 2026-08-16: running, phase 1 at step ~1800 of 5222.** Mining finished
2026-08-14, retrain started 2026-08-15 13:04, interrupted overnight, resumed
2026-08-16 09:45 from `checkpoint-1750`. ~80 min per 250 steps, so ~18h of phase 1
left, then `phase1_bench`, then phase 2. §8.1 records what mining produced, §8.2
reads the curve so far, §8.3 is the decision gate. **Let it finish** — §8.2 explains
why stopping early answers nothing.

Do §7 before the retrain below. Mining and training are separate scripts:
`mine_hard_negatives.py` does not chain into `run_train.sh`, so nothing starts on
its own. But the retrain runs `STAGE=all`, which is phase 1 and phase 2, and with
the round-1 phase-2 config it reproduces the §5 collapse: ~12.6h of phase 1 plus a
phase 2 that discards the result after step 2000.

Mining itself is safe to run first and safe to let finish. It is inference-only: it
reads `outputs/final`, writes `outputs/mined_hn`, and touches no weights. Mining
with the round-1 model is also the intended design rather than a compromise;
ColBERTv2 mines with the round-1 model. The miner needs to be good, not final.

```bash
uv run python scripts/mine_hard_negatives.py \
  --model outputs/final --output outputs/mined_hn \
  --queries 50000 --corpus-docs 500000 --negatives 8 --skip-top 5
```

Then uncomment `mined_negatives_path = "outputs/mined_hn"` in
`configs/phase1_contrastive.toml` and retrain:

```bash
# REQUIRED: finished stages are skipped, so round 2 needs the old ones cleared.
# keep the round-1 model: it is the current best model and the baseline the
# §7 phase-2 fix has to be measured against.
mv outputs/final outputs/final_round1
rm -rf outputs/phase1 outputs/phase2
bash scripts/run_train.sh
```

Without the `rm -rf`, `run_train.sh` sees `final/` in both stages and exits having
done nothing.

`--skip-top 5` discards the highest-ranked hits before sampling: the top hits of a
decent model are frequently unlabelled positives, and training on those teaches the
model to demote correct answers. If quality regresses after round 2, raise it to 10
before concluding that mining does not help.

### 8.1 What mining actually produced (2026-08-14)

The command that ran differs from the one above: `--corpus-docs 200000`, not
500000, after an OOM on 27 GB WSL during Voyager retrieval (fixes in WORKLOG §8).
A 200k pool makes the mined negatives easier than specified.

- 46,583 rows, avg 8.0 negatives per query, 94/94 chunks.
- Phase 1 loads 4 per query: **186,332 triplets**, on top of round 1's 2,487,135.
  Total 2,673,467 → 5222 steps at batch 512, against round 1's 4858.

**The mined set is 100% mMARCO.** `mine_hard_negatives.py:83-106` builds both the
query list and the mining corpus from `load_mmarco_italian`, so round 2's only new
phase-1 data is harder negatives drawn from the distribution the model is already
strongest on. §4 and §10.2 both warn about exactly this: the init is a mMARCO
specialist, and more mMARCO reinforces the axis that is already fine. This was not
noticed when §8 was written.

### 8.2 Reading the phase-1 curve at 34% (2026-08-16)

Per-step numbers are in WORKLOG.md under "Round 2 phase 1 IR eval". The shape:
in-training mMARCO up at 7 of 7 evals, MLDR down at 6 of 7, composite flat. That is
what §8.1 predicts — harder same-distribution negatives sharpen the in-domain axis
and give the out-of-domain one nothing.

**Do not act on this table.** Three reasons:

1. The in-training MLDR slice is 150 queries over 2000 docs. Standard error is
   ~0.03 to 0.04, so no single gap here is significant. Only the consistent sign
   across six evals carries any weight, and that is weak evidence, not a result.
2. It is not the benchmark. Round 1's phase-1 end read .4479 on this slice and
   .3484 on the full 10k corpus. Absolute values from this evaluator mean nothing.
3. Round 1's own MLDR curve peaked at .485 at step 1250 and fell to .448 by 4750.
   The early curve is not monotone, and both runs are mid-warmup-decay here.

Stopping now would trade a measurable answer for an unmeasurable one: it leaves the
model at a high learning rate (§4), and it skips the phase-2 fixes (§7), which are
the untested half of round 2 and the half with the larger expected effect.

### 8.3 Decision gate — `phase1_bench`, not the in-training slice

`run_train.sh` benchmarks `outputs/final_phase1` on all four suites before phase 2.
That is the apples-to-apples comparison. Round 1 phase1-only:

| benchmark | round-1 phase1-only | round-2 verdict |
|---|---|---|
| MLDR-it nDCG@10 | .3484 | the number that decides whether mining helped |
| mMARCO-it MRR@10 | .7784 | expected up; in-domain, do not lead with it |
| MIRACL-ita nDCG@10 | .6903 | |
| SQuAD-ita nDCG@10 | .9041 | |

Then test it rather than eyeballing it:

```bash
uv run python scripts/compare_models.py --benchmark-dir outputs/benchmark \
  --a "ItColBERT (phase1-only)" --b "ItColBERT (round1)" --benchmark mldr-it
```

Keep `outputs/final_round1`. It is the current best model until round 2 benches,
and the baseline every round-2 claim is measured against.

**If MLDR lands at or below .348 while mMARCO rises**, mining did what §8.1
predicts and the conclusion is not "mining does not work" but "mining more mMARCO
does not work". Do not raise `--skip-top` to 10 and re-mine; that tunes negative
difficulty when the problem is negative *distribution*. Go to §10.5 and §10.6
instead, both cheap and both never checked, and if mining is repeated, mine over a
corpus that includes the wiki-style and long-document sources.

**If MLDR rises**, the mined negatives transferred, and §10.5 / §10.6 stay as
listed rather than becoming the next round.

Either way, phase 2 still runs: §7 is being tested for the first time and its
result is independent of how the phase-1 half lands.

---

## 9. Release

1. **Community-comparable numbers.** There is no Italian MTEB (there is
   MTEB-French, PL-MTEB, VN-MTEB, FaMTEB, MTEB-BR, but nothing for Italian). Start
   by checking what already exists:

   ```bash
   uv pip install "mteb>=1.29"
   uv run python scripts/run_mteb_italian.py --list-tasks
   uv run python scripts/run_mteb_italian.py --model outputs/final
   ```

   If the installed `mteb` lacks multi-vector support, use it for the dense
   baselines (`--dense`) and keep the in-repo harness for late interaction.

2. **dim-64 variant** (`configs/*_dim64.toml`). Multi-vector index size is the main
   practical cost of ColBERT. Train it only after dim-128 has a number to compare
   against.

3. **Model card contents.** The harness, the exact dataset list, the pooled-corpus
   caveat, the confidence intervals, and the limitations, including that MIRACL-ita
   and SQuAD-ita are community machine translations. Also, from §11: how phase-2
   checkpoint selection was done and on which queries (§11.1), whether MLDR numbers
   are truncated or chunked (§11.2), pinned dataset revisions (§11.8), and index
   size / query latency (§11.7).

   **§11.2 is a release blocker.** A BM25 comparison where only one side reads the
   whole document puts a claim in the model card that the harness does not support.
   §11.1 is fixed in code and its round-1 numbers came back clean, so it is now a
   disclosure item rather than a blocker — say which half the numbers are from.

4. **What round 1 supports.** Not "first Italian ColBERT", and not best-on-Italian
   either: §6 shows 6th, 6th and 5th of 13 on mMARCO, MIRACL and MLDR. What the
   numbers do support:
   - ahead of every other late-interaction model on all 4 benchmarks (SauerkrautLM,
     ColBERT-XM, and jina-colbert-v2 on 3 of 4); mLateOn still wins 3 of 4;
   - the only Italian-specialised ColBERT, against multilingual generalists;
   - competitive with dense multilingual baselines at similar size.

   SQuAD-ita (.9026, 2nd) should not lead, since it is in-domain here and
   out-of-domain for every baseline.

---

## 10. Open items

1. ~~Verify no train/test query overlap between the MIRACL-ita / SQuAD-ita `train`
   splits used in phase 1 and the `dev` / `test` splits used in the benchmark.~~
   Checked 2026-08-13 (`yuri-no/miracl-ita-argos`, `yuri-no/squad-ita`, direct
   query-text set intersection): MIRACL-ita train (2859 unique) vs dev (799), 0
   overlap. SQuAD-ita train (53989) vs test (7583), 9 overlapping query strings,
   hand-checked: each pairs the same generic question text with a different
   positive passage, a known SQuAD annotator artifact rather than duplicated rows.
   No real leakage. KD's `squadv2_it` split
   (`lightonai/embeddings-fine-tuning-filtered-it`, different source and
   translation) was checked too: 1 overlap out of 7583 test queries, same pattern,
   not real leakage. Nothing left open here.
2. **More phase-1 mMARCO.** The `triples.train.ids.small` file holds ~39.8M rows,
   but only ~500k unique queries; the rest is the same queries re-paired with
   different BM25 negatives. At batch 512 each query already sees 511 in-batch
   negatives per step, far more than the ~80 stored ones, so extra rows buy little.
   The init checkpoint was itself trained on all 39.8M, so re-running them teaches
   the MaxSim objective, not new Italian retrieval knowledge. Raise `mmarco_samples`
   only if `it-ir_score` is still climbing at the last phase-1 eval; otherwise §8 is
   the better use of the same GPU-hours.
3. **Full-corpus mMARCO eval.** Index all 8.8M with PLAID once, at the end, for a
   number comparable to the literature. Brute-force MaxSim over the full collection
   is not viable: the 100k-doc pooled run already takes hours per model, and linear
   scaling puts the full corpus at roughly a day and a half for a single model.
4. **`ReDiX/wikipediaQA-ita`** (105k wiki QA rows), a plausible extra Italian
   source, but it ships as one raw JSONL with no dataset-server access, so the
   schema needs checking by hand.
5. **Query length** is fixed at 32. Check MLDR-it query token lengths; if a real
   fraction truncate, raise to 48 or 64. MLDR-it is the weakest benchmark (§6), so
   this is worth checking before assuming the gap is purely a document-length
   problem. **Promoted by §8.3:** this is a few minutes of tokenizer counting with
   no training attached, so run it while round 2 finishes rather than after.
6. **A dedicated long-document Italian training source.** MLDR-it is the only clean
   out-of-domain benchmark and the weakest result, and `mldr_it` is 0.15% of the KD
   mixture. Nothing in phase 1 or phase 2 currently targets long documents.
   **Promoted by §8.3:** round 2 adds 186k more mMARCO triplets and nothing
   long-document, so if its MLDR number does not move, this is the next round, not
   another mining pass. Candidates to check before committing GPU time: the Italian
   slice of `Shitao/MLDR` *train* (disjoint from the test split used in §3),
   Italian Wikipedia full-article retrieval pairs, and `ReDiX/wikipediaQA-ita`
   (item 4) once its schema is verified.
7. **Mining is single-domain.** `mine_hard_negatives.py` draws queries and corpus
   from `load_mmarco_italian` only (§8.1). Any future mining pass should take a
   `--corpus` / `--queries` source list so negatives can be mined over the wiki and
   long-document sources too, otherwise every round tightens the same axis.

---

## 11. Methodology audit — 2026-08-16

Code audit while round 2 phase 1 was running. Items 1 and 2 both inflate the
headline in the same direction and both affect claims already written down in §6
and §9, so they are not deferrable. Nothing here requires stopping the run.

Order: **11.1 and 11.2 before round-2 phase 2 starts** (~20h of phase 1 left as of
this writing), 11.3 during phase 2, the rest after round 2 benches.

### 11.1 Checkpoint selection runs on the test sets — fix before phase 2

`ir_eval.py:101` builds its MLDR split from `load_mldr_italian(split="test")`: the
same 200 queries `run_benchmark.py` reports. `configs/phase2_distill.toml:62` sets
`ir_eval_mldr_queries = 200`, so it is all of them, not a sample. MIRACL is the
same story — `ir_eval.py:116-131` uses `TEVATRON_STYLE_ITALIAN["miracl-ita"]`,
which is the benchmark's dev split.

Phase 2 drives `load_best_model_at_end` and early stopping off `it-ir_score`. So
round 1 chose step 2000 from 5 candidates *by MLDR test nDCG@10*, and then §6
reports MLDR test nDCG@10 = .4008 as the result. That is selection on the test set.

**Measured 2026-08-16: round 1 shows no detectable inflation.** Splitting the
existing per-query files by the new partition and comparing every benchmarked
model:

| mldr-it nDCG@10 | selection half | report half | delta |
|---|---|---|---|
| ItColBERT (selected) | .4136 | .3890 | +.0246 |
| field mean, 13 models | — | — | +.0308 |
| Italian-ModernBERT-mmarco-mnrl | .3534 | .2545 | +.0989 |
| BM25 | .4638 | .5047 | −.0409 |

The selection half is intrinsically easier for almost every model, selected or
not, so the delta measures query difficulty. ItColBERT's +.0246 is *below* the
13-model average, which is not what winner's curse looks like. MIRACL-ita agrees
and is the cleaner control: it was not in round-1's `ir_eval` at all, so no bias is
possible there, and none appears (ItColBERT −.0468 against a field mean of −.0199).

Why the theory over-predicted: round 1 drew from 5 checkpoints and picked the
first, so selection pressure was almost nil. **So round-1 numbers stand.** The
earlier estimate of +.02 to +.04 in this section was an upper bound from theory,
not a measurement, and the measurement does not support it.

The exposure is prospective, and it is the reason to fix this now rather than
after: `eval_steps` 2000 → 500 (§7 item 3) turns 5 candidates into dozens, and the
bias grows with the number of draws. Round 2 is where this would start to bite.

Phase 1 is clean. It runs the same evaluator but never selects on it
(`load_best_model_at_end = false`, `early_stopping_patience = 0`, §4), so there it
is diagnostic only.

**Implemented 2026-08-16.** `split_query_ids()` in `benchmark/stats.py` assigns
each query to a `selection` or `report` half by a keyed blake2b hash *of the query
id*, not by slicing a shuffled list. That matters: training sees a sampled slice,
the benchmark scores only queries with qrels, and a comparison keeps only the
queries two models share, so a list-relative split would move the boundary with
the input and hand a selection query back as a reporting query. Per-id hashing
makes any subset split the same way the full set does. The halves come out near-
equal rather than exact — MLDR 96/104, MIRACL 379/420, mMARCO 3451/3529,
SQuAD 3895/3714.

Wiring:

- `ItIREvaluator(query_half=…)`, applied before subsampling, so raising
  `ir_eval_*_queries` cannot leak a reporting query into the trainer's view;
- `ir_eval_query_half` on both configs. **Phase 2 defaults to `"selection"`**
  because it selects; **phase 1 defaults to `"all"`** because it does not — its
  curve is read by a human deciding whether to train longer, and keeping the full
  set leaves round-2 phase-1 numbers comparable with round 1. The evaluator logs a
  warning whenever it runs on `"all"`;
- `scripts/report_query_half.py --half report` for held-out means with intervals,
  and `compare_models.py --query-half report` for held-out paired tests. Both read
  the per-query files the benchmark already wrote, so no rerun is needed — all 52
  existing files were checked and are correctly aligned;
- `save_per_query` now stores the ids `per_query_metrics` actually scored, via
  `scored_query_ids()`. It skips queries with no qrels, so the unfiltered list
  would have misaligned scores against ids. No current suite triggers it; this is
  a guard, not a repair.

Still to do: state in the model card which half the reported numbers come from.

### 11.2 The MLDR-vs-BM25 comparison is length-asymmetric

`BM25Retriever.__init__` (`benchmark/retrievers.py:191-196`) tokenizes `d["text"]`
whole — no truncation anywhere in that path. Every neural model is capped at 512
tokens. `chunk_chars` defaults to `0` (`benchmark/run.py:225`), so
`chunk_long_documents` / `maxpool_chunks_to_documents` exist and have **never run
in any reported number**. The `effective_length: 512` recorded for BM25 in
`results.json` is cosmetic; it truncates nothing.

MLDR documents are long. BM25 reads all of each one, every neural model reads the
first 512 tokens. bge-m3 is clipped to 512 too despite an 8192 window, so the
benchmark as run cannot answer whether long context helps here.

This does not erase the gap — multilingual-e5-large is also capped at 512 and still
scores .431 against ItColBERT's .401 — but "loses to BM25 on the only clean
out-of-domain benchmark" (§6) is partly a protocol artifact and is stated in the
release plan as if it were a model property.

Settle it with one benchmark run, no training, ~30 min:

```bash
uv run python scripts/run_benchmark.py --benchmarks mldr-it \
  --chunk-chars 2000 --chunk-overlap-chars 200 \
  --output-dir outputs/benchmark_chunked
```

Separate `--output-dir`: chunked and unchunked MLDR numbers are different
protocols and must not land in the same `results.json`. Whichever becomes the
headline, report both and say which is which.

### 11.3 Round 2 bundles changes that cannot be separated

Phase 1 gets mined HN; phase 2 gets five changes at once (§7 items 1-4 plus the
config's lr/warmup pair). §7 item 5 says "not together with another change" and
then items 1-4 shipped together.

`phase1_bench` rescues the phase-1 half (§8.3). The phase-2 half will not be
attributable: if it beats round 1, the cause is unknown. Phase 2 costs ~1.8h, so
run it more than once — at minimum anchor-only, then anchor + the lr change. Record
in WORKLOG which variant produced `outputs/final`.

### 11.4 n=1 everywhere

Bootstrap CIs measure query-sampling variance, not training variance. No seed
repeats, so a .015 round-2-over-round-1 delta is unfalsifiable.

Cheap partial estimate, no extra training: benchmark 2-3 checkpoints from the phase-1
decay tail (4000 / 4500 / 4750) and treat the spread as a floor on run-internal
noise. Round 1's tail evals sat in .4403-.4507 on the in-training slice, which
already suggests the floor is not small.

### 11.5 Every corrective lever pushes the same axis

Mining is 100% mMARCO (§8.1). The phase-2 anchor stream is *also* 100% mMARCO:
`train_phase2.py:87` calls `build_phase1_dataset(mmarco_samples=…,
include_wiki_hn=False)`, so it replays mMARCO triples only.

For the anchor that is coherent — its stated job is holding the short-passage
distribution while KD pulls elsewhere. But taken together, round 2 has two levers
and both strengthen the axis that is already strongest, with nothing aimed at the
one benchmark that is out-of-domain. See §10.6.

### 11.6 No hybrid BM25 + ColBERT

It was item 5 of the pre-rewrite roadmap (WORKLOG §7) and did not survive into this
runbook. BM25 leads MLDR outright and ItColBERT is 6th on mMARCO; rank fusion is
nearly free, needs no training, and is what anyone deploying this would actually
run. Add it as a benchmark row (`ItColBERT + BM25 (RRF)`) rather than a model.

### 11.7 No efficiency numbers

Index size on disk, query latency, peak memory — none are measured. That is
late interaction's actual argument at base size, and §9.2's dim-64 variant has no
measurement plan attached to it. Record per-model index bytes and mean query
latency in `results.json` during the next benchmark run; both are cheap to capture
where the index is already being built.

### 11.8 Dataset versions unpinned

The round-2 logs show `datasets` resolving every source from the local cache in
offline mode ("couldn't be found on the Hugging Face Hub (offline mode is
enabled)"). No `revision=` pins in the loaders. Reproducible on this machine,
not on anyone else's. Pin revisions in `data.py` and record them in the model card
before release.
