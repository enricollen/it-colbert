# Italian ColBERT — IR benchmark

Full SOTA comparison + ablations (doc256, full LightOn KD). Local eval on MLDR-it (~10k docs) and mMARCO-it (pooled 100k).

**Tags:** MLDR-it · nDCG@10 · mMARCO-it pooled · MRR@10 · fullkd ↑ MLDR · fullkd ↓ mMARCO

| Metric | Value |
| --- | --- |
| fullkd MLDR nDCG@10 | 0.352 |
| Δ vs ours@512 (MLDR) | +0.075 |
| Δ fullkd mMARCO MRR | −0.151 |

> ⚠️ **Superseded interpretation.** The numbers below are what the runs produced,
> but the "KD scale" reading of them was wrong. `load_kd_italian` drained the
> LightOn splits sequentially, so every capped run spent its whole budget on
> `msmarco_it`. fullkd (uncapped) is the only run that ever saw `trivia_it`,
> `nq_it`, `hotpotqa_it`, `fever_it` and `squadv2_it`, and mixkd was short-passage
> MS MARCO on both branches. Fixed via proportional split budgets — see
> [TODO.md](TODO.md) for the v2 recipe.

> **fullkd lifts long-doc MLDR — because of split diversity, not scale**  
> Phase-2 over the full LightOn Italian KD set (~1.56M, early-stop at step 92k by hold-out KL) raises MLDR nDCG@10 from 0.277 → 0.352, above BM25 and within ~0.017 of jina-colbert-v2. The cause is that this is the only run containing the wiki-style splits; `mldr_it` itself is ~1.4k rows (0.09% of the set) and contributed almost nothing.

> **same run hurts pooled mMARCO**  
> On mMARCO-it pooled 100k, fullkd drops MRR@10 from 0.491 → 0.341. The 80k checkpoint it is compared against was trained on `msmarco_it` alone, so this is a comparison between two differently-composed datasets, not between two KD budgets.

> **Protocol note**  
> mMARCO-it scores here use a 100k pooled corpus (qrel positives + reservoir sample), so absolute MRR is inflated vs published full 8.8M corpus numbers. Use them for relative ranking only. Literature MRR@10 for full-corpus mMARCO-it is in the published table below.

> **Length matching (not applied to the tables below)**  
> These runs indexed `jina-colbert-v2` at 180 tokens while ours ran at 512, so the strongest late-interaction baseline was handicapped on long documents and the real gap is wider than shown. BM25 also ran with a whitespace tokenizer (no stemming, no stopwords), which understates it. Both are fixed; re-run with `--colbert-doc-length 512` before quoting these tables.

## MLDR-it — long-document retrieval

Source: Shitao/MLDR italian test · 200 queries · 10,000 docs · primary metric nDCG@10 · leader: bge-m3

| Model | Kind | nDCG@10 | MRR@10 | R@100 |
| --- | --- | ---: | ---: | ---: |
| bge-m3 | dense | 0.453 | 0.422 | 0.710 |
| multilingual-e5-large | dense | 0.431 | 0.397 | 0.715 |
| multilingual-e5-base | dense | 0.429 | 0.400 | 0.655 |
| jina-colbert-v2 | colbert | 0.369 | 0.330 | 0.640 |
| **italian-colbert (fullkd)** | colbert | **0.352** | **0.319** | **0.565** |
| BM25 | lexical | 0.332 | 0.294 | 0.590 |
| Italian-ModernBERT-mmarco-mnrl | dense | 0.302 | 0.283 | 0.515 |
| **italian-colbert (ours@512)** | colbert | **0.277** | **0.247** | **0.550** |
| **italian-colbert (doc256)** | colbert | **0.227** | **0.200** | **0.495** |
| **italian-colbert (doc180)** | colbert | **0.216** | **0.188** | **0.465** |
| Ita-Search | dense | 0.189 | 0.166 | 0.425 |
| Italian-ModernBERT-base | dense | 0.004 | 0.002 | 0.125 |

## mMARCO-it — pooled 100k (relative)

Source: unicamp-dl/mmarco italian · 6,980 queries · 100,000 pooled docs · primary metric MRR@10 · leader: jina-colbert-v2

| Model | Kind | MRR@10 | nDCG@10 | R@100 |
| --- | --- | ---: | ---: | ---: |
| jina-colbert-v2 | colbert | 0.849 | 0.871 | 0.981 |
| multilingual-e5-large | dense | 0.824 | 0.850 | 0.978 |
| multilingual-e5-base | dense | 0.791 | 0.820 | 0.971 |
| bge-m3 | dense | 0.781 | 0.811 | 0.969 |
| Ita-Search | dense | 0.565 | 0.602 | 0.885 |
| Italian-ModernBERT-mmarco-mnrl | dense | 0.508 | 0.541 | 0.802 |
| **italian-colbert (ours)** | colbert | **0.491** | **0.517** | **0.699** |
| **italian-colbert (doc256)** | colbert | **0.411** | **0.432** | **0.599** |
| **italian-colbert (fullkd)** | colbert | **0.341** | **0.355** | **0.450** |
| Italian-ModernBERT-base | dense | 0.017 | 0.020 | 0.086 |

## Published mMARCO-it MRR@10 (full corpus)

Literature / model-card numbers on the official 8.8M-passage corpus — not directly comparable to the pooled run above.

| Model | MRR@10 | Source |
| --- | ---: | --- |
| jina-colbert-v2 | 0.337 | Jina model card |
| mono-mT5 | 0.303 | Bonifacio et al. 2021 |
| mColBERT | 0.292 | Bonifacio et al. 2021 |
| mE5-base | 0.280 | Wang et al. / ColBERT-XM |
| ColBERT-XM | 0.265 | Louis et al. 2024 |
| BM25 | 0.153 | Bonifacio / Jina table |

## Final takeaways

- fullkd vs the 80k checkpoint is a **data-composition** difference, not a KD-budget one. Neither is the right recipe; a proportional mixture of all LightOn splits plus mMARCO hard negatives is.
- Dense multilingual still owns long-doc MLDR (bge-m3 0.453). Late interaction SOTA on pooled mMARCO remains jina-colbert-v2 (0.849), and it was measured here at a 180-token handicap.
- The deeper problem is undertraining, not tuning: phase 1 started from a raw MLM (0.004 nDCG@10 before training) at lr 3e-6 for ~3.1k steps, with `mini_batch_size == per_device_train_batch_size` so GradCache did nothing and each query saw 256 in-batch negatives instead of 2048.
- Fixed levers, in order of expected gain: initialize from an already-IR-tuned checkpoint, raise lr to 1e-5, use the GradCache batch, train on mined hard negatives, select checkpoints on nDCG/MRR instead of hold-out KL. See [TODO.md](TODO.md).

---

Artifact: `outputs/benchmark/results.json` · fullkd = best val-KL @ step 92000 · MIRACL has no Italian split (not used).
