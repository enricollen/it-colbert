---
language:
- it
library_name: pylate
tags:
- colbert
- late-interaction
- sentence-transformers
- italian
- retrieval
- rag
pipeline_tag: sentence-similarity
license: apache-2.0
base_model: nickprock/Italian-ModernBERT-base-embed-mmarco-mnrl
---

# ItColBERT

Monolingual **Italian late-interaction (ColBERT)** retriever for semantic search
and RAG, built with [PyLate](https://github.com/lightonai/pylate).

## Why this model

Italian retrieval is mostly served by multilingual dense models and by
multilingual late-interaction models that include Italian among many languages.
This checkpoint is specialized on Italian retrieval data with a true MaxSim
multi-vector architecture, and is published together with a reproducible Italian
evaluation harness.

## Architecture

- Backbone: Italian ModernBERT base (via an mMARCO-tuned dense checkpoint)
- Projection: Dense `768 -> 128`
- Similarity: MaxSim (late interaction)
- Query length: 32 tokens
- Document length: 512 tokens

## Training

Two-stage recipe, following the ColBERT-Zero efficiency result that a supervised
contrastive stage followed by distillation reaches ~99.4% of full multi-vector
pretraining at roughly 10× lower cost.

**1. Supervised contrastive** (`CachedContrastive`, temperature 0.02):

- [`unicamp-dl/mmarco`](https://huggingface.co/datasets/unicamp-dl/mmarco) Italian triples
- [`hotchpotch/mmarco-hard-negatives-reranker-filtered`](https://huggingface.co/datasets/hotchpotch/mmarco-hard-negatives-reranker-filtered) — reranker-mined hard negatives, with likely false negatives filtered out
- [`nickprock/it-wiki-retrieval-synthetic-hn`](https://huggingface.co/datasets/nickprock/it-wiki-retrieval-synthetic-hn)
- [`yuri-no/miracl-ita-argos`](https://huggingface.co/datasets/yuri-no/miracl-ita-argos) and [`yuri-no/squad-ita`](https://huggingface.co/datasets/yuri-no/squad-ita) — community machine translations, included to widen the domain past machine-translated mMARCO

**2. Knowledge distillation** (`Distillation`, KL) from a single cross-encoder
teacher: [`lightonai/embeddings-fine-tuning-filtered-it`](https://huggingface.co/datasets/lightonai/embeddings-fine-tuning-filtered-it)
(`mxbai-rerank-large-v2` scores), with the sample budget spread proportionally
across all 8 splits.

Checkpoints are selected on pooled MLDR-it nDCG@10 and mMARCO-it MRR@10, not on
hold-out KD KL divergence.

## How we got here

The short version of how this model came to be, and why some obvious next
steps aren't in it:

1. **Start from a model that already retrieves.** Rather than fine-tuning a
   raw language model, we started from an Italian ModernBERT checkpoint
   already tuned for retrieval on mMARCO, and adapted it into ColBERT's
   multi-vector format. A published efficiency result shows this reaches
   ~99% of training a multi-vector model from scratch, at roughly a tenth of
   the cost.
2. **Broaden the training data, then distil.** The starting checkpoint knew
   only machine-translated mMARCO. We added more varied Italian data
   (Wikipedia-style Q&A, MIRACL, SQuAD), then applied a second
   distillation stage from a cross-encoder teacher. This is the model in
   this repository, and it became the strongest Italian-specialized
   late-interaction model we could benchmark it against — except on long
   documents, which was its weakest result.
3. **The long-document weakness turned out to be mostly mechanical.** The
   test documents in our hardest benchmark are long — a few thousand words
   each — and were being cut off at 512 tokens during indexing, discarding
   roughly 80% of the average document before the model ever saw it.
   Splitting long documents into overlapping chunks at query time, with no
   retraining at all, recovered most of the gap (see the two MLDR-it rows
   below).
4. **Two follow-up training attempts, both tested, neither beat that free
   fix.**
   - *Harder training examples*, mined from the model's own predictions on
     the same short-passage data: no improvement, and a small but real drop
     on the benchmark that measures generalization.
   - *Training a version of the model to read twice as much text per
     document*, instead of relying on chunking at query time: statistically
     no better than chunking a normally-trained model, once both were
     compared fairly on held-out data — and marginally worse.
5. **What shipped.** Since neither training attempt beat "train normally,
   then chunk long documents at query time," that's what this repository is:
   the model below, plus the chunking recipe as the recommended way to
   handle documents longer than 512 tokens.

## Evaluation

Four Italian retrieval benchmarks, all against the same 512-token document
index unless noted. 95% confidence intervals in brackets; treat any gap
smaller than the interval width as noise, not a real difference.

| Benchmark | Metric | Score (95% CI) | In-domain? |
|---|---|---|---|
| MLDR-it (test, ~10k docs) | nDCG@10 | 0.4008 [0.3404, 0.4589] | No — the clean out-of-domain test |
| MLDR-it, **chunked at query time** (see below) | nDCG@10 | 0.4610 [0.4002, 0.5212] | No |
| mMARCO-it (dev, pooled 100k) | MRR@10 (rank only) | 0.7196 [0.7104, 0.7291] | Partially — mMARCO is in the training mix |
| MIRACL-ita (dev, pooled) | nDCG@10 | 0.7194 [0.6984, 0.7375] | Partially — different split of a training source |
| SQuAD-ita (test, pooled) | nDCG@10 | 0.9026 [0.8974, 0.9075] | Partially — different split of a training source |

**MLDR-it is the only clean out-of-domain benchmark**, and the one worth
weighing most heavily. Its documents are long (median ~2,700 tokens); at a
plain 512-token index almost all of that content is truncated. Splitting each
document into 2,000-character overlapping chunks and max-pooling the scores
at query time — no retraining, ~7× the indexing cost — recovers most of the
gap. Use the truncated number if index size or latency is the constraint, the
chunked number if document coverage matters more.

### How it compares

Significance-tested (paired bootstrap) against the closest available
alternatives, same protocol:

| vs. | MLDR-it | mMARCO-it | MIRACL-ita | SQuAD-ita |
|---|---|---|---|---|
| SauerkrautLM-Multi-ModernColBERT | win | win | win | win |
| ColBERT-XM | win | win | win | win |
| jina-colbert-v2 | tie | lose | lose | win |
| mLateOn (strongest late-interaction competitor found) | lose | lose | lose | lose |
| large multilingual dense embedders (bge-m3, mE5-large) | mixed | lose | lose | win |
| BM25 | tie once chunked (loses truncated) | win | win | win |

Read plainly: this is the strongest **Italian-specialized** late-interaction
model we could find and benchmark against, and it beats most general-purpose
late-interaction alternatives outright. It does not beat the single strongest
multilingual late-interaction model found (mLateOn), and it does not beat
large multilingual dense embedders on most benchmarks — beating those was
never the goal; they are a different, much larger model class.

Protocol notes that must travel with these numbers:

- All late-interaction models are indexed at the **same** document length
  unless a row is explicitly marked "chunked".
- Pooled corpora inflate absolute scores and are **not** comparable to
  published full-corpus numbers. Use them for relative ranking only.
- MLDR-it has 200 queries; differences under ~0.03 nDCG@10 are inside the
  noise. Report intervals, and use the paired bootstrap for any head-to-head
  claim.
- MIRACL-ita and SQuAD-ita are community machine translations, not official
  resources, and both overlap the training data's source (different
  splits, checked for direct leakage).

Harness and exact protocol: see the project repository.

## Usage

```python
from pylate import models, rank

model = models.ColBERT("PATH_OR_HUB_ID", document_length=512, query_length=32)

docs = [
    "Roma è la capitale d'Italia.",
    "Milano è un centro finanziario.",
]
doc_emb = model.encode(docs, is_query=False)
q_emb = model.encode(["Qual è la capitale d'Italia?"], is_query=True)
results = rank.rerank(
    documents_ids=[list(range(len(docs)))],
    queries_embeddings=q_emb,
    documents_embeddings=[doc_emb],
)
print(results)
```

## Intended use

- Italian RAG first-stage retrieval or reranking
- Semantic search over Italian corpora

## Limitations

- Optimized for Italian; cross-lingual retrieval is not the focus.
- Much of the training data derives from machine-translated mMARCO, and it
  shows: short-passage retrieval is the strongest result, long-document
  retrieval the weakest, even after the chunking fix.
- Documents longer than 512 tokens are truncated at index time unless you
  chunk and max-pool — see "How it compares" above for the size of that
  effect and the cost of fixing it.
- Multi-vector indexes are substantially larger than dense bi-encoder indexes,
  and chunking multiplies that further. A 64-dimension variant trades some
  quality for smaller indexes if that matters more here.
- Behind the strongest multilingual late-interaction model found (mLateOn) on
  every benchmark tested, and behind large multilingual dense embedders on
  most — see "How it compares".

## Citation

If you use this model, please also cite ColBERT/ColBERTv2, PyLate, mMARCO, MLDR
and Italian-ModernBERT.
