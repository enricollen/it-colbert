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

> **Fill in the results table before publishing.** The numbers below are
> placeholders; replace them with your own benchmark output and keep the
> confidence intervals.

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

## Evaluation

| Benchmark | Metric | Score (95% CI) |
|---|---|---|
| MLDR-it (test, ~10k docs) | nDCG@10 | _fill in_ |
| mMARCO-it (dev, pooled) | MRR@10 | _fill in — rank only_ |
| MIRACL-ita (dev, pooled) | nDCG@10 | _fill in_ |
| SQuAD-ita (test, pooled) | nDCG@10 | _fill in_ |

Protocol notes that must travel with these numbers:

- All late-interaction models are indexed at the **same** document length.
- Pooled corpora inflate absolute scores and are **not** comparable to published
  full-corpus numbers. Use them for relative ranking only.
- MLDR-it has 200 queries; differences under ~0.03 nDCG@10 are inside the noise.
  Report intervals, and use the paired bootstrap for any head-to-head claim.
- MIRACL-ita and SQuAD-ita are community machine translations, not official
  resources.

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
- Most training data derives from machine-translated mMARCO. Long-document and
  native-Italian performance is weaker than short-passage performance.
- Multi-vector indexes are substantially larger than dense bi-encoder indexes. A
  64-dimension variant is available if index size matters more than the last
  points of quality.
- Documents longer than 512 tokens are truncated at index time unless you chunk
  and max-pool.

## Citation

If you use this model, please also cite ColBERT/ColBERTv2, PyLate, mMARCO, MLDR
and Italian-ModernBERT.
