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
base_model: DeepMount00/Italian-ModernBERT-base
---

# ItColBERT

Monolingual **Italian late-interaction (ColBERT)** retriever for semantic search and RAG.

Built with [PyLate](https://github.com/lightonai/pylate) from [`DeepMount00/Italian-ModernBERT-base`](https://huggingface.co/DeepMount00/Italian-ModernBERT-base).

## Why this model

At release time there was no dedicated Italian-only ColBERT. Multilingual late-interaction models cover Italian, but this checkpoint is specialized on Italian IR data with a true MaxSim multi-vector architecture.

## Architecture

- Backbone: Italian ModernBERT base
- Projection: Dense `768 -> 128`
- Similarity: MaxSim (late interaction)
- Query length: 32 tokens
- Document length: 256 tokens

## Training

Two-phase recipe (ColBERT-Zero / GTE-ModernColBERT style, without unsupervised mega-pretrain):

1. **Supervised contrastive** (`CachedContrastive`, temperature 0.02) on:
   - [`unicamp-dl/mmarco`](https://huggingface.co/datasets/unicamp-dl/mmarco) Italian triples (joined from official TSV files)
   - [`nickprock/it-wiki-retrieval-synthetic-hn`](https://huggingface.co/datasets/nickprock/it-wiki-retrieval-synthetic-hn)
2. **Knowledge distillation** (`Distillation` / KL) on:
   - [`lightonai/embeddings-fine-tuning-filtered-it`](https://huggingface.co/datasets/lightonai/embeddings-fine-tuning-filtered-it) using cross-encoder `rerank_scores`
   - [`hotchpotch/mmarco-hard-negatives-reranker-filtered`](https://huggingface.co/datasets/hotchpotch/mmarco-hard-negatives-reranker-filtered) (`italian-hard-negatives`, `bge-reranker-v2-m3` scores)

   The KD budget is spent proportionally across all LightOn splits. Earlier
   checkpoints used a sequential budget that landed entirely on `msmarco_it`; the
   resulting "more KD helps long documents" claim was a data-composition
   artifact, not a scaling effect.

Checkpoints are selected on pooled MLDR-it nDCG@10 and mMARCO-it MRR@10, not on
hold-out KD KL divergence.

## Usage

```python
from pylate import models, rank

model = models.ColBERT("PATH_OR_HUB_ID")

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

- Optimized for Italian; cross-lingual retrieval is not the focus
- mMARCO Italian is machine-translated; wiki HN and LightOn KD help with native / higher-quality signal
- Index size is larger than dense bi-encoders (multi-vector)

## Citation

If you use this model, please also cite ColBERT, PyLate, mMARCO, and Italian-ModernBERT.
