"""italian ir dataset loaders for pylate contrastive and distillation training."""

from __future__ import annotations

import logging
import math
from pathlib import Path

from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from huggingface_hub import hf_hub_download
from pylate import utils

logger = logging.getLogger(__name__)

KD_SPLITS = (
    "msmarco_it",
    "nq_it",
    "hotpotqa_it",
    "fever_it",
    "trivia_it",
    "squadv2_it",
    "fiqa_it",
    "mldr_it",
)

# unicamp-dl/mmarco still ships a dataset script; datasets>=5 rejects scripts,
# so we join the published tsv files ourselves (same logic as mmarco.py).
MMARCO_REPO = "unicamp-dl/mmarco"
MMARCO_TRIPLES = "data/triples.train.ids.small.tsv"
MMARCO_IT_COLLECTION = "data/google/collections/italian_collection.tsv"
MMARCO_IT_QUERIES = "data/google/queries/train/italian_queries.train.tsv"

# mmarco-it hard negatives already scored by bge-reranker-v2-m3 (option b kd)
MMARCO_HN_KD_REPO = "hotchpotch/mmarco-hard-negatives-reranker-filtered"
MMARCO_HN_KD_CONFIG = "italian-hard-negatives"


def _download_mmarco_file(filename: str) -> Path:
    return Path(
        hf_hub_download(
            repo_id=MMARCO_REPO,
            filename=filename,
            repo_type="dataset",
        )
    )


def load_mmarco_italian(
    max_samples: int | None = 2_000_000,
    seed: int = 42,
    split: str = "train",
) -> Dataset:
    """load italian mmarco triplets as query/positive/negative columns."""
    if split != "train":
        raise ValueError("only the train split is available from mmarco tsv joins")

    logger.info("downloading unicamp-dl/mmarco italian tsv files...")
    triples_path = _download_mmarco_file(MMARCO_TRIPLES)
    collection_path = _download_mmarco_file(MMARCO_IT_COLLECTION)
    queries_path = _download_mmarco_file(MMARCO_IT_QUERIES)

    logger.info("indexing italian collection and queries...")
    collection: dict[str, str] = {}
    with collection_path.open(encoding="utf-8") as f:
        for line in f:
            doc_id, doc = line.rstrip("\n").split("\t", 1)
            collection[doc_id] = doc

    queries: dict[str, str] = {}
    with queries_path.open(encoding="utf-8") as f:
        for line in f:
            query_id, query = line.rstrip("\n").split("\t", 1)
            queries[query_id] = query

    out_q, out_p, out_n = [], [], []
    skipped = 0
    with triples_path.open(encoding="utf-8") as f:
        for line in f:
            if max_samples is not None and len(out_q) >= max_samples:
                break
            query_id, pos_id, neg_id = line.rstrip("\n").split("\t")
            q = queries.get(query_id)
            pos = collection.get(pos_id)
            neg = collection.get(neg_id)
            if q is None or pos is None or neg is None:
                skipped += 1
                continue
            out_q.append(q)
            out_p.append(pos)
            out_n.append(neg)

    if skipped:
        logger.warning("mmarco-it skipped %s triples with missing ids", skipped)

    ds = Dataset.from_dict(
        {"query": out_q, "positive": out_p, "negative": out_n}
    )
    if max_samples is not None and len(ds) > max_samples:
        ds = ds.shuffle(seed=seed).select(range(max_samples))
    elif seed is not None:
        ds = ds.shuffle(seed=seed)
    logger.info("mmarco-it ready: %s examples", len(ds))
    return ds


def load_wiki_hn_italian(
    max_hard_negatives: int = 2,
    seed: int = 42,
) -> Dataset:
    """load native italian wiki retrieval pairs with hard negatives expanded to triplets."""
    logger.info("loading nickprock/it-wiki-retrieval-synthetic-hn...")
    ds = load_dataset("nickprock/it-wiki-retrieval-synthetic-hn", split="train")

    query_col = "query" if "query" in ds.column_names else "anchor"
    hn_col = (
        "hard_negatives"
        if "hard_negatives" in ds.column_names
        else ("negatives" if "negatives" in ds.column_names else None)
    )
    if hn_col is None:
        raise KeyError(
            f"wiki hn dataset missing hard negatives column; got {ds.column_names}"
        )

    queries, positives, negatives = [], [], []
    for row in ds:
        query = row[query_col]
        positive = row["positive"]
        hard = row[hn_col] or []
        if isinstance(hard, str):
            hard = [hard]
        hard = [h for h in hard if h][:max_hard_negatives]
        for neg in hard:
            queries.append(query)
            positives.append(positive)
            negatives.append(neg)

    out = Dataset.from_dict(
        {"query": queries, "positive": positives, "negative": negatives}
    ).shuffle(seed=seed)
    logger.info("wiki-hn ready: %s triplet examples", len(out))
    return out


def build_phase1_dataset(
    mmarco_samples: int = 2_000_000,
    include_wiki_hn: bool = True,
    wiki_max_hard_negatives: int = 2,
    eval_samples: int = 2_000,
    seed: int = 42,
) -> tuple[Dataset, Dataset]:
    """build contrastive train/eval splits for phase 1."""
    parts: list[Dataset] = [
        load_mmarco_italian(max_samples=mmarco_samples, seed=seed)
    ]
    if include_wiki_hn:
        parts.append(
            load_wiki_hn_italian(
                max_hard_negatives=wiki_max_hard_negatives,
                seed=seed,
            )
        )
    combined = concatenate_datasets(parts).shuffle(seed=seed)
    n_eval = min(eval_samples, max(1, len(combined) // 100))
    split = combined.train_test_split(test_size=n_eval, seed=seed)
    train_ds, eval_ds = split["train"], split["test"]
    logger.info(
        "phase1 dataset: %s train / %s eval",
        len(train_ds),
        len(eval_ds),
    )
    return train_ds, eval_ds


def load_kd_italian(
    max_samples: int | None = None,
    seed: int = 42,
    splits: tuple[str, ...] = KD_SPLITS,
    n_ways: int = 11,
    use_rerank_scores: bool = True,
) -> Dataset:
    """
    load lighton italian kd dataset for pylate Distillation.

    returns rows with: query (str), documents (list[str]), scores (list[float])
    teacher labels prefer cross-encoder `rerank_scores` when available.
    """
    logger.info("loading lightonai/embeddings-fine-tuning-filtered-it...")
    scores_dd: DatasetDict = load_dataset(
        "lightonai/embeddings-fine-tuning-filtered-it", "scores"
    )
    queries_dd: DatasetDict = load_dataset(
        "lightonai/embeddings-fine-tuning-filtered-it", "queries"
    )
    documents_dd: DatasetDict = load_dataset(
        "lightonai/embeddings-fine-tuning-filtered-it", "documents"
    )

    # if only a small sample is needed, prefer the largest split first
    ordered = [s for s in splits if s in scores_dd]
    if max_samples is not None and max_samples <= 10_000:
        ordered = ["msmarco_it"] if "msmarco_it" in scores_dd else ordered[:1]

    prepared_parts: list[Dataset] = []
    remaining = max_samples
    for split in ordered:
        scores = scores_dd[split]
        if remaining is not None:
            take = min(remaining, len(scores))
            scores = scores.shuffle(seed=seed).select(range(take))

        queries = queries_dd[split].rename_columns({"query": "text"})
        documents = documents_dd[split].rename_columns({"document": "text"})

        if use_rerank_scores and "rerank_scores" in scores.column_names:
            drop = [c for c in ("scores",) if c in scores.column_names]
            scores = scores.remove_columns(drop).rename_columns(
                {"rerank_scores": "scores"}
            )
        elif "rerank_scores" in scores.column_names:
            scores = scores.remove_columns(["rerank_scores"])

        processor = utils.KDProcessing(
            queries=queries,
            documents=documents,
            n_ways=n_ways,
        )
        mapped = scores.map(
            processor.map,
            remove_columns=[
                c
                for c in scores.column_names
                if c not in {"query", "documents", "scores"}
            ],
            desc=f"kd-join-{split}",
        )
        # keep only the fields pylate distillation needs
        keep = [c for c in ("query", "documents", "scores") if c in mapped.column_names]
        mapped = mapped.select_columns(keep)
        prepared_parts.append(mapped)
        logger.info("kd split %s: %s examples", split, len(mapped))

        if remaining is not None:
            remaining -= len(mapped)
            if remaining <= 0:
                break

    if not prepared_parts:
        raise RuntimeError("no kd splits could be prepared")

    prepared = concatenate_datasets(prepared_parts).shuffle(seed=seed)
    if max_samples is not None and max_samples < len(prepared):
        prepared = prepared.select(range(max_samples))
    logger.info("kd italian ready: %s examples", len(prepared))
    return prepared


def load_mmarco_hn_kd_italian(
    max_samples: int | None = None,
    seed: int = 42,
    n_ways: int = 11,
    dataset_id: str = MMARCO_HN_KD_REPO,
    config_name: str = MMARCO_HN_KD_CONFIG,
) -> Dataset:
    """convert mmarco-it ce-scored hard negatives into pylate kd rows.

    source: hotchpotch/mmarco-hard-negatives-reranker-filtered (italian-hard-negatives).
    teacher scores are from BAAI/bge-reranker-v2-m3 (already in the dataset).

    returns rows with: query, documents (pos + negs), scores (aligned floats).
    """
    logger.info("loading %s [%s]...", dataset_id, config_name)
    ds = load_dataset(dataset_id, config_name, split="train")
    if max_samples is not None and max_samples < len(ds):
        ds = ds.shuffle(seed=seed).select(range(max_samples))

    def _to_kd(batch: dict) -> dict:
        queries: list[str] = []
        documents: list[list[str]] = []
        scores: list[list[float]] = []
        pad_doc = "documento non pertinente."
        # bge-reranker stores sigmoid-like probs in [0,1]; pylate Distillation
        # expects logits (docstring + colbertv2: log_softmax on teacher scores).
        eps = 1e-6

        def _prob_to_logit(p: float) -> float:
            x = min(max(float(p), eps), 1.0 - eps)
            return math.log(x / (1.0 - x))

        for q, pos, negs, pos_s, neg_s in zip(
            batch["query"],
            batch["pos_text"],
            batch["negs_text"],
            batch["pos_score"],
            batch["negs_score"],
            strict=True,
        ):
            neg_texts = list(negs or [])
            neg_scores = [float(x) for x in (neg_s or [])]
            # keep score/text alignment if lists differ
            n = min(len(neg_texts), len(neg_scores))
            docs = [pos] + neg_texts[:n]
            # convert ce probs → logits so scale matches lighton mxbai-style labels
            sc = [_prob_to_logit(pos_s)] + [_prob_to_logit(s) for s in neg_scores[:n]]
            if len(docs) < 2:
                continue
            # pylate collator stacks scores into a rectangular tensor → fixed n_ways
            if len(docs) > n_ways:
                docs = docs[:n_ways]
                sc = sc[:n_ways]
            while len(docs) < n_ways:
                docs.append(pad_doc)
                sc.append(_prob_to_logit(eps))
            queries.append(q)
            documents.append(docs)
            scores.append(sc)
        return {"query": queries, "documents": documents, "scores": scores}

    mapped = ds.map(
        _to_kd,
        batched=True,
        remove_columns=ds.column_names,
        desc="mmarco-hn-to-kd-logit11",
        load_from_cache_file=False,
    )
    mapped = mapped.shuffle(seed=seed)
    logger.info("mmarco-hn kd ready: %s examples (n_ways<=%s)", len(mapped), n_ways)
    return mapped


def _holdout_split(
    prepared: Dataset,
    eval_samples: int,
    seed: int,
    max_train_samples: int | None,
    exclude_eval_from_train: bool,
) -> tuple[Dataset, Dataset | None]:
    """optional kd hold-out for kl monitoring."""
    n_eval = int(eval_samples or 0)
    if n_eval <= 0:
        if max_train_samples is not None and len(prepared) > max_train_samples:
            prepared = prepared.select(range(max_train_samples))
        return prepared, None
    if n_eval >= len(prepared):
        raise ValueError(
            f"kd_eval_samples={n_eval} >= prepared size {len(prepared)}"
        )

    if not exclude_eval_from_train:
        eval_ds = prepared.shuffle(seed=seed + 123).select(range(n_eval))
        train_ds = prepared
        if max_train_samples is not None and len(train_ds) > max_train_samples:
            train_ds = train_ds.select(range(max_train_samples))
        logger.info(
            "kd eval monitor: %s examples (may overlap train; train size=%s)",
            len(eval_ds),
            len(train_ds),
        )
        return train_ds, eval_ds

    split = prepared.train_test_split(test_size=n_eval, seed=seed)
    train_ds, eval_ds = split["train"], split["test"]
    if max_train_samples is not None and len(train_ds) > max_train_samples:
        train_ds = train_ds.select(range(max_train_samples))
    logger.info(
        "kd split: %s train / %s hold-out (never in train)",
        len(train_ds),
        len(eval_ds),
    )
    return train_ds, eval_ds


def load_kd_italian_train_eval(
    max_samples: int | None = None,
    eval_samples: int = 0,
    seed: int = 42,
    splits: tuple[str, ...] = KD_SPLITS,
    n_ways: int = 11,
    use_rerank_scores: bool = True,
    exclude_eval_from_train: bool = True,
    include_mmarco_hn: bool = False,
    mmarco_hn_max_samples: int | None = None,
    mmarco_hn_dataset: str = MMARCO_HN_KD_REPO,
    mmarco_hn_config: str = MMARCO_HN_KD_CONFIG,
) -> tuple[Dataset, Dataset | None]:
    """load lighton (+ optional mmarco hn) kd and optionally hold out for kl monitoring.

    if exclude_eval_from_train is false (e.g. mid-run resume), train stays the full
    set so step counts match the original job; eval may overlap train.

    when include_mmarco_hn is true, max_samples caps the lighton budget only;
    mmarco_hn_max_samples caps the mmarco hn budget (none = all rows).
    """
    parts: list[Dataset] = []

    # lighton: reserve eval rows only when lighton-only and excluding eval from train
    lighton_need = max_samples
    if (
        not include_mmarco_hn
        and eval_samples
        and max_samples is not None
        and exclude_eval_from_train
    ):
        lighton_need = max_samples + eval_samples

    load_lighton = not (include_mmarco_hn and max_samples == 0)
    if load_lighton:
        parts.append(
            load_kd_italian(
                max_samples=lighton_need,
                seed=seed,
                splits=splits,
                n_ways=n_ways,
                use_rerank_scores=use_rerank_scores,
            )
        )

    if include_mmarco_hn:
        parts.append(
            load_mmarco_hn_kd_italian(
                max_samples=mmarco_hn_max_samples,
                seed=seed + 17,
                n_ways=n_ways,
                dataset_id=mmarco_hn_dataset,
                config_name=mmarco_hn_config,
            )
        )

    if not parts:
        raise RuntimeError("no kd sources enabled")

    prepared = (
        parts[0]
        if len(parts) == 1
        else concatenate_datasets(parts).shuffle(seed=seed)
    )
    if include_mmarco_hn:
        logger.info(
            "kd mix ready: %s total examples (lighton_cap=%s, mmarco_hn_cap=%s)",
            len(prepared),
            max_samples,
            mmarco_hn_max_samples,
        )

    # for mix, max_samples already applied per-source; do not re-cap the concat
    train_cap = None if include_mmarco_hn else max_samples
    return _holdout_split(
        prepared,
        eval_samples=eval_samples,
        seed=seed,
        max_train_samples=train_cap,
        exclude_eval_from_train=exclude_eval_from_train,
    )


def build_mmarco_eval_triplet(
    n_samples: int = 2000,
    seed: int = 42,
) -> Dataset:
    """small held-out italian mmarco slice for quick triplet accuracy checks."""
    ds = load_mmarco_italian(max_samples=n_samples + 1000, seed=seed + 7)
    return ds.select(range(min(n_samples, len(ds))))
