"""italian ir dataset loaders for pylate contrastive and distillation training."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import shutil
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

# italian long-document retrieval supervision (TODO.md §10.6). wikipedia-derived,
# ungated, cc-by-sa-4.0/gfdl. 650,885 rows: median ~1,033 tokens per document,
# 98.2% over 512 tokens, and the query's evidence span starts past the
# 512-token mark in 30.7% of rows — which is the length lever being tested.
# revision is pinned so the mixture stays reproducible from the model card.
LONGDOC_REPO = "hotchpotch/wikipedia-multilingual-synthetic-ir-query"
LONGDOC_CONFIG = "it-long_doc"
LONGDOC_REVISION = "5089a24f75a297ecacfe51fb88a0b5138c5081aa"
# the generator emits several query styles. `title` is near-verbatim the article's
# own first line, and `summary` / `keywords` / `synonym_keywords` are not
# search-shaped, so only the question-like styles are kept as supervision.
LONGDOC_KEEP_TYPES = ("query", "faq", "alt_query")
# char cap before tokenization, matching load_mldr_italian's own default. at
# 12k chars a 1024-token window is never starved and tokenizing 32k-char
# articles down to 1024 tokens is avoided.
LONGDOC_MAX_CHARS = 12_000

_LONGDOC_DROP = re.compile(r"[^0-9a-z\u00e0-\u00ff\s]+")
_LONGDOC_WS = re.compile(r"\s+")


def longdoc_normalize(text: str) -> str:
    """lowercase alphanumeric word stream.

    absorbs the markdown headings and punctuation differences between the
    long-doc training text and the mldr-it benchmark corpus, so the same
    wikipedia article normalizes identically on both sides.
    """
    return _LONGDOC_WS.sub(" ", _LONGDOC_DROP.sub(" ", text.casefold())).strip()


def longdoc_doc_key(text: str) -> str:
    """stable article identity, shared by this loader and check_longdoc_overlap.py.

    text-derived on purpose: the exclusion list stays valid across shard
    layouts, row orders and dataset revisions.
    """
    return hashlib.sha1(longdoc_normalize(text).encode("utf-8")).hexdigest()[:16]


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


def load_mmarco_hn_contrastive_italian(
    max_samples: int | None = None,
    negatives_per_query: int = 4,
    seed: int = 42,
    dataset_id: str = MMARCO_HN_KD_REPO,
    config_name: str = MMARCO_HN_KD_CONFIG,
    false_negative_margin: float = 0.1,
    skip_hardest: int = 1,
) -> Dataset:
    """reranker-mined mmarco-it hard negatives as contrastive triplets.

    the official `triples.train.ids.small` negatives are BM25-sampled and easy;
    these are mined by a dense retriever and re-scored by bge-reranker-v2-m3, so
    they sit much closer to the positive and give the contrastive loss real work.

    `false_negative_margin` drops negatives the teacher scores within that margin
    of the positive (they are usually unlabelled positives), and `skip_hardest`
    drops the top-scoring negatives outright, which is the standard guard against
    training on mislabelled mmarco passages.
    """
    logger.info("loading %s [%s] as contrastive triplets...", dataset_id, config_name)
    ds = load_dataset(dataset_id, config_name, split="train")
    if max_samples is not None and max_samples < len(ds):
        ds = ds.shuffle(seed=seed).select(range(max_samples))

    queries, positives, negatives = [], [], []
    dropped_fn = 0
    for row in ds:
        q, pos = row["query"], row["pos_text"]
        pos_score = float(row["pos_score"])
        negs = list(row["negs_text"] or [])
        neg_scores = [float(x) for x in (row["negs_score"] or [])]
        n = min(len(negs), len(neg_scores))
        # hardest first, then drop the ones too close to the positive
        ranked = sorted(zip(negs[:n], neg_scores[:n]), key=lambda x: -x[1])
        ranked = ranked[skip_hardest:]
        kept = []
        for text, score in ranked:
            if score >= pos_score - false_negative_margin:
                dropped_fn += 1
                continue
            kept.append(text)
            if len(kept) >= negatives_per_query:
                break
        for neg in kept:
            queries.append(q)
            positives.append(pos)
            negatives.append(neg)

    if dropped_fn:
        logger.info("mmarco-hn contrastive: dropped %s likely false negatives", dropped_fn)
    out = Dataset.from_dict(
        {"query": queries, "positive": positives, "negative": negatives}
    ).shuffle(seed=seed)
    logger.info("mmarco-hn contrastive ready: %s triplets", len(out))
    return out


# native-ish italian retrieval sets in the tevatron layout, used to widen phase 1
# beyond machine-translated mmarco (the init checkpoint is already a mmarco
# specialist, so more mmarco mostly reinforces what the model can already do)
ITALIAN_CONTRASTIVE_SOURCES = (
    ("yuri-no/miracl-ita-argos", "train"),
    ("yuri-no/squad-ita", "train"),
)


def load_tevatron_style_contrastive_italian(
    dataset_id: str,
    split: str = "train",
    max_samples: int | None = None,
    negatives_per_query: int = 2,
    seed: int = 42,
) -> Dataset:
    """triplets from a dataset with positive_passages / negative_passages lists."""
    logger.info("loading %s [%s] as contrastive triplets...", dataset_id, split)
    ds = load_dataset(dataset_id, split=split)
    if max_samples is not None and max_samples < len(ds):
        ds = ds.shuffle(seed=seed).select(range(max_samples))

    def _text(passage: dict) -> str:
        title = (passage.get("title") or "").strip()
        body = (passage.get("text") or "").strip()
        return f"{title} {body}".strip() if title else body

    queries, positives, negatives = [], [], []
    for row in ds:
        pos_list = row.get("positive_passages") or []
        neg_list = row.get("negative_passages") or []
        if not pos_list or not neg_list:
            continue
        positive = _text(pos_list[0])
        for passage in neg_list[:negatives_per_query]:
            negative = _text(passage)
            if not negative:
                continue
            queries.append(row["query"])
            positives.append(positive)
            negatives.append(negative)

    out = Dataset.from_dict(
        {"query": queries, "positive": positives, "negative": negatives}
    ).shuffle(seed=seed)
    logger.info("%s ready: %s triplets", dataset_id, len(out))
    return out


def load_wikipedia_longdoc_italian(
    max_samples: int | None = None,
    keep_instruction_types: tuple[str, ...] | None = LONGDOC_KEEP_TYPES,
    exclude_path: str | None = None,
    max_chars: int | None = LONGDOC_MAX_CHARS,
    seed: int = 42,
) -> Dataset:
    """long-document italian triplets from it-long_doc.

    the documents are whole wikipedia articles and the query is generated from
    one span inside them, so at document_length 512 a third of these examples
    supervise a match the model cannot see. that truncation is deliberate and
    is left to the tokenizer: cropping around `query_source_text_start` would
    erase the very signal the 512-vs-1024 ablation measures.

    the source ships no negatives, so each query is paired with another
    article from the same subsample. in-batch negatives at batch 512 carry the
    contrastive signal; this only fills the `negative` column the other phase-1
    sources already use.
    """
    logger.info(
        "loading %s [%s] rev %s...", LONGDOC_REPO, LONGDOC_CONFIG, LONGDOC_REVISION[:8]
    )
    ds = load_dataset(
        LONGDOC_REPO,
        name=LONGDOC_CONFIG,
        split="train",
        revision=LONGDOC_REVISION,
    )
    n_raw = len(ds)

    if keep_instruction_types:
        keep = set(keep_instruction_types)
        ds = ds.filter(
            lambda batch: [t in keep for t in batch["instruction_type"]],
            batched=True,
            batch_size=10_000,
            desc="filtering instruction_type",
        )
        logger.info(
            "longdoc instruction_type filter %s: %s -> %s rows",
            sorted(keep),
            f"{n_raw:,}",
            f"{len(ds):,}",
        )

    excluded_keys: set[str] = set()
    if exclude_path:
        path = Path(exclude_path)
        if not path.exists():
            raise FileNotFoundError(
                f"longdoc exclusion list {path} not found; "
                "run scripts/check_longdoc_overlap.py first"
            )
        excluded_keys = {
            line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
        }
        logger.info(
            "longdoc exclusion list: %s article keys from %s",
            f"{len(excluded_keys):,}",
            path,
        )

    if excluded_keys:
        n_before = len(ds)
        ds = ds.filter(
            lambda batch: [
                longdoc_doc_key(doc) not in excluded_keys for doc in batch["document"]
            ],
            batched=True,
            batch_size=2_000,
            desc="dropping mldr-overlapping articles",
        )
        logger.info(
            "longdoc mldr overlap filter: %s -> %s rows (%s dropped)",
            f"{n_before:,}",
            f"{len(ds):,}",
            f"{n_before - len(ds):,}",
        )

    ds = ds.shuffle(seed=seed)
    if max_samples is not None and max_samples < len(ds):
        ds = ds.select(range(max_samples))

    types_seen: dict[str, int] = {}
    queries: list[str] = []
    positives: list[str] = []
    for row in ds:
        document = row["document"] or ""
        query = row["query"] or ""
        if not document or not query:
            continue
        if max_chars is not None and len(document) > max_chars:
            document = document[:max_chars]
        queries.append(query)
        positives.append(document)
        style = row.get("instruction_type") or "?"
        types_seen[style] = types_seen.get(style, 0) + 1

    n = len(positives)
    if n < 2:
        raise RuntimeError(f"longdoc source produced {n} usable rows")

    # rotate the positives by half the corpus to get one negative per query.
    # the fingerprint walk covers the case where a rotation lines up two rows
    # generated from the same article.
    fingerprints = [hash(text) for text in positives]
    offset = max(1, n // 2)
    negatives: list[str] = []
    for i in range(n):
        j = (i + offset) % n
        guard = 0
        while fingerprints[j] == fingerprints[i] and guard < 8:
            j = (j + 1) % n
            guard += 1
        negatives.append(positives[j])

    out = Dataset.from_dict(
        {"query": queries, "positive": positives, "negative": negatives}
    ).shuffle(seed=seed)

    doc_chars = sorted(len(text) for text in positives)
    logger.info(
        "longdoc ready: %s triplets | styles %s | doc chars p50=%s p90=%s max=%s",
        f"{len(out):,}",
        {k: types_seen[k] for k in sorted(types_seen)},
        doc_chars[n // 2],
        doc_chars[int(0.9 * n)],
        doc_chars[-1],
    )
    return out


def load_mined_hard_negatives(
    path: str,
    negatives_per_query: int = 4,
    seed: int = 42,
) -> Dataset:
    """load triplets mined by `scripts/mine_hard_negatives.py`.

    round-2 self-mined negatives (retrieve with your own checkpoint, train on
    what it wrongly ranks high) are the standard ColBERTv2 loop and consistently
    beat any static negative pack.
    """
    logger.info("loading mined hard negatives from %s...", path)
    ds = Dataset.load_from_disk(path)
    queries, positives, negatives = [], [], []
    for row in ds:
        for negative in (row["negatives"] or [])[:negatives_per_query]:
            queries.append(row["query"])
            positives.append(row["positive"])
            negatives.append(negative)
    out = Dataset.from_dict(
        {"query": queries, "positive": positives, "negative": negatives}
    ).shuffle(seed=seed)
    logger.info("mined hard negatives ready: %s triplets", len(out))
    return out


def _cache_key(**parts) -> str:
    """short stable digest of the arguments that define a built dataset."""
    payload = json.dumps(parts, sort_keys=True, default=str)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def build_phase1_dataset(
    mmarco_samples: int = 2_000_000,
    include_wiki_hn: bool = True,
    wiki_max_hard_negatives: int = 2,
    eval_samples: int = 2_000,
    seed: int = 42,
    include_mmarco_hn: bool = False,
    mmarco_hn_samples: int | None = None,
    mmarco_hn_negatives_per_query: int = 4,
    include_italian_sources: bool = False,
    italian_source_max_samples: int | None = None,
    include_longdoc: bool = False,
    longdoc_samples: int | None = None,
    longdoc_exclude_path: str | None = None,
    mined_negatives_path: str | None = None,
    mined_negatives_per_query: int = 4,
    cache_dir: str | None = "outputs/dataset_cache",
) -> tuple[Dataset, Dataset]:
    """build contrastive train/eval splits for phase 1.

    sources, all optional and mixed by concatenation:
    - `mmarco_samples`: official BM25-negative mmarco triples
    - `include_mmarco_hn`: reranker-mined mmarco hard negatives (much harder)
    - `include_wiki_hn`: native italian wiki retrieval pairs
    - `include_italian_sources`: MIRACL-ita / SQuAD-ita, to widen the domain
      past machine-translated mmarco
    - `include_longdoc`: it-long_doc whole wikipedia articles, the only source
      in the mixture whose documents exceed the truncation window
    - `mined_negatives_path`: round-2 negatives mined with your own checkpoint

    the result is cached under `cache_dir` keyed by these arguments. building it
    means indexing the full 8.8M-passage mmarco collection into memory, so
    without the cache every overnight stop/resume pays that cost again.
    """
    cache_path: Path | None = None
    if cache_dir:
        key = _cache_key(
            mmarco_samples=mmarco_samples,
            include_wiki_hn=include_wiki_hn,
            wiki_max_hard_negatives=wiki_max_hard_negatives,
            eval_samples=eval_samples,
            seed=seed,
            include_mmarco_hn=include_mmarco_hn,
            mmarco_hn_samples=mmarco_hn_samples,
            mmarco_hn_negatives_per_query=mmarco_hn_negatives_per_query,
            include_italian_sources=include_italian_sources,
            italian_source_max_samples=italian_source_max_samples,
            include_longdoc=include_longdoc,
            longdoc_samples=longdoc_samples,
            longdoc_exclude_path=longdoc_exclude_path,
            mined_negatives_path=mined_negatives_path,
            mined_negatives_per_query=mined_negatives_per_query,
        )
        cache_path = Path(cache_dir) / f"phase1-{key}"
        if (cache_path / "train").exists() and (cache_path / "eval").exists():
            logger.info("reusing cached phase1 dataset at %s", cache_path)
            return (
                Dataset.load_from_disk(str(cache_path / "train")),
                Dataset.load_from_disk(str(cache_path / "eval")),
            )

    parts: list[Dataset] = []
    if mmarco_samples:
        parts.append(load_mmarco_italian(max_samples=mmarco_samples, seed=seed))
    if include_mmarco_hn:
        parts.append(
            load_mmarco_hn_contrastive_italian(
                max_samples=mmarco_hn_samples,
                negatives_per_query=mmarco_hn_negatives_per_query,
                seed=seed + 17,
            )
        )
    if include_italian_sources:
        for dataset_id, source_split in ITALIAN_CONTRASTIVE_SOURCES:
            parts.append(
                load_tevatron_style_contrastive_italian(
                    dataset_id=dataset_id,
                    split=source_split,
                    max_samples=italian_source_max_samples,
                    seed=seed + 23,
                )
            )
    if include_longdoc:
        parts.append(
            load_wikipedia_longdoc_italian(
                max_samples=longdoc_samples,
                exclude_path=longdoc_exclude_path,
                seed=seed + 29,
            )
        )
    if mined_negatives_path:
        parts.append(
            load_mined_hard_negatives(
                path=mined_negatives_path,
                negatives_per_query=mined_negatives_per_query,
                seed=seed + 31,
            )
        )
    if include_wiki_hn:
        parts.append(
            load_wiki_hn_italian(
                max_hard_negatives=wiki_max_hard_negatives,
                seed=seed,
            )
        )
    if not parts:
        raise RuntimeError("phase 1 dataset is empty; enable at least one source")
    combined = concatenate_datasets(parts).shuffle(seed=seed) if len(parts) > 1 else parts[0].shuffle(seed=seed)
    if eval_samples <= 0:
        logger.info("phase1 dataset: %s train / no eval split", len(combined))
        train_ds, eval_ds = combined, combined.select(range(0))
    else:
        n_eval = min(eval_samples, max(1, len(combined) // 100))
        split = combined.train_test_split(test_size=n_eval, seed=seed)
        train_ds, eval_ds = split["train"], split["test"]
        logger.info(
            "phase1 dataset: %s train / %s eval",
            len(train_ds),
            len(eval_ds),
        )

    if cache_path is not None:
        # write to a temp dir then rename, so an interrupt cannot leave a
        # half-written cache that later loads as a silently truncated dataset
        tmp = cache_path.with_name(cache_path.name + ".tmp")
        if tmp.exists():
            shutil.rmtree(tmp)
        train_ds.save_to_disk(str(tmp / "train"))
        eval_ds.save_to_disk(str(tmp / "eval"))
        tmp.rename(cache_path)
        logger.info("cached phase1 dataset at %s", cache_path)

    return train_ds, eval_ds


def _proportional_quotas(
    sizes: dict[str, int],
    budget: int,
    min_share: float = 0.0,
) -> dict[str, int]:
    """split a row budget across kd splits proportionally to their size.

    `min_share` guarantees every split at least that fraction of the budget
    (capped by its real size), so tiny but on-domain splits like mldr_it are not
    erased by a big one like msmarco_it. remainder goes back to the big splits.
    """
    quotas: dict[str, int] = {}
    floor = int(budget * min_share) if min_share > 0 else 0
    for name, size in sizes.items():
        quotas[name] = min(size, floor)
    left = budget - sum(quotas.values())
    if left <= 0:
        return quotas

    # distribute what is left proportionally to remaining capacity
    for _ in range(3):  # a couple of passes reclaim rows freed by exhausted splits
        capacity = {n: sizes[n] - quotas[n] for n in sizes if sizes[n] > quotas[n]}
        total = sum(capacity.values())
        if not capacity or left <= 0:
            break
        for name, cap in capacity.items():
            add = min(cap, int(left * cap / total))
            quotas[name] += add
        left = budget - sum(quotas.values())
    return quotas


def load_kd_italian(
    max_samples: int | None = None,
    seed: int = 42,
    splits: tuple[str, ...] = KD_SPLITS,
    n_ways: int = 11,
    use_rerank_scores: bool = True,
    split_sampling: str = "proportional",
    split_min_share: float = 0.02,
    split_quotas: dict[str, int] | None = None,
) -> Dataset:
    """
    load lighton italian kd dataset for pylate Distillation.

    returns rows with: query (str), documents (list[str]), scores (list[float])
    teacher labels prefer cross-encoder `rerank_scores` when available.

    when `max_samples` caps the budget, `split_sampling` decides how it is spent:
    - "proportional" (default): every split contributes, sized by its row count
      with a `split_min_share` floor. keeps long-doc splits (mldr_it, trivia_it)
      in the mix instead of spending the whole budget on msmarco_it.
    - "sequential": legacy behaviour — drain splits in `splits` order.
    `split_quotas` overrides both with explicit per-split row counts.
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

    ordered = [s for s in splits if s in scores_dd]
    if not ordered:
        raise RuntimeError(f"none of {splits} present in kd dataset")

    # per-split row budget
    quotas: dict[str, int] | None = None
    if split_quotas:
        quotas = {s: int(n) for s, n in split_quotas.items() if s in scores_dd and n > 0}
        ordered = [s for s in ordered if s in quotas]
    elif max_samples is not None and split_sampling == "proportional":
        quotas = _proportional_quotas(
            {s: len(scores_dd[s]) for s in ordered},
            budget=max_samples,
            min_share=split_min_share,
        )
    if quotas is not None:
        logger.info("kd split quotas: %s", quotas)

    prepared_parts: list[Dataset] = []
    remaining = None if quotas is not None else max_samples
    for split in ordered:
        scores = scores_dd[split]
        if quotas is not None:
            take = min(quotas.get(split, 0), len(scores))
            if take <= 0:
                continue
            scores = scores.shuffle(seed=seed).select(range(take))
        elif remaining is not None:
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
    # quota mode already spends the budget across splits; re-capping here would
    # just truncate the shuffled mix again for no reason
    if quotas is None and max_samples is not None and max_samples < len(prepared):
        prepared = prepared.select(range(max_samples))
    logger.info("kd italian ready: %s examples", len(prepared))
    return prepared


def load_mmarco_hn_kd_italian(
    max_samples: int | None = None,
    seed: int = 42,
    n_ways: int = 11,
    dataset_id: str = MMARCO_HN_KD_REPO,
    config_name: str = MMARCO_HN_KD_CONFIG,
    score_temperature: float = 1.0,
    score_clip: float | None = None,
) -> Dataset:
    """convert mmarco-it ce-scored hard negatives into pylate kd rows.

    source: hotchpotch/mmarco-hard-negatives-reranker-filtered (italian-hard-negatives).
    teacher scores are from BAAI/bge-reranker-v2-m3 (already in the dataset).

    `score_temperature` / `score_clip` shape how peaked the teacher distribution is
    after logit conversion. bge probs saturate near 0 and 1, so raw logits span
    ~[-13.8, +13.8] and softmax over them is nearly one-hot — far sharper than the
    lighton mxbai labels (~5-9). dividing by a temperature (or clipping the range)
    brings the two sources to comparable entropy so neither dominates the mixed KL.
    run `scripts/inspect_kd_scores.py` to pick the value.

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

        temp = float(score_temperature) if score_temperature else 1.0

        def _prob_to_logit(p: float) -> float:
            x = min(max(float(p), eps), 1.0 - eps)
            z = math.log(x / (1.0 - x))
            if score_clip is not None:
                z = min(max(z, -abs(score_clip)), abs(score_clip))
            return z / temp

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
    logger.info(
        "mmarco-hn kd ready: %s examples (n_ways<=%s, temp=%s, clip=%s)",
        len(mapped),
        n_ways,
        score_temperature,
        score_clip,
    )
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
    mmarco_hn_score_temperature: float = 1.0,
    mmarco_hn_score_clip: float | None = None,
    split_sampling: str = "proportional",
    split_min_share: float = 0.02,
    split_quotas: dict[str, int] | None = None,
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
                split_sampling=split_sampling,
                split_min_share=split_min_share,
                split_quotas=split_quotas,
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
                score_temperature=mmarco_hn_score_temperature,
                score_clip=mmarco_hn_score_clip,
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
