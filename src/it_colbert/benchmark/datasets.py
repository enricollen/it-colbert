"""load italian retrieval benchmarks: mmarco-it and mldr-it."""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import hf_hub_download

logger = logging.getLogger(__name__)

MMARCO_REPO = "unicamp-dl/mmarco"
MMARCO_IT_COLLECTION = "data/google/collections/italian_collection.tsv"
MMARCO_IT_QUERIES_DEV = "data/google/queries/dev/italian_queries.dev.small.tsv"
# ms marco qrels are language-agnostic (same passage ids)
MMARCO_QRELS_DEV = "data/qrels.dev.small.tsv"

IR_METRICS = [
    "mrr@10",
    "ndcg@1",
    "ndcg@5",
    "ndcg@10",
    "ndcg@100",
    "map@100",
    "recall@1",
    "recall@5",
    "recall@10",
    "recall@100",
    "precision@10",
    "hits@1",
    "hits@5",
    "hits@10",
]


@dataclass
class RetrievalSplit:
    name: str
    documents: list[dict]  # {"id": str, "text": str}
    queries: dict[str, str]  # qid -> text
    qrels: dict[str, dict[str, int]]  # qid -> {docid: rel}


def _download_mmarco(filename: str) -> Path:
    return Path(
        hf_hub_download(
            repo_id=MMARCO_REPO,
            filename=filename,
            repo_type="dataset",
        )
    )


def load_mmarco_italian_dev(
    max_corpus_docs: int | None = 200_000,
    max_queries: int | None = None,
    seed: int = 42,
) -> RetrievalSplit:
    """load mmarco-it official-style dev retrieval set.

    protocol notes:
    - queries: italian google-translated ms marco dev.small queries
    - qrels: official ms marco qrels.dev.small (passage ids shared)
    - corpus: all qrel-relevant passages + random sample up to max_corpus_docs

    full 8.8m collection is impractical for multi-model colbert indexing on one
    24gb gpu overnight; the pooled corpus keeps relative comparisons fair and
    still reports standard ir metrics (mrr@10, ndcg@10, recall@k).
    """
    logger.info("loading mmarco-it collection / queries / qrels...")
    collection_path = _download_mmarco(MMARCO_IT_COLLECTION)
    queries_path = _download_mmarco(MMARCO_IT_QUERIES_DEV)
    qrels_path = _download_mmarco(MMARCO_QRELS_DEV)

    queries: dict[str, str] = {}
    with queries_path.open(encoding="utf-8") as f:
        for line in f:
            qid, text = line.rstrip("\n").split("\t", 1)
            queries[qid] = text

    qrels: dict[str, dict[str, int]] = {}
    needed_doc_ids: set[str] = set()
    with qrels_path.open(encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) == 4:
                qid, _, docid, rel = parts
            else:
                # sometimes space-separated
                qid, _, docid, rel = line.split()
            rel_i = int(rel)
            if rel_i <= 0:
                continue
            if qid not in queries:
                continue
            qrels.setdefault(qid, {})[docid] = rel_i
            needed_doc_ids.add(docid)

    # keep only queries that have at least one relevant
    queries = {qid: text for qid, text in queries.items() if qid in qrels}
    if max_queries is not None and len(queries) > max_queries:
        rng = random.Random(seed)
        keep = set(rng.sample(sorted(queries.keys()), max_queries))
        queries = {qid: queries[qid] for qid in keep}
        qrels = {qid: qrels[qid] for qid in keep}
        needed_doc_ids = {d for qid in keep for d in qrels[qid]}

    # stream collection once: always keep qrel docs, sample the rest
    rng = random.Random(seed)
    docs_by_id: dict[str, str] = {}
    reservoir: list[tuple[str, str]] = []
    n_seen_extra = 0
    with collection_path.open(encoding="utf-8") as f:
        for line in f:
            doc_id, text = line.rstrip("\n").split("\t", 1)
            if doc_id in needed_doc_ids:
                docs_by_id[doc_id] = text
                continue
            if max_corpus_docs is None:
                docs_by_id[doc_id] = text
                continue
            # reservoir sample for distractors
            budget = max(0, max_corpus_docs - len(needed_doc_ids))
            if budget <= 0:
                continue
            n_seen_extra += 1
            if len(reservoir) < budget:
                reservoir.append((doc_id, text))
            else:
                j = rng.randrange(n_seen_extra)
                if j < budget:
                    reservoir[j] = (doc_id, text)

    for doc_id, text in reservoir:
        docs_by_id[doc_id] = text

    missing = needed_doc_ids - docs_by_id.keys()
    if missing:
        logger.warning("mmarco-it missing %s qrel docs from collection", len(missing))

    documents = [{"id": did, "text": txt} for did, txt in docs_by_id.items()]
    logger.info(
        "mmarco-it ready: %s queries, %s docs (qrel=%s), qrels=%s",
        len(queries),
        len(documents),
        len(needed_doc_ids),
        sum(len(v) for v in qrels.values()),
    )
    return RetrievalSplit(
        name="mmarco-it-dev",
        documents=documents,
        queries=queries,
        qrels=qrels,
    )


def load_mldr_italian(
    split: str = "test",
    max_doc_chars: int | None = 12_000,
) -> RetrievalSplit:
    """load mldr italian retrieval split (wikipedia long-doc).

    primary italian nDCG@10 / recall benchmark with a tractable ~10k corpus.
    downloads raw jsonl/tsv from Shitao/MLDR (avoids deprecated dataset scripts).
    """
    import gzip
    import json

    logger.info("loading Shitao/MLDR italian split=%s via jsonl...", split)
    queries_path = Path(
        hf_hub_download(
            repo_id="Shitao/MLDR",
            filename=f"mldr-v1.0-it/{split}.jsonl.gz",
            repo_type="dataset",
        )
    )
    corpus_path = Path(
        hf_hub_download(
            repo_id="Shitao/MLDR",
            filename="mldr-v1.0-it/corpus.jsonl.gz",
            repo_type="dataset",
        )
    )
    qrels_path = Path(
        hf_hub_download(
            repo_id="Shitao/MLDR",
            filename=f"qrels/qrels.mldr-v1.0-it-{split}.tsv",
            repo_type="dataset",
        )
    )

    documents: list[dict] = []
    with gzip.open(corpus_path, "rt", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            doc_id = str(row.get("docid") or row.get("id") or row.get("_id"))
            text = row.get("text") or ""
            title = row.get("title") or ""
            full = f"{title} {text}".strip() if title else text
            if max_doc_chars is not None and len(full) > max_doc_chars:
                full = full[:max_doc_chars]
            documents.append({"id": doc_id, "text": full})

    queries: dict[str, str] = {}
    with gzip.open(queries_path, "rt", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            qid = str(row.get("query_id") or row.get("qid") or row.get("_id"))
            qtext = row.get("query") or row.get("text") or ""
            queries[qid] = qtext

    qrels: dict[str, dict[str, int]] = {}
    with qrels_path.open(encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4 or parts[0] in {"query-id", "qid"}:
                continue
            qid, _, docid, rel = parts[0], parts[1], parts[2], parts[3]
            rel_i = int(float(rel))
            if rel_i <= 0:
                continue
            qrels.setdefault(qid, {})[docid] = rel_i

    queries = {qid: t for qid, t in queries.items() if qid in qrels and qrels[qid]}
    logger.info(
        "mldr-it ready: %s queries, %s docs, qrels=%s",
        len(queries),
        len(documents),
        sum(len(v) for v in qrels.values()),
    )
    return RetrievalSplit(
        name=f"mldr-it-{split}",
        documents=documents,
        queries=queries,
        qrels=qrels,
    )
