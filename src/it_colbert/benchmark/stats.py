"""per-query IR metrics and bootstrap confidence intervals.

why this exists: MLDR-it has 200 queries. the standard error on nDCG@10 at that
sample size is roughly +/-0.02-0.03, which means the previously reported
"fullkd 0.352 vs jina-colbert-v2 0.369" gap was never a real difference — it sat
inside the noise band. any ranking claim published without an interval is
contestable in the first comment under the model card.

the aggregate numbers still come from pylate/ranx; this module recomputes the
same metrics per query so they can be resampled. with binary qrels (which is what
MLDR-it, mMARCO-it, MIRACL-ita and SQuAD-ita all use) linear and exponential gain
definitions coincide, so the two implementations agree.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# fixed on purpose, and deliberately not the training seed: the halves have to
# stay put when a config changes `seed`, otherwise a rerun moves the boundary and
# a model selected under the old split gets reported on queries it was tuned on.
QUERY_SPLIT_SEED = 20260816
QUERY_HALVES = ("all", "selection", "report")


def _query_bucket(query_id: str, seed: int = QUERY_SPLIT_SEED) -> int:
    """stable 0/1 bucket for a single query id.

    hashed per id rather than by slicing a shuffled list, because the callers do
    not all see the same query set: training evaluates a sampled slice, the
    benchmark scores only queries with qrels, and a comparison keeps only the
    queries two models share. a list-relative split would move the boundary with
    the input and quietly hand a selection query back as a reporting query, which
    is the failure this whole mechanism exists to prevent.

    python's built-in `hash()` is salted per process and would give a different
    partition on every run, so it cannot be used here.
    """
    digest = hashlib.blake2b(
        str(query_id).encode("utf-8"),
        digest_size=8,
        key=str(seed).encode("utf-8"),
    ).digest()
    return int.from_bytes(digest, "big") % 2


def split_query_ids(
    query_ids: list[str],
    half: str = "all",
    seed: int = QUERY_SPLIT_SEED,
) -> list[str]:
    """deterministic disjoint halves of a query set, in the caller's order.

    training-time checkpoint selection and the reported number must not read the
    same queries. phase 2 picks its checkpoint by nDCG@10 on MLDR-it and
    MIRACL-ita, so reporting those metrics on those same queries publishes the
    winner's curse as a result (TODO.md §11.1). "selection" is what the trainer
    is allowed to see; "report" is what the headline is read from.

    membership depends only on the query id, so any subset of a set splits the
    same way the whole set does. the halves are therefore near-equal rather than
    exactly equal in size — on 200 MLDR queries expect roughly 100 +/- 7, which
    the bootstrap interval already accounts for.

    order is preserved rather than sorted so the result stays aligned with score
    lists that were built in the original query order.
    """
    if half == "all":
        return list(query_ids)
    if half not in QUERY_HALVES:
        raise ValueError(f"unknown query half {half!r}; expected one of {QUERY_HALVES}")
    want = 0 if half == "selection" else 1
    return [q for q in query_ids if _query_bucket(q, seed) == want]


def scored_query_ids(
    qrels: dict[str, dict[str, int]],
    queries: list[str],
) -> list[str]:
    """the subset of `queries` that `per_query_metrics` actually scores, in order.

    queries with no qrels are skipped there, so the metric lists come back shorter
    than the query list whenever a split has any. saving the unfiltered list
    alongside them would misalign scores and query ids, which breaks any later
    filtering by query id.
    """
    return [q for q in queries if qrels.get(q)]


def _dcg(relevances: list[int], k: int) -> float:
    """discounted cumulative gain over the first k ranked relevance grades."""
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances[:k]) if rel)


def ndcg_at_k(ranked_relevances: list[int], all_relevances: list[int], k: int) -> float:
    """nDCG@k for one query; ideal ranking is every known relevant, best first."""
    ideal = sorted(all_relevances, reverse=True)
    idcg = _dcg(ideal, k)
    if idcg <= 0:
        return 0.0
    return _dcg(ranked_relevances, k) / idcg


def mrr_at_k(ranked_relevances: list[int], k: int) -> float:
    """reciprocal rank of the first relevant hit inside the top k, else 0."""
    for i, rel in enumerate(ranked_relevances[:k]):
        if rel > 0:
            return 1.0 / (i + 1)
    return 0.0


def recall_at_k(ranked_relevances: list[int], n_relevant: int, k: int) -> float:
    """fraction of this query's relevant documents found in the top k."""
    if n_relevant <= 0:
        return 0.0
    return sum(1 for rel in ranked_relevances[:k] if rel > 0) / n_relevant


def per_query_metrics(
    scores: list[list[dict]],
    qrels: dict[str, dict[str, int]],
    queries: list[str],
    metrics: tuple[str, ...] = ("ndcg@10", "mrr@10", "recall@100"),
) -> dict[str, list[float]]:
    """score every query separately so the results can be resampled.

    `scores` must be aligned with `queries` — the same order the retriever was
    called with.
    """
    out: dict[str, list[float]] = {m: [] for m in metrics}
    for qid, ranked in zip(queries, scores, strict=True):
        rels = qrels.get(qid, {})
        if not rels:
            continue
        ranked_relevances = [int(rels.get(str(hit["id"]), 0)) for hit in ranked]
        all_relevances = list(rels.values())
        for metric in metrics:
            name, _, k_str = metric.partition("@")
            k = int(k_str)
            if name == "ndcg":
                out[metric].append(ndcg_at_k(ranked_relevances, all_relevances, k))
            elif name == "mrr":
                out[metric].append(mrr_at_k(ranked_relevances, k))
            elif name == "recall":
                out[metric].append(recall_at_k(ranked_relevances, len(rels), k))
            else:
                raise ValueError(f"unsupported per-query metric: {metric}")
    return out


def bootstrap_ci(
    values: list[float],
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict[str, float]:
    """mean and percentile bootstrap interval over per-query scores."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0, "n_queries": 0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    means = arr[idx].mean(axis=1)
    return {
        "mean": float(arr.mean()),
        "ci_low": float(np.quantile(means, alpha / 2)),
        "ci_high": float(np.quantile(means, 1 - alpha / 2)),
        "n_queries": int(arr.size),
    }


def paired_bootstrap(
    values_a: list[float],
    values_b: list[float],
    n_boot: int = 2000,
    seed: int = 42,
) -> dict[str, float]:
    """paired bootstrap on the per-query difference between two systems.

    paired, not independent: both systems answered the same queries, so
    resampling the query set jointly removes the query-difficulty variance that
    dominates a small evaluation set. returns a two-sided p-value for
    "the two systems are equal".
    """
    a = np.asarray(values_a, dtype=np.float64)
    b = np.asarray(values_b, dtype=np.float64)
    if a.size != b.size:
        raise ValueError(
            f"paired test needs matching query sets: {a.size} vs {b.size}"
        )
    diff = a - b
    observed = float(diff.mean())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, diff.size, size=(n_boot, diff.size))
    resampled = diff[idx].mean(axis=1)
    # centre the resampled distribution on zero to test the null hypothesis
    centred = resampled - observed
    p_value = float((np.abs(centred) >= abs(observed)).mean())
    return {
        "delta": observed,
        "ci_low": float(np.quantile(resampled, 0.025)),
        "ci_high": float(np.quantile(resampled, 0.975)),
        "p_value": p_value,
        "significant": bool(p_value < 0.05),
        "n_queries": int(diff.size),
    }


def save_per_query(
    output_dir: str | Path,
    benchmark: str,
    model_name: str,
    per_query: dict[str, list[float]],
    query_ids: list[str],
) -> Path:
    """persist per-query scores so models can be compared statistically later.

    the aggregate numbers in results.json cannot answer "is this difference
    real?"; these files can, via scripts/compare_models.py.
    """
    safe = model_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    directory = Path(output_dir) / "per_query"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{benchmark}__{safe}.json"
    path.write_text(
        json.dumps(
            {"benchmark": benchmark, "model": model_name,
             "query_ids": query_ids, "metrics": per_query},
            indent=2,
        ),
        encoding="utf-8",
    )
    return path
