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

import json
import logging
import math
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


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
