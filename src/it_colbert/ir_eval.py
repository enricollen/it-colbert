"""training-time IR evaluator so checkpoints are selected on retrieval quality.

hold-out KL only says "the student copies the teacher on the teacher's own
distribution". it does not say the model retrieves better: the fullkd run picked
its best-KL step while pooled mMARCO MRR@10 fell 0.491 -> 0.341. this evaluator
scores real nDCG/MRR on small pooled slices of MLDR-it and mMARCO-it during
training so `metric_for_best_model` tracks the benchmark instead.

sizes are deliberately small (a few thousand docs) — a full 10k/100k corpus pass
every eval would cost more than the training step it is meant to guide.
"""

from __future__ import annotations

import logging
import random
from typing import Any

from sentence_transformers.evaluation import SentenceEvaluator

from pylate import evaluation

from it_colbert.benchmark.datasets import (
    TEVATRON_STYLE_ITALIAN,
    RetrievalSplit,
    load_mldr_italian,
    load_mmarco_italian_dev,
    load_tevatron_style_italian,
)
from it_colbert.benchmark.retrievers import maxsim_topk, scores_to_pylate
from it_colbert.benchmark.stats import split_query_ids

logger = logging.getLogger(__name__)


def _subsample(
    split: RetrievalSplit,
    max_queries: int,
    max_docs: int,
    seed: int,
    half: str = "all",
) -> RetrievalSplit:
    """shrink a retrieval split, always keeping every relevant document.

    `half` restricts the split to one side of the selection/report partition
    before any sampling, so raising `max_queries` can never leak a reporting
    query into a set the trainer selects on.
    """
    rng = random.Random(seed)
    qids = split_query_ids(sorted(split.queries.keys()), half=half)
    if max_queries and len(qids) > max_queries:
        qids = rng.sample(qids, max_queries)
    qids_set = set(qids)
    qrels = {q: split.qrels[q] for q in qids if q in split.qrels}
    needed = {d for rels in qrels.values() for d in rels}

    keep: list[dict] = []
    distractors: list[dict] = []
    for doc in split.documents:
        (keep if doc["id"] in needed else distractors).append(doc)
    budget = max(0, max_docs - len(keep))
    if budget and distractors:
        keep.extend(rng.sample(distractors, min(budget, len(distractors))))

    return RetrievalSplit(
        name=split.name,
        documents=keep,
        queries={q: split.queries[q] for q in qids_set},
        qrels=qrels,
    )


class ItIREvaluator(SentenceEvaluator):
    """nDCG@10 / MRR@10 on pooled MLDR-it, MIRACL-ita and mMARCO-it slices.

    reports per-split metrics and `<name>_score`, the weighted mean of the
    primary metric on each enabled split. `score` is the one to select
    checkpoints on.
    """

    def __init__(
        self,
        name: str = "it-ir",
        mldr_queries: int = 200,
        mldr_docs: int = 3_000,
        mmarco_queries: int = 500,
        mmarco_docs: int = 5_000,
        mmarco_pool_docs: int = 50_000,
        miracl_queries: int = 0,
        miracl_docs: int = 0,
        miracl_pool_docs: int = 30_000,
        batch_size: int = 32,
        top_k: int = 100,
        seed: int = 42,
        weights: tuple[float, ...] = (0.5, 0.5),
        query_half: str = "all",
    ):
        super().__init__()
        self.name = name
        self.batch_size = batch_size
        self.top_k = top_k
        self.weights = weights
        self.query_half = query_half
        self.primary_metric = f"{name}_score"
        self.greater_is_better = True

        self.splits: list[tuple[str, RetrievalSplit, str]] = []
        if mldr_queries and mldr_docs:
            mldr = _subsample(
                load_mldr_italian(split="test"),
                max_queries=mldr_queries,
                max_docs=mldr_docs,
                seed=seed,
                half=query_half,
            )
            self.splits.append(("mldr", mldr, "ndcg@10"))
        if mmarco_queries and mmarco_docs:
            mmarco = _subsample(
                load_mmarco_italian_dev(max_corpus_docs=mmarco_pool_docs, seed=seed),
                max_queries=mmarco_queries,
                max_docs=mmarco_docs,
                seed=seed,
                half=query_half,
            )
            self.splits.append(("mmarco", mmarco, "mrr@10"))
        if miracl_queries and miracl_docs:
            meta = TEVATRON_STYLE_ITALIAN["miracl-ita"]
            miracl = _subsample(
                load_tevatron_style_italian(
                    dataset_id=meta["dataset_id"],
                    split=meta["split"],
                    corpus_id=meta["corpus_id"],
                    corpus_split=meta["corpus_split"],
                    max_corpus_docs=miracl_pool_docs,
                    name="miracl-ita",
                    seed=seed,
                ),
                max_queries=miracl_queries,
                max_docs=miracl_docs,
                seed=seed + 3,
                half=query_half,
            )
            self.splits.append(("miracl", miracl, "ndcg@10"))
        if not self.splits:
            raise ValueError("ItIREvaluator needs at least one split enabled")

        for key, split, _ in self.splits:
            logger.info(
                "ir eval %s: %s queries / %s docs (query half: %s)",
                key,
                len(split.queries),
                len(split.documents),
                query_half,
            )
        if query_half == "all":
            logger.warning(
                "ir eval reads the full benchmark query set. safe while nothing "
                "selects on it, but any run with load_best_model_at_end or early "
                "stopping is selecting on test queries — set query_half to "
                "'selection' there (TODO.md §11.1)"
            )

    def __call__(
        self,
        model: Any,
        output_path: str | None = None,
        epoch: int = -1,
        steps: int = -1,
        **kwargs: Any,
    ) -> dict[str, float]:
        metrics: dict[str, float] = {}
        primaries: list[float] = []

        for key, split, primary in self.splits:
            qids = list(split.queries.keys())
            qtexts = [split.queries[q] for q in qids]
            doc_emb = model.encode(
                [d["text"] for d in split.documents],
                batch_size=self.batch_size,
                is_query=False,
                show_progress_bar=False,
            )
            q_emb = model.encode(
                qtexts,
                batch_size=self.batch_size,
                is_query=True,
                show_progress_bar=False,
            )
            ranked_ids, ranked_scores = maxsim_topk(
                doc_emb,
                q_emb,
                [d["id"] for d in split.documents],
                k=min(self.top_k, len(split.documents)),
            )
            scored = evaluation.evaluate(
                scores=scores_to_pylate([], ranked_ids, ranked_scores),
                qrels=split.qrels,
                queries=qids,
                metrics=["ndcg@10", "mrr@10", "recall@10"],
            )
            for metric, value in scored.items():
                metrics[f"{self.name}_{key}_{metric}"] = float(value)
            primaries.append(float(scored[primary]))

        weights = self.weights[: len(primaries)]
        total = sum(weights) or 1.0
        metrics[self.primary_metric] = sum(
            w * v for w, v in zip(weights, primaries)
        ) / total
        logger.info("ir eval @ step %s: %s", steps, metrics)
        return metrics


class CombinedEvaluator(SentenceEvaluator):
    """run several evaluators per eval step and merge their metric dicts.

    written instead of `SequentialEvaluator` because we need a specific child's
    metric as `primary_metric` (the IR score, not the KL) and the sentence-
    transformers version pins the merge/primary behaviour differently across 3.x/5.x.
    """

    def __init__(self, evaluators: list[SentenceEvaluator], primary_metric: str):
        super().__init__()
        self.evaluators = evaluators
        self.primary_metric = primary_metric

    def __call__(
        self,
        model: Any,
        output_path: str | None = None,
        epoch: int = -1,
        steps: int = -1,
        **kwargs: Any,
    ) -> dict[str, float]:
        merged: dict[str, float] = {}
        for evaluator in self.evaluators:
            result = evaluator(model, output_path=output_path, epoch=epoch, steps=steps)
            if isinstance(result, dict):
                merged.update({k: float(v) for k, v in result.items()})
            else:
                merged[getattr(evaluator, "primary_metric", "score")] = float(result)
        if self.primary_metric not in merged:
            raise KeyError(
                f"primary metric {self.primary_metric} missing; got {sorted(merged)}"
            )
        return merged
