#!/usr/bin/env python3
"""mine round-2 hard negatives with a trained ItColBERT checkpoint.

the standard ColBERTv2 loop: retrieve with your own model, keep the documents it
wrongly ranks above or near the true positive, and retrain on those. both the
ColBERT-Zero and mxbai-edge-colbert reports name hard-negative mining and
training-data composition as the primary quality drivers — bigger than any
hyperparameter change.

output: a `datasets` directory with columns query / positive / negatives (list),
consumable via `mined_negatives_path` in a phase 1 config.

usage:
    uv run python scripts/mine_hard_negatives.py \\
      --model outputs/final \\
      --output outputs/mined_hn \\
      --queries 50000 --corpus-docs 500000 --negatives 8
"""

from __future__ import annotations

import argparse
import logging
import random
import shutil
from pathlib import Path

from datasets import Dataset

from it_colbert.benchmark.retrievers import ColBERTRetriever
from it_colbert.data import load_mmarco_italian

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="round-2 hard negative mining")
    parser.add_argument("--model", required=True, help="trained colbert checkpoint")
    parser.add_argument("--output", required=True, help="output dataset directory")
    parser.add_argument("--queries", type=int, default=50_000)
    parser.add_argument(
        "--corpus-docs",
        type=int,
        default=500_000,
        help="passages to mine against; larger finds harder negatives but costs more",
    )
    parser.add_argument("--negatives", type=int, default=8)
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="retrieval depth to sample negatives from",
    )
    parser.add_argument(
        "--skip-top",
        type=int,
        default=5,
        help=(
            "discard this many highest-ranked hits before sampling. the very top "
            "hits of a decent model are frequently unlabelled positives, and "
            "training on those teaches the model to demote correct answers"
        ),
    )
    parser.add_argument("--document-length", type=int, default=512)
    parser.add_argument("--query-length", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--index-folder", default="outputs/mining_index")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="re-mine even if the output dataset already exists",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    rng = random.Random(args.seed)

    # mining is one long job with no mid-run checkpoint, so make re-running the
    # command after an interrupt cheap rather than destructive
    output = Path(args.output)
    if output.exists() and not args.overwrite:
        logger.info(
            "%s already exists; nothing to do (pass --overwrite to re-mine)", output
        )
        return

    # mmarco triples give (query, positive); the negatives are what we re-mine
    logger.info("loading mmarco-it triples for %s queries", args.queries)
    triples = load_mmarco_italian(max_samples=args.queries, seed=args.seed)

    # deduplicate queries, keep one positive each
    positive_by_query: dict[str, str] = {}
    for row in triples:
        positive_by_query.setdefault(row["query"], row["positive"])
    queries = list(positive_by_query.keys())
    logger.info("%s unique queries", len(queries))

    # corpus: every positive (so the right answer is always reachable) plus a
    # sample of the collection as the pool to mine distractors from
    corpus_texts: dict[str, str] = {}
    for i, text in enumerate(positive_by_query.values()):
        corpus_texts[f"pos{i}"] = text
    extra = load_mmarco_italian(max_samples=args.corpus_docs, seed=args.seed + 5)
    for i, row in enumerate(extra):
        if len(corpus_texts) >= args.corpus_docs:
            break
        corpus_texts[f"neg{i}"] = row["negative"]
    documents = [{"id": did, "text": text} for did, text in corpus_texts.items()]
    logger.info("mining corpus: %s passages", len(documents))

    retriever = ColBERTRetriever(
        model_name_or_path=args.model,
        documents=documents,
        index_folder=args.index_folder,
        index_name="mining",
        batch_size=args.batch_size,
        document_length=args.document_length,
        query_length=args.query_length,
        override_index=True,
    )
    ranked = retriever.retrieve(queries, k=args.top_k)

    text_by_id = {d["id"]: d["text"] for d in documents}
    out_queries, out_positives, out_negatives = [], [], []
    for query, hits in zip(queries, ranked, strict=True):
        positive = positive_by_query[query]
        candidates = [
            text_by_id[hit["id"]]
            for hit in hits[args.skip_top :]
            if text_by_id.get(hit["id"]) and text_by_id[hit["id"]] != positive
        ]
        if not candidates:
            continue
        rng.shuffle(candidates)
        out_queries.append(query)
        out_positives.append(positive)
        out_negatives.append(candidates[: args.negatives])

    mined = Dataset.from_dict(
        {
            "query": out_queries,
            "positive": out_positives,
            "negatives": out_negatives,
        }
    )
    # write then rename, so an interrupt cannot leave a partial dataset that a
    # later run would treat as complete
    tmp = output.with_name(output.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    mined.save_to_disk(str(tmp))
    if output.exists():
        shutil.rmtree(output)
    tmp.rename(output)
    logger.info(
        "wrote %s rows to %s (avg %.1f negatives/query)",
        len(mined),
        args.output,
        sum(len(n) for n in out_negatives) / max(1, len(out_negatives)),
    )


if __name__ == "__main__":
    main()
