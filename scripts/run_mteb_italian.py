#!/usr/bin/env python3
"""evaluate a checkpoint on the Italian tasks in MMTEB.

why this exists: the in-repo benchmark uses a pooled mMARCO corpus, which
inflates absolute scores (jina scores 0.849 there vs 0.337 published on the full
8.8M corpus). numbers from a bespoke harness are not something the community can
check. MMTEB covers 250+ languages and is the de facto leaderboard, so running
its Italian retrieval tasks produces comparable numbers.

there is no Italian MTEB variant the way there is MTEB-French, PL-MTEB, VN-MTEB,
FaMTEB or MTEB-BR. this script selects the Italian tasks out of the multilingual
suite; publishing that selection is itself worth something to Italian NLP.

install:  uv pip install "mteb>=1.29"
usage:
    uv run python scripts/run_mteb_italian.py --model outputs/final
    uv run python scripts/run_mteb_italian.py --list-tasks
"""

from __future__ import annotations

import argparse
import logging

logger = logging.getLogger(__name__)

# task types worth reporting for a retriever; retrieval first, reranking second
DEFAULT_TASK_TYPES = ("Retrieval", "Reranking")


def _load_colbert_for_mteb(model_path: str, document_length: int, query_length: int):
    """wrap a PyLate ColBERT so mteb can call it like a sentence-transformer.

    mteb scores single-vector models by cosine over pooled embeddings. late
    interaction has no single vector, so this returns PyLate's model directly and
    relies on mteb's multi-vector support when present; otherwise the caller
    should stick to the in-repo benchmark for ColBERT and use this script for
    dense baselines.
    """
    from pylate import models

    return models.ColBERT(
        model_name_or_path=model_path,
        document_length=document_length,
        query_length=query_length,
        trust_remote_code=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="MMTEB italian evaluation")
    parser.add_argument("--model", default="outputs/final")
    parser.add_argument("--output-dir", default="outputs/mteb_ita")
    parser.add_argument("--document-length", type=int, default=512)
    parser.add_argument("--query-length", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--task-types",
        nargs="+",
        default=list(DEFAULT_TASK_TYPES),
        help="mteb task types to run",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        help="explicit task names; overrides --task-types",
    )
    parser.add_argument(
        "--list-tasks",
        action="store_true",
        help="print the italian tasks mteb knows about and exit",
    )
    parser.add_argument(
        "--dense",
        action="store_true",
        help="load the model as a plain sentence-transformer instead of PyLate ColBERT",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        import mteb
    except ImportError:
        raise SystemExit(
            'mteb is not installed. run: uv pip install "mteb>=1.29"'
        ) from None

    tasks = mteb.get_tasks(languages=["ita"], task_types=args.task_types)
    if args.tasks:
        wanted = set(args.tasks)
        tasks = [t for t in tasks if t.metadata.name in wanted]

    if args.list_tasks:
        for task in tasks:
            print(f"{task.metadata.name}\t{task.metadata.type}")
        print(f"\n{len(tasks)} italian tasks")
        return

    if not tasks:
        raise SystemExit(f"no italian mteb tasks matched types {args.task_types}")
    logger.info("running %s italian tasks: %s", len(tasks), [t.metadata.name for t in tasks])

    if args.dense:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(args.model, trust_remote_code=True)
    else:
        model = _load_colbert_for_mteb(
            args.model, args.document_length, args.query_length
        )

    evaluation = mteb.MTEB(tasks=tasks)
    results = evaluation.run(
        model,
        output_folder=args.output_dir,
        encode_kwargs={"batch_size": args.batch_size},
    )
    for result in results:
        logger.info("%s: %s", result.task_name, result.get_score())


if __name__ == "__main__":
    main()
