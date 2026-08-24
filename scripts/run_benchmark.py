#!/usr/bin/env python3
"""run italian ir benchmark comparison."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from it_colbert.benchmark.run import (
    DEFAULT_MODELS,
    BenchmarkConfig,
    ModelSpec,
    run_benchmark,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="italian ir sota comparison")
    parser.add_argument("--output-dir", default="outputs/benchmark")
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=["mldr-it", "mmarco-it", "miracl-ita", "squad-ita"],
        choices=["mldr-it", "mmarco-it", "miracl-ita", "squad-ita"],
    )
    parser.add_argument("--mmarco-max-corpus-docs", type=int, default=100_000)
    parser.add_argument("--mmarco-max-queries", type=int, default=None)
    parser.add_argument("--extra-max-corpus-docs", type=int, default=50_000)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument(
        "--only",
        nargs="+",
        default=None,
        help="subset of model display names to run",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="print default models and exit",
    )
    parser.add_argument("--dense-batch-size", type=int, default=64)
    parser.add_argument("--colbert-batch-size", type=int, default=32)
    parser.add_argument(
        "--colbert-doc-length",
        type=int,
        default=512,
        help="index every colbert model at this length (fair length-matched ranking)",
    )
    parser.add_argument(
        "--colbert-query-length",
        type=int,
        default=None,
        help="override every colbert spec's query length (default: each spec's own)",
    )
    parser.add_argument(
        "--chunk-chars",
        type=int,
        default=0,
        help="long-doc mode: split docs into chunks of N chars, max-pool per document",
    )
    parser.add_argument("--chunk-overlap-chars", type=int, default=0)
    parser.add_argument(
        "--mldr-max-doc-chars",
        type=int,
        default=12_000,
        help=(
            "truncate mldr documents to N chars for every retriever including "
            "bm25; lower it to make the bm25 comparison length-symmetric"
        ),
    )
    parser.add_argument(
        "--colbert-brute-force-limit",
        type=int,
        default=25_000,
        help="score corpora up to this many docs with exact maxsim instead of ann",
    )
    parser.add_argument(
        "--extra-colbert",
        nargs="+",
        default=None,
        metavar="NAME=PATH",
        help=(
            "additional colbert checkpoints to score, e.g. "
            "'ItColBERT (round1)=outputs/final_round1'. lets a run compare "
            "intermediate checkpoints without editing DEFAULT_MODELS"
        ),
    )
    parser.add_argument(
        "--models-only-extra",
        action="store_true",
        help="score only --extra-colbert entries, skipping DEFAULT_MODELS",
    )
    args = parser.parse_args()

    if args.list_models:
        for m in DEFAULT_MODELS:
            print(f"{m.name}\t{m.kind}\t{m.model_id or '-'}\t{m.notes}")
        return

    models = [] if args.models_only_extra else list(DEFAULT_MODELS)
    for entry in args.extra_colbert or []:
        if "=" not in entry:
            parser.error(f"--extra-colbert expects NAME=PATH, got {entry!r}")
        name, path = entry.split("=", 1)
        name, path = name.strip(), path.strip()
        if not Path(path).exists():
            parser.error(f"--extra-colbert path does not exist: {path}")
        if any(m.name == name for m in models):
            parser.error(f"--extra-colbert name collides with an existing model: {name}")
        models.append(
            ModelSpec(
                name=name,
                kind="colbert",
                model_id=path,
                document_length=args.colbert_doc_length,
                query_length=args.colbert_query_length or 32,
                notes="ad-hoc checkpoint passed via --extra-colbert",
            )
        )

    cfg = BenchmarkConfig(
        output_dir=args.output_dir,
        index_root=str(Path(args.output_dir) / "indexes"),
        benchmarks=list(args.benchmarks),
        mmarco_max_corpus_docs=args.mmarco_max_corpus_docs,
        mmarco_max_queries=args.mmarco_max_queries,
        extra_max_corpus_docs=args.extra_max_corpus_docs,
        top_k=args.top_k,
        dense_batch_size=args.dense_batch_size,
        colbert_batch_size=args.colbert_batch_size,
        only_models=args.only,
        models=models,
        colbert_document_length=args.colbert_doc_length,
        colbert_query_length=args.colbert_query_length,
        chunk_chars=args.chunk_chars,
        chunk_overlap_chars=args.chunk_overlap_chars,
        mldr_max_doc_chars=args.mldr_max_doc_chars,
        colbert_brute_force_limit=args.colbert_brute_force_limit,
    )
    payload = run_benchmark(cfg)
    print(json.dumps({k: v for k, v in payload["results"].items()}, indent=2)[:2000])
    # hard-exit: cuda/faiss destructors have hung after successful writes
    os._exit(0)


if __name__ == "__main__":
    main()
