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
        default=["mldr-it", "mmarco-it"],
        choices=["mldr-it", "mmarco-it"],
    )
    parser.add_argument("--mmarco-max-corpus-docs", type=int, default=200_000)
    parser.add_argument("--mmarco-max-queries", type=int, default=None)
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
    args = parser.parse_args()

    if args.list_models:
        for m in DEFAULT_MODELS:
            print(f"{m.name}\t{m.kind}\t{m.model_id or '-'}\t{m.notes}")
        return

    cfg = BenchmarkConfig(
        output_dir=args.output_dir,
        index_root=str(Path(args.output_dir) / "indexes"),
        benchmarks=list(args.benchmarks),
        mmarco_max_corpus_docs=args.mmarco_max_corpus_docs,
        mmarco_max_queries=args.mmarco_max_queries,
        top_k=args.top_k,
        dense_batch_size=args.dense_batch_size,
        colbert_batch_size=args.colbert_batch_size,
        only_models=args.only,
    )
    payload = run_benchmark(cfg)
    print(json.dumps({k: v for k, v in payload["results"].items()}, indent=2)[:2000])
    # hard-exit: cuda/faiss destructors have hung after successful writes
    os._exit(0)


if __name__ == "__main__":
    main()
