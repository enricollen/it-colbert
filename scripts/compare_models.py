#!/usr/bin/env python3
"""test whether the gap between two models is real, using a paired bootstrap.

a benchmark table of means cannot answer "is this difference established?".
MLDR-it has 200 queries; the standard error on nDCG@10 there is roughly
+/-0.02-0.03. the previously published "fullkd 0.352 vs jina 0.369" gap sat
inside that band and was never a real result.

reads the per-query files written by the benchmark run.

usage:
    uv run python scripts/compare_models.py --benchmark-dir outputs/benchmark \\
      --benchmark mldr-it --a "ItColBERT" --b "jina-colbert-v2"

    # every model against one reference, on every benchmark:
    uv run python scripts/compare_models.py --benchmark-dir outputs/benchmark \\
      --baseline "ItColBERT" --all
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from it_colbert.benchmark.stats import paired_bootstrap


def _safe(name: str) -> str:
    return name.lower().replace(" ", "_").replace("(", "").replace(")", "")


def _load(directory: Path, benchmark: str, model: str) -> dict:
    path = directory / "per_query" / f"{benchmark}__{_safe(model)}.json"
    if not path.exists():
        raise SystemExit(f"missing per-query file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _align(a: dict, b: dict, metric: str) -> tuple[list[float], list[float]]:
    """keep only queries both runs scored, in the same order."""
    a_by_qid = dict(zip(a["query_ids"], a["metrics"][metric], strict=False))
    b_by_qid = dict(zip(b["query_ids"], b["metrics"][metric], strict=False))
    shared = [q for q in a["query_ids"] if q in b_by_qid and q in a_by_qid]
    return [a_by_qid[q] for q in shared], [b_by_qid[q] for q in shared]


def _report(benchmark: str, metric: str, name_a: str, name_b: str, result: dict) -> None:
    verdict = "SIGNIFICANT" if result["significant"] else "not significant"
    print(
        f"{benchmark:<12} {metric:<12} {name_a} - {name_b} = "
        f"{result['delta']:+.4f} "
        f"[{result['ci_low']:+.4f}, {result['ci_high']:+.4f}] "
        f"p={result['p_value']:.4f}  {verdict}  (n={result['n_queries']})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="paired bootstrap between models")
    parser.add_argument("--benchmark-dir", default="outputs/benchmark")
    parser.add_argument("--benchmark", default=None, help="e.g. mldr-it")
    parser.add_argument("--a", default=None, help="model A display name")
    parser.add_argument("--b", default=None, help="model B display name")
    parser.add_argument("--baseline", default=None, help="compare everything to this")
    parser.add_argument("--all", action="store_true", help="every benchmark and model")
    parser.add_argument("--metrics", nargs="+", default=["ndcg@10", "mrr@10"])
    parser.add_argument("--n-boot", type=int, default=2000)
    args = parser.parse_args()

    directory = Path(args.benchmark_dir)
    per_query_dir = directory / "per_query"
    if not per_query_dir.exists():
        raise SystemExit(
            f"{per_query_dir} not found — run the benchmark first, it writes per-query scores"
        )

    if args.all:
        if not args.baseline:
            raise SystemExit("--all requires --baseline")
        found: dict[str, list[str]] = {}
        for path in sorted(per_query_dir.glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            found.setdefault(payload["benchmark"], []).append(payload["model"])
        for benchmark, models in found.items():
            if args.baseline not in models:
                print(f"{benchmark}: baseline {args.baseline!r} not run, skipping")
                continue
            base = _load(directory, benchmark, args.baseline)
            for model in models:
                if model == args.baseline:
                    continue
                other = _load(directory, benchmark, model)
                for metric in args.metrics:
                    if metric not in base["metrics"]:
                        continue
                    a_vals, b_vals = _align(base, other, metric)
                    result = paired_bootstrap(a_vals, b_vals, n_boot=args.n_boot)
                    _report(benchmark, metric, args.baseline, model, result)
            print()
        return

    if not (args.benchmark and args.a and args.b):
        raise SystemExit("need --benchmark --a --b, or --all with --baseline")

    a = _load(directory, args.benchmark, args.a)
    b = _load(directory, args.benchmark, args.b)
    for metric in args.metrics:
        if metric not in a["metrics"]:
            continue
        a_vals, b_vals = _align(a, b, metric)
        result = paired_bootstrap(a_vals, b_vals, n_boot=args.n_boot)
        _report(args.benchmark, metric, args.a, args.b, result)


if __name__ == "__main__":
    main()
