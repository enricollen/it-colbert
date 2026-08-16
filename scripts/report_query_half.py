#!/usr/bin/env python3
"""report benchmark means on one side of the selection/report query split.

phase 2 selects its checkpoint by nDCG@10 on MLDR-it and MIRACL-ita, using the
same queries `run_benchmark.py` reports (TODO.md §11.1). a number selected and
reported on one query set carries the winner's curse: with ~5 candidates at
SE ~0.03, it is inflated by roughly 0.02-0.04 before anything is learned.

this reads the per-query files the benchmark already wrote, so a held-out number
costs no GPU time and needs no rerun — every model already benchmarked can be
re-read on the half that selection never saw.

usage:
    uv run python scripts/report_query_half.py --half report
    uv run python scripts/report_query_half.py --half report --benchmark mldr-it
    uv run python scripts/report_query_half.py --half selection --metric ndcg@10
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from it_colbert.benchmark.stats import QUERY_HALVES, bootstrap_ci, split_query_ids


def _restrict(payload: dict, metric: str, half: str) -> list[float]:
    """per-query scores for `metric`, limited to one half of the query set."""
    by_qid = dict(zip(payload["query_ids"], payload["metrics"][metric], strict=False))
    keep = split_query_ids(list(by_qid.keys()), half=half)
    return [by_qid[q] for q in keep]


def main() -> None:
    parser = argparse.ArgumentParser(description="benchmark means on a query half")
    parser.add_argument("--benchmark-dir", default="outputs/benchmark")
    parser.add_argument("--half", choices=QUERY_HALVES, default="report")
    parser.add_argument("--benchmark", default=None, help="only this suite")
    parser.add_argument("--metric", default="ndcg@10")
    parser.add_argument("--n-boot", type=int, default=2000)
    args = parser.parse_args()

    per_query_dir = Path(args.benchmark_dir) / "per_query"
    if not per_query_dir.exists():
        raise SystemExit(f"{per_query_dir} not found — run the benchmark first")

    by_benchmark: dict[str, list[tuple[float, str, dict]]] = {}
    for path in sorted(per_query_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        bench = payload["benchmark"]
        if args.benchmark and bench != args.benchmark:
            continue
        if args.metric not in payload["metrics"]:
            continue
        values = _restrict(payload, args.metric, args.half)
        if not values:
            continue
        ci = bootstrap_ci(values, n_boot=args.n_boot)
        by_benchmark.setdefault(bench, []).append((ci["mean"], payload["model"], ci))

    if not by_benchmark:
        raise SystemExit(f"no per-query files carried metric {args.metric!r}")

    for bench, rows in by_benchmark.items():
        first_n = rows[0][2]["n_queries"]
        print(f"=== {bench}  {args.metric}  half={args.half}  n={first_n}")
        for mean, model, ci in sorted(rows, key=lambda r: -r[0]):
            print(
                f"{model:<44s} {mean:.4f}  "
                f"[{ci['ci_low']:.4f}, {ci['ci_high']:.4f}]"
            )
        print()


if __name__ == "__main__":
    main()
