#!/usr/bin/env python3
"""TODO.md §13.4 gate: paired bootstrap of arm B (document_length=1024) against
arm A (document_length=512), on the report (held-out) query half.

Reuses the same per-query files and stats helpers as compare_models.py /
report_query_half.py, just pointed at two separate --output-dir trees, because
run_benchmark.py's --colbert-doc-length is a whole-run flag and can't mix both
arms' index lengths in one invocation (TODO.md §13.3).

usage:
    uv run python scripts/compare_longdoc_arms.py
"""

from __future__ import annotations

import json
from pathlib import Path

from it_colbert.benchmark.stats import paired_bootstrap, split_query_ids

ROOT = Path(__file__).resolve().parents[1]
NOISE_FLOOR = 0.0030  # TODO.md §11.4, measured checkpoint-endpoint jitter on MLDR-it


def _safe(name: str) -> str:
    return name.lower().replace(" ", "_").replace("(", "").replace(")", "")


def _load(directory: Path, benchmark: str, model: str) -> dict | None:
    path = directory / "per_query" / f"{benchmark}__{_safe(model)}.json"
    if not path.exists():
        print(f"  (missing: {path})")
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _align(a: dict, b: dict, metric: str, half: str = "report"):
    a_by = dict(zip(a["query_ids"], a["metrics"][metric], strict=False))
    b_by = dict(zip(b["query_ids"], b["metrics"][metric], strict=False))
    shared = [q for q in a["query_ids"] if q in a_by and q in b_by]
    shared = split_query_ids(shared, half=half)
    return [a_by[q] for q in shared], [b_by[q] for q in shared]


def compare(
    label: str,
    dir_a: Path,
    model_a: str,
    dir_b: Path,
    model_b: str,
    benchmark: str,
    metric: str,
    is_guardrail: bool = False,
) -> None:
    a = _load(dir_a, benchmark, model_a)
    b = _load(dir_b, benchmark, model_b)
    if a is None or b is None:
        print(f"{label}: per-query file missing, skipped\n")
        return
    if metric not in a["metrics"] or metric not in b["metrics"]:
        print(f"{label}: metric {metric!r} not present, skipped\n")
        return
    a_vals, b_vals = _align(a, b, metric)
    if not a_vals:
        print(f"{label}: no shared report-half queries, skipped\n")
        return
    result = paired_bootstrap(b_vals, a_vals, n_boot=2000)  # delta = B - A
    verdict = "SIGNIFICANT" if result["significant"] else "not significant"
    if is_guardrail:
        flag = "GUARDRAIL VIOLATED" if result["delta"] < -NOISE_FLOOR else "guardrail ok"
    else:
        flag = "CLEARS the .0030 floor" if result["delta"] > NOISE_FLOOR else "does NOT clear the floor"
    print(
        f"{label}\n"
        f"  {benchmark} {metric}: B-A = {result['delta']:+.4f}  "
        f"[{result['ci_low']:+.4f}, {result['ci_high']:+.4f}]  "
        f"p={result['p_value']:.4f}  {verdict}  n={result['n_queries']}\n"
        f"  -> {flag}\n"
    )


def main() -> None:
    d512 = ROOT / "outputs" / "benchmark_longdoc_512"
    d1024 = ROOT / "outputs" / "benchmark_longdoc_1024"
    d512c = ROOT / "outputs" / "benchmark_longdoc_512_chunked"
    d1024c = ROOT / "outputs" / "benchmark_longdoc_1024_chunked"

    print("=" * 78)
    print("TODO.md SS13.4 gate -- arm B (1024) vs arm A (512), report half")
    print("=" * 78 + "\n")

    print("--- primary: like-for-like, unchunked, at each arm's own index length ---\n")
    compare(
        "unchunked MLDR-it",
        d512, "longdoc-512", d1024, "longdoc-1024",
        "mldr-it", "ndcg@10",
    )

    print("--- primary: chunked (2000/200 chars), both arms ---\n")
    compare(
        "chunked MLDR-it",
        d512c, "longdoc-512 chunked", d1024c, "longdoc-1024 chunked",
        "mldr-it", "ndcg@10",
    )

    print("--- guardrails: must not regress by more than the .0030 floor ---\n")
    compare(
        "mMARCO-it (guardrail)",
        d512, "longdoc-512", d1024, "longdoc-1024",
        "mmarco-it", "mrr@10", is_guardrail=True,
    )
    compare(
        "MIRACL-ita (guardrail)",
        d512, "longdoc-512", d1024, "longdoc-1024",
        "miracl-ita", "ndcg@10", is_guardrail=True,
    )

    print("=" * 78)
    print("Reminder (TODO.md SS13.4): both arms are phase-1-only on a 500k")
    print("mixture. Do NOT compare either arm to .4008 / .4610 (outputs/final_round1)")
    print("-- this gate is strictly A-vs-B, and only decides whether SS13.5's full")
    print("run is worth training.")
    print("=" * 78)


if __name__ == "__main__":
    main()
