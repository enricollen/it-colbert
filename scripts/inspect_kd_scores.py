"""compare teacher label sharpness across the kd sources used in phase 2.

pylate Distillation does KL(log_softmax(student) || log_softmax(teacher_scores)),
so what the student actually learns is the *softmax* of the teacher scores, not
their raw scale. two sources with different score ranges therefore teach at
different sharpness even though both are "logits".

this prints, per source: raw score range, softmax top-1 probability and entropy.
use it to pick `mmarco_hn_score_temperature` (and optionally `mmarco_hn_score_clip`)
so both sources land at a similar entropy before they are mixed.

usage:
    uv run python scripts/inspect_kd_scores.py --samples 2000
    uv run python scripts/inspect_kd_scores.py --temperatures 1 2 3 4
"""

from __future__ import annotations

import argparse
import math
import statistics

from it_colbert.data import KD_SPLITS, load_kd_italian, load_mmarco_hn_kd_italian


def _softmax_stats(
    rows: list[list[float]],
    documents: list[list[str]] | None = None,
    pad_marker: str = "documento non pertinente.",
) -> dict[str, float]:
    """top-1 probability, entropy (nats) and real documents-per-row.

    `real_docs` is what decides whether raising `kd_n_ways` can help: asking for
    32 ways when the source stores 11 just adds padding, which costs a full
    encoder pass per padded slot and teaches nothing.
    """
    top1: list[float] = []
    entropy: list[float] = []
    real_docs: list[int] = []
    lo, hi = math.inf, -math.inf
    for i, scores in enumerate(rows):
        if not scores:
            continue
        if documents is not None and i < len(documents):
            real_docs.append(
                sum(1 for d in documents[i] if d and d != pad_marker)
            )
        else:
            real_docs.append(len(scores))
        m = max(scores)
        exps = [math.exp(s - m) for s in scores]
        total = sum(exps)
        probs = [e / total for e in exps]
        top1.append(max(probs))
        entropy.append(-sum(p * math.log(p) for p in probs if p > 0))
        lo = min(lo, min(scores))
        hi = max(hi, max(scores))
    return {
        "n_rows": len(top1),
        "score_min": lo,
        "score_max": hi,
        "mean_top1_prob": statistics.fmean(top1),
        "mean_entropy": statistics.fmean(entropy),
        "median_entropy": statistics.median(entropy),
        "mean_real_docs": statistics.fmean(real_docs) if real_docs else 0.0,
        "min_real_docs": min(real_docs) if real_docs else 0,
    }


def _report(label: str, stats: dict[str, float]) -> None:
    print(
        f"{label:<34} rows={stats['n_rows']:<6} "
        f"range=[{stats['score_min']:7.3f}, {stats['score_max']:7.3f}] "
        f"top1={stats['mean_top1_prob']:.3f} "
        f"entropy(mean/med)={stats['mean_entropy']:.3f}/{stats['median_entropy']:.3f} "
        f"docs(mean/min)={stats['mean_real_docs']:.1f}/{stats['min_real_docs']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument(
        "--temperatures",
        type=float,
        nargs="*",
        default=[1.0, 2.0, 3.0, 4.0],
        help="candidate mmarco_hn_score_temperature values to preview",
    )
    parser.add_argument("--n-ways", type=int, default=11)
    args = parser.parse_args()

    print("max entropy for a uniform %s-way target: %.3f nats" % (
        args.n_ways,
        math.log(args.n_ways),
    ))
    print(
        "read two things: (1) match mmarco_hn entropy to the lighton splits by "
        "picking a temperature; (2) check docs(mean/min) before raising "
        "kd_n_ways above %s — padded slots cost encoder passes and teach nothing.\n"
        % args.n_ways
    )

    print("--- lighton kd, per split ---")
    for split in KD_SPLITS:
        try:
            ds = load_kd_italian(
                max_samples=args.samples,
                splits=(split,),
                n_ways=args.n_ways,
                split_sampling="sequential",
            )
        except Exception as exc:  # noqa: BLE001
            print(f"{split:<34} skipped ({exc})")
            continue
        _report(split, _softmax_stats(list(ds["scores"]), list(ds["documents"])))

    print("\n--- mmarco hard negatives, by temperature ---")
    for temp in args.temperatures:
        ds = load_mmarco_hn_kd_italian(
            max_samples=args.samples,
            n_ways=args.n_ways,
            score_temperature=temp,
        )
        _report(
            f"mmarco_hn (T={temp})",
            _softmax_stats(list(ds["scores"]), list(ds["documents"])),
        )


if __name__ == "__main__":
    main()
