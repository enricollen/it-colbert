#!/usr/bin/env python3
"""measure query/document token lengths against the configured truncation limits.

two questions this answers, both open in TODO.md:

- §10.5: query_length is pinned at 32 in both training phases and in every
  benchmark ModelSpec. if a real fraction of MLDR-it queries truncate, that is a
  one-line config fix on the weakest benchmark.
- §11.2: document_length is 512 while MLDR-it documents are long, so the chunked
  re-run needs to know how many chunks it would produce and how much multi-vector
  memory that implies before it is launched on a 24gb gpu / 27gb host.

tokenizer-only: no model weights, no gpu, no retrieval.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Any

from it_colbert.benchmark.datasets import (
    TEVATRON_STYLE_ITALIAN,
    RetrievalSplit,
    load_mldr_italian,
    load_mmarco_italian_dev,
    load_tevatron_style_italian,
)

logger = logging.getLogger(__name__)

BENCHMARKS = ["mldr-it", "mmarco-it", "miracl-ita", "squad-ita"]


def _load_split(
    name: str,
    mmarco_max_corpus_docs: int,
    extra_max_corpus_docs: int,
    mldr_max_doc_chars: int | None,
) -> RetrievalSplit:
    # mirrors benchmark.run._load_split so the texts measured here are the exact
    # strings the benchmark encodes, truncation defaults included
    if name == "mldr-it":
        return load_mldr_italian(split="test", max_doc_chars=mldr_max_doc_chars)
    if name == "mmarco-it":
        return load_mmarco_italian_dev(
            max_corpus_docs=mmarco_max_corpus_docs, max_queries=None
        )
    if name in TEVATRON_STYLE_ITALIAN:
        spec = TEVATRON_STYLE_ITALIAN[name]
        return load_tevatron_style_italian(
            dataset_id=spec["dataset_id"],
            split=spec["split"],
            corpus_id=spec["corpus_id"],
            corpus_split=spec["corpus_split"],
            max_corpus_docs=extra_max_corpus_docs,
            name=name,
        )
    raise ValueError(f"unknown benchmark: {name}")


def _percentiles(values: list[int]) -> dict[str, float]:
    if not values:
        return {}
    ordered = sorted(values)

    def pct(p: float) -> int:
        idx = min(len(ordered) - 1, max(0, int(round(p / 100 * (len(ordered) - 1)))))
        return ordered[idx]

    return {
        "n": len(ordered),
        "mean": round(sum(ordered) / len(ordered), 1),
        "p50": pct(50),
        "p90": pct(90),
        "p95": pct(95),
        "p99": pct(99),
        "max": ordered[-1],
    }


def _truncation_stats(lengths: list[int], limit: int) -> dict[str, Any]:
    """how many sequences hit the cap, and how much text the cap discards.

    `tokens_kept_frac` is the metric that matters for a retriever: a corpus where
    90% of documents truncate but only 5% of content is lost is a different
    problem from one where half the tokens are never encoded.
    """
    if not lengths:
        return {}
    total = sum(lengths)
    kept = sum(min(n, limit) for n in lengths)
    over = [n for n in lengths if n > limit]
    return {
        "limit": limit,
        "n_truncated": len(over),
        "frac_truncated": round(len(over) / len(lengths), 4),
        "tokens_kept_frac": round(kept / total, 4) if total else 1.0,
        # among the sequences that do truncate, how long are they really
        "truncated_len": _percentiles(over),
    }


def _chunk_projection(
    doc_char_lengths: list[int],
    doc_token_lengths: list[int],
    chunk_chars: int,
    overlap_chars: int,
    document_length: int,
    dim: int,
    n_docs_total: int,
) -> dict[str, Any]:
    """project chunk count and index size for benchmark.run's chunk_chars mode.

    replicates chunk_long_documents' stepping so the estimate matches what the
    benchmark would actually build, then sizes the multi-vector payload. the
    sampled docs are scaled up to the full corpus.
    """
    if chunk_chars <= 0:
        return {}
    step = max(1, chunk_chars - max(0, overlap_chars))
    n_chunks = 0
    for n_chars in doc_char_lengths:
        if n_chars <= chunk_chars:
            n_chunks += 1
            continue
        for start in range(0, n_chars, step):
            n_chunks += 1
            if start + chunk_chars >= n_chars:
                break
    sampled = len(doc_char_lengths)
    scale = n_docs_total / sampled if sampled else 1.0

    # a chunk is chunk_chars long, so it encodes to roughly the same token count
    # as a document of that size, capped at document_length
    chars_per_token = (
        sum(doc_char_lengths) / sum(doc_token_lengths) if sum(doc_token_lengths) else 4.0
    )
    tokens_per_chunk = min(document_length, chunk_chars / chars_per_token)
    total_vectors = n_chunks * scale * tokens_per_chunk
    # unchunked baseline: every doc capped at document_length
    baseline_vectors = sum(min(n, document_length) for n in doc_token_lengths) * scale

    return {
        "chunk_chars": chunk_chars,
        "overlap_chars": overlap_chars,
        "est_chunks": int(round(n_chunks * scale)),
        "chunks_per_doc": round(n_chunks / sampled, 2) if sampled else 0.0,
        "chars_per_token": round(chars_per_token, 2),
        "est_tokens_per_chunk": int(round(tokens_per_chunk)),
        "est_total_vectors": int(round(total_vectors)),
        "est_embedding_gb_fp32": round(total_vectors * dim * 4 / 1024**3, 2),
        "est_embedding_gb_fp16": round(total_vectors * dim * 2 / 1024**3, 2),
        "unchunked_embedding_gb_fp32": round(baseline_vectors * dim * 4 / 1024**3, 2),
        # benchmark.retrievers.ColBERTRetriever switches to an ann index above
        # brute_force_limit=25_000 documents; chunking can cross that line
        "exceeds_bruteforce_limit_25k": bool(round(n_chunks * scale) > 25_000),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmarks", nargs="+", default=BENCHMARKS, choices=BENCHMARKS)
    parser.add_argument(
        "--tokenizer",
        default="outputs/final_round1",
        help="tokenizer source; any ItColBERT checkpoint shares the modernbert vocab",
    )
    parser.add_argument("--query-length", type=int, default=32)
    parser.add_argument("--document-length", type=int, default=512)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--chunk-chars", type=int, default=2000)
    parser.add_argument("--chunk-overlap-chars", type=int, default=200)
    parser.add_argument(
        "--max-docs",
        type=int,
        default=2000,
        help="sample this many documents per corpus for tokenization",
    )
    parser.add_argument("--mmarco-max-corpus-docs", type=int, default=100_000)
    parser.add_argument("--extra-max-corpus-docs", type=int, default=50_000)
    parser.add_argument(
        "--mldr-max-doc-chars",
        type=int,
        default=12_000,
        help="load_mldr_italian's own default; 0 disables to see untruncated lengths",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="outputs/benchmark/length_audit.json")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    def count(texts: list[str]) -> list[int]:
        out: list[int] = []
        for i in range(0, len(texts), 256):
            batch = texts[i : i + 256]
            enc = tok(batch, add_special_tokens=True, truncation=False)
            out.extend(len(ids) for ids in enc["input_ids"])
        return out

    report: dict[str, Any] = {
        "tokenizer": args.tokenizer,
        "query_length": args.query_length,
        "document_length": args.document_length,
        "benchmarks": {},
    }

    for bench in args.benchmarks:
        logger.info("=== %s ===", bench)
        split = _load_split(
            bench,
            mmarco_max_corpus_docs=args.mmarco_max_corpus_docs,
            extra_max_corpus_docs=args.extra_max_corpus_docs,
            mldr_max_doc_chars=args.mldr_max_doc_chars or None,
        )

        qtexts = list(split.queries.values())
        qlens = count(qtexts)

        rng = random.Random(args.seed)
        docs = split.documents
        sample = docs if len(docs) <= args.max_docs else rng.sample(docs, args.max_docs)
        dtexts = [d["text"] for d in sample]
        dlens = count(dtexts)
        dchars = [len(t) for t in dtexts]

        entry: dict[str, Any] = {
            "n_queries": len(split.queries),
            "n_docs": len(docs),
            "docs_sampled": len(sample),
            "query_tokens": _percentiles(qlens),
            "query_truncation": _truncation_stats(qlens, args.query_length),
            "doc_tokens": _percentiles(dlens),
            "doc_truncation": _truncation_stats(dlens, args.document_length),
            "doc_chars": _percentiles(dchars),
        }
        if bench == "mldr-it":
            entry["chunk_projection"] = _chunk_projection(
                dchars,
                dlens,
                chunk_chars=args.chunk_chars,
                overlap_chars=args.chunk_overlap_chars,
                document_length=args.document_length,
                dim=args.dim,
                n_docs_total=len(docs),
            )
        report["benchmarks"][bench] = entry

        q = entry["query_truncation"]
        d = entry["doc_truncation"]
        logger.info(
            "%s: queries p95=%s max=%s | %.1f%% over %s tokens",
            bench,
            entry["query_tokens"]["p95"],
            entry["query_tokens"]["max"],
            100 * q["frac_truncated"],
            args.query_length,
        )
        logger.info(
            "%s: docs p50=%s p95=%s | %.1f%% over %s tokens, %.1f%% of doc tokens encoded",
            bench,
            entry["doc_tokens"]["p50"],
            entry["doc_tokens"]["p95"],
            100 * d["frac_truncated"],
            args.document_length,
            100 * d["tokens_kept_frac"],
        )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("wrote %s", out)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
