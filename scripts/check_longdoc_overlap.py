#!/usr/bin/env python3
"""measure document overlap between it-long_doc training text and the mldr-it test corpus.

both corpora are wikipedia-derived, so the same article can appear in the
long-document training source and in the benchmark corpus. that is not query
leakage (it-long_doc queries are independently generated), but training on the
test corpus documents still weakens the "mldr-it is out of domain" claim the
release rests on. this script measures the overlap and writes an exclusion list
so the phase-1 loader can drop the offending articles.

method: content-defined word-shingle containment.
  - normalize both sides to lowercase alphanumeric word streams, which absorbs
    the markdown headings and punctuation differences between the two sources
  - a shingle is a WINDOW_WORDS-word n-gram, kept only at positions whose first
    word hashes to 0 mod ANCHOR_MOD. selection depends on content, not offset,
    so the same passage yields the same shingles in both corpora even when the
    surrounding article is cut differently
  - an article overlaps if at least MIN_HITS of its shingles appear in the
    reference set built from mldr-it

cpu-only: no model weights, no gpu, no retrieval.
"""

from __future__ import annotations

import argparse
import json
import logging
import zlib
from pathlib import Path
from typing import Any, Iterator

# normalization and article identity live in data.py so the keys written here are
# exactly the keys the phase-1 loader filters on
from it_colbert.data import (
    LONGDOC_CONFIG,
    LONGDOC_REPO,
    longdoc_doc_key,
    longdoc_normalize,
)

logger = logging.getLogger(__name__)

WINDOW_WORDS = 13
ANCHOR_MOD = 16
MIN_HITS = 2

# crc32 per distinct word, memoized: italian wikipedia reuses vocabulary heavily,
# so this turns tens of millions of hash calls into dict lookups
_WORD_HASH: dict[str, int] = {}
_WORD_HASH_CAP = 2_000_000


def _word_hash(word: str) -> int:
    cached = _WORD_HASH.get(word)
    if cached is not None:
        return cached
    value = zlib.crc32(word.encode("utf-8"))
    if len(_WORD_HASH) < _WORD_HASH_CAP:
        _WORD_HASH[word] = value
    return value


def shingles(
    text: str,
    window: int = WINDOW_WORDS,
    anchor_mod: int = ANCHOR_MOD,
) -> set[int]:
    words = longdoc_normalize(text).split()
    last = len(words) - window
    if last < 0:
        return set()
    out: set[int] = set()
    for i in range(last + 1):
        if _word_hash(words[i]) % anchor_mod == 0:
            out.add(zlib.crc32(" ".join(words[i : i + window]).encode("utf-8")))
    return out


def build_reference(max_doc_chars: int | None, anchor_mod: int) -> tuple[set[int], int]:
    """shingle set over the full, untruncated mldr-it test corpus."""
    from it_colbert.benchmark.datasets import load_mldr_italian

    logger.info("loading mldr-it test corpus (max_doc_chars=%s)...", max_doc_chars)
    split = load_mldr_italian(split="test", max_doc_chars=max_doc_chars)
    logger.info("mldr-it: %s documents; building shingle set...", len(split.documents))

    reference: set[int] = set()
    for n, doc in enumerate(split.documents, 1):
        reference |= shingles(doc["text"], anchor_mod=anchor_mod)
        if n % 2000 == 0:
            logger.info("  %s/%s docs -> %s shingles", n, len(split.documents), len(reference))
    logger.info("mldr-it reference: %s shingles from %s docs", len(reference), len(split.documents))
    return reference, len(split.documents)


def iter_longdoc_rows(longdoc_dir: Path | None, repo_id: str, config: str) -> Iterator[dict]:
    """stream (document, title_section, instruction_type) row-group by row-group.

    the document column is ~4gb decompressed, so nothing is accumulated here.
    """
    import pyarrow.parquet as pq

    columns = ["document", "title_section", "instruction_type"]
    if longdoc_dir is not None:
        files = sorted(longdoc_dir.glob("*.parquet"))
        if not files:
            raise SystemExit(f"no parquet files under {longdoc_dir}")
        logger.info("reading %s local shards from %s", len(files), longdoc_dir)
        for path in files:
            pf = pq.ParquetFile(path)
            for rg in range(pf.num_row_groups):
                for row in pf.read_row_group(rg, columns=columns).to_pylist():
                    yield row
        return

    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    base = f"datasets/{repo_id}/{config}/"
    files = sorted(fs.glob(base + "*.parquet"))
    logger.info("reading %s remote shards from %s", len(files), base)
    for path in files:
        with fs.open(path, "rb") as handle:
            pf = pq.ParquetFile(handle)
            for rg in range(pf.num_row_groups):
                for row in pf.read_row_group(rg, columns=columns).to_pylist():
                    yield row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=LONGDOC_REPO)
    parser.add_argument("--config", default=LONGDOC_CONFIG)
    parser.add_argument(
        "--longdoc-dir",
        default=None,
        help="local directory of it-long_doc parquet shards; omitted = stream from the hub",
    )
    parser.add_argument(
        "--mldr-max-doc-chars",
        type=int,
        default=0,
        help="0 = untruncated (the right setting here: overlap can sit anywhere in the article)",
    )
    parser.add_argument("--window-words", type=int, default=WINDOW_WORDS)
    parser.add_argument("--anchor-mod", type=int, default=ANCHOR_MOD)
    parser.add_argument("--min-hits", type=int, default=MIN_HITS)
    parser.add_argument("--max-rows", type=int, default=None, help="debug: stop after n rows")
    parser.add_argument("--output", default="outputs/benchmark/longdoc_overlap.json")
    parser.add_argument("--exclusion-output", default="outputs/longdoc_exclude.txt")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    reference, n_mldr_docs = build_reference(
        max_doc_chars=args.mldr_max_doc_chars or None,
        anchor_mod=args.anchor_mod,
    )

    # per unique article: hit count. rows repeat articles (several queries and
    # instruction types per document), so shingling is done once per article.
    hits_by_key: dict[str, int] = {}
    titles_by_key: dict[str, str] = {}
    rows_by_key: dict[str, int] = {}
    n_rows = 0

    longdoc_dir = Path(args.longdoc_dir) if args.longdoc_dir else None
    for row in iter_longdoc_rows(longdoc_dir, args.repo_id, args.config):
        n_rows += 1
        text = row.get("document") or ""
        key = longdoc_doc_key(text)
        rows_by_key[key] = rows_by_key.get(key, 0) + 1
        if key not in hits_by_key:
            titles_by_key[key] = row.get("title_section") or ""
            found = shingles(text, window=args.window_words, anchor_mod=args.anchor_mod)
            hits_by_key[key] = len(found & reference)
        if n_rows % 50_000 == 0:
            flagged = sum(1 for v in hits_by_key.values() if v >= args.min_hits)
            logger.info(
                "  %s rows / %s unique articles / %s flagged",
                f"{n_rows:,}",
                f"{len(hits_by_key):,}",
                f"{flagged:,}",
            )
        if args.max_rows is not None and n_rows >= args.max_rows:
            break

    overlapping = sorted(k for k, v in hits_by_key.items() if v >= args.min_hits)
    rows_dropped = sum(rows_by_key[k] for k in overlapping)
    n_unique = len(hits_by_key)

    report: dict[str, Any] = {
        "repo_id": args.repo_id,
        "config": args.config,
        "method": {
            "window_words": args.window_words,
            "anchor_mod": args.anchor_mod,
            "min_hits": args.min_hits,
            "mldr_max_doc_chars": args.mldr_max_doc_chars or None,
        },
        "mldr_reference": {"documents": n_mldr_docs, "shingles": len(reference)},
        "longdoc": {
            "rows_scanned": n_rows,
            "unique_articles": n_unique,
            "rows_per_article": round(n_rows / max(n_unique, 1), 2),
        },
        "overlap": {
            "articles_flagged": len(overlapping),
            "articles_flagged_frac": round(len(overlapping) / max(n_unique, 1), 6),
            "rows_dropped": rows_dropped,
            "rows_dropped_frac": round(rows_dropped / max(n_rows, 1), 6),
        },
        "examples": [
            {"key": k, "title_section": titles_by_key[k], "shingle_hits": hits_by_key[k]}
            for k in overlapping[:25]
        ],
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    excl = Path(args.exclusion_output)
    excl.parent.mkdir(parents=True, exist_ok=True)
    excl.write_text("".join(f"{k}\n" for k in overlapping), encoding="utf-8")

    logger.info(
        "overlap: %s/%s articles (%.3f%%), %s/%s rows (%.3f%%)",
        f"{len(overlapping):,}",
        f"{n_unique:,}",
        100 * len(overlapping) / max(n_unique, 1),
        f"{rows_dropped:,}",
        f"{n_rows:,}",
        100 * rows_dropped / max(n_rows, 1),
    )
    logger.info("wrote %s and %s", out, excl)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
