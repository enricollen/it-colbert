#!/usr/bin/env python3
"""mine round-2 hard negatives with a trained ItColBERT checkpoint.

the standard ColBERTv2 loop: retrieve with your own model, keep the documents it
wrongly ranks above or near the true positive, and retrain on those. both the
ColBERT-Zero and mxbai-edge-colbert reports name hard-negative mining and
training-data composition as the primary quality drivers — bigger than any
hyperparameter change.

output: a `datasets` directory with columns query / positive / negatives (list),
consumable via `mined_negatives_path` in a phase 1 config.

checkpoints under --state-dir so a crash during retrieval resumes from the last
finished query chunk instead of re-encoding the corpus and rebuilding the index.

each retrieval chunk runs in a fresh subprocess so the ~16gb voyager index is
not held alongside encode buffers and rerank spikes (27gb wsl oom otherwise).

usage:
    uv run python scripts/mine_hard_negatives.py \\
      --model outputs/final \\
      --output outputs/mined_hn \\
      --queries 50000 --corpus-docs 500000 --negatives 8
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import random
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from datasets import Dataset

from it_colbert.benchmark.retrievers import ColBERTRetriever, clear_cuda
from it_colbert.data import load_mmarco_italian

logger = logging.getLogger(__name__)

MANIFEST_VERSION = 1


def _manifest(args: argparse.Namespace, n_documents: int, n_queries: int) -> dict[str, Any]:
    return {
        "version": MANIFEST_VERSION,
        "model": args.model,
        "corpus_docs": args.corpus_docs,
        "queries": args.queries,
        "seed": args.seed,
        "document_length": args.document_length,
        "query_length": args.query_length,
        "top_k": args.top_k,
        "skip_top": args.skip_top,
        "negatives": args.negatives,
        "query_chunk_size": args.query_chunk_size,
        "retrieve_batch_size": args.retrieve_batch_size,
        "n_documents": n_documents,
        "n_queries": n_queries,
    }


def _index_ready(index_folder: Path, index_name: str) -> bool:
    voyager = index_folder / index_name / "index.voyager"
    return voyager.is_file() and voyager.stat().st_size > 0


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_write_json(path: Path, payload: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def _build_corpus_and_queries(
    args: argparse.Namespace,
) -> tuple[list[dict], list[str], dict[str, str]]:
    logger.info("loading mmarco-it triples for %s queries", args.queries)
    triples = load_mmarco_italian(max_samples=args.queries, seed=args.seed)

    positive_by_query: dict[str, str] = {}
    for row in triples:
        positive_by_query.setdefault(row["query"], row["positive"])
    queries = list(positive_by_query.keys())
    logger.info("%s unique queries", len(queries))

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
    return documents, queries, positive_by_query


def _save_state(
    state_dir: Path,
    manifest: dict[str, Any],
    documents: list[dict],
    queries: list[str],
    positive_by_query: dict[str, str],
) -> None:
    state_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(state_dir / "manifest.json", manifest)
    _atomic_write_json(
        state_dir / "corpus.json",
        {d["id"]: d["text"] for d in documents},
    )
    _atomic_write_json(
        state_dir / "queries.json",
        {
            "queries": queries,
            "positive_by_query": positive_by_query,
        },
    )


def _load_state(state_dir: Path) -> tuple[dict[str, Any], list[dict], list[str], dict[str, str]]:
    manifest = _load_json(state_dir / "manifest.json")
    corpus = _load_json(state_dir / "corpus.json")
    qdata = _load_json(state_dir / "queries.json")
    documents = [{"id": did, "text": text} for did, text in corpus.items()]
    return manifest, documents, qdata["queries"], qdata["positive_by_query"]


def _load_text_by_id(state_dir: Path) -> dict[str, str]:
    return _load_json(state_dir / "corpus.json")


def _mine_chunk_rows(
    queries: list[str],
    ranked: list[list[dict]],
    positive_by_query: dict[str, str],
    text_by_id: dict[str, str],
    args: argparse.Namespace,
    rng: random.Random,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
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
        rows.append(
            {
                "query": query,
                "positive": positive,
                "negatives": candidates[: args.negatives],
            }
        )
    return rows


def _chunk_path(chunks_dir: Path, chunk_idx: int) -> Path:
    return chunks_dir / f"chunk_{chunk_idx:05d}.json"


def _load_completed_rows(chunks_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(chunks_dir.glob("chunk_*.json")):
        rows.extend(_load_json(path))
    return rows


def _next_chunk_idx(chunks_dir: Path) -> int:
    return len(list(chunks_dir.glob("chunk_*.json")))


def _worker_argv(args: argparse.Namespace, chunk_idx: int) -> list[str]:
    argv = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--model",
        args.model,
        "--output",
        args.output,
        "--queries",
        str(args.queries),
        "--corpus-docs",
        str(args.corpus_docs),
        "--negatives",
        str(args.negatives),
        "--top-k",
        str(args.top_k),
        "--skip-top",
        str(args.skip_top),
        "--document-length",
        str(args.document_length),
        "--query-length",
        str(args.query_length),
        "--batch-size",
        str(args.batch_size),
        "--query-chunk-size",
        str(args.query_chunk_size),
        "--retrieve-batch-size",
        str(args.retrieve_batch_size),
        "--index-folder",
        args.index_folder,
        "--state-dir",
        args.state_dir,
        "--seed",
        str(args.seed),
        "--worker-chunk-idx",
        str(chunk_idx),
    ]
    return argv


def _retrieve_one_chunk(
    args: argparse.Namespace,
    queries: list[str],
    positive_by_query: dict[str, str],
    text_by_id: dict[str, str],
    chunk_idx: int,
    chunks_dir: Path,
    index_folder: Path,
    index_name: str,
) -> None:
    chunk = max(1, args.query_chunk_size)
    out_path = _chunk_path(chunks_dir, chunk_idx)
    if out_path.exists():
        logger.info("skip chunk %s (already on disk)", chunk_idx)
        return

    start = chunk_idx * chunk
    batch = queries[start : start + chunk]
    logger.info(
        "retrieving chunk %s: queries %s-%s / %s (retrieve bs=%s)",
        chunk_idx,
        start + 1,
        start + len(batch),
        len(queries),
        args.retrieve_batch_size,
    )

    retriever = ColBERTRetriever(
        model_name_or_path=args.model,
        documents=[],
        index_folder=str(index_folder),
        index_name=index_name,
        batch_size=args.batch_size,
        document_length=args.document_length,
        query_length=args.query_length,
        encode_documents=False,
    )
    ranked = retriever.retrieve(
        batch,
        k=args.top_k,
        retrieve_batch_size=args.retrieve_batch_size,
    )
    rng = random.Random(args.seed + chunk_idx)
    rows = _mine_chunk_rows(
        batch, ranked, positive_by_query, text_by_id, args, rng
    )
    _atomic_write_json(out_path, rows)
    logger.info("checkpoint chunk %s -> %s (%s rows)", chunk_idx, out_path, len(rows))
    del retriever, ranked, rows
    gc.collect()
    clear_cuda()


def _worker_main(args: argparse.Namespace) -> None:
    assert args.worker_chunk_idx is not None
    state_dir = Path(args.state_dir)
    chunks_dir = state_dir / "chunks"
    index_folder = Path(args.index_folder)
    index_name = "mining"

    manifest, _documents, queries, positive_by_query = _load_state(state_dir)
    expected = _manifest(args, manifest["n_documents"], manifest["n_queries"])
    core_keys = (
        "version",
        "model",
        "corpus_docs",
        "queries",
        "seed",
        "document_length",
        "query_length",
        "top_k",
        "skip_top",
        "negatives",
        "query_chunk_size",
        "n_documents",
        "n_queries",
    )
    stale = {k for k in core_keys if manifest.get(k) != expected.get(k)}
    if stale:
        raise RuntimeError(
            f"state dir {state_dir} does not match current args on {sorted(stale)}"
        )
    if not _index_ready(index_folder, index_name):
        raise RuntimeError("worker started before index was built")

    text_by_id = _load_text_by_id(state_dir)
    _retrieve_one_chunk(
        args,
        queries,
        positive_by_query,
        text_by_id,
        args.worker_chunk_idx,
        chunks_dir,
        index_folder,
        index_name,
    )


def _spawn_retrieval_chunks(
    args: argparse.Namespace,
    queries: list[str],
    chunks_dir: Path,
) -> None:
    chunk = max(1, args.query_chunk_size)
    n_chunks = (len(queries) + chunk - 1) // chunk
    start_chunk = _next_chunk_idx(chunks_dir)
    if start_chunk >= n_chunks:
        logger.info("all query chunks already retrieved")
        return

    logger.info(
        "retrieval: subprocess mode from chunk %s / %s (query-chunk-size=%s)",
        start_chunk,
        n_chunks,
        chunk,
    )
    for chunk_idx in range(start_chunk, n_chunks):
        out_path = _chunk_path(chunks_dir, chunk_idx)
        if out_path.exists():
            logger.info("skip chunk %s (already on disk)", chunk_idx)
            continue
        logger.info("spawning worker for chunk %s", chunk_idx)
        subprocess.run(_worker_argv(args, chunk_idx), check=True)


def _assemble_output(args: argparse.Namespace, chunks_dir: Path, output: Path) -> None:
    all_rows = _load_completed_rows(chunks_dir)
    if len(all_rows) == 0:
        raise RuntimeError("no mined rows; retrieval may have failed")

    mined = Dataset.from_dict(
        {
            "query": [r["query"] for r in all_rows],
            "positive": [r["positive"] for r in all_rows],
            "negatives": [r["negatives"] for r in all_rows],
        }
    )
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
        sum(len(n) for n in mined["negatives"]) / max(1, len(mined)),
    )


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
    parser.add_argument(
        "--query-chunk-size",
        type=int,
        default=200,
        help="queries per checkpoint chunk (each chunk runs in its own subprocess)",
    )
    parser.add_argument(
        "--retrieve-batch-size",
        type=int,
        default=5,
        help="queries per pylate rerank batch during retrieval (lower = less ram)",
    )
    parser.add_argument("--index-folder", default="outputs/mining_index")
    parser.add_argument(
        "--state-dir",
        default="outputs/mining_state",
        help="checkpoint dir: corpus, manifest, per-chunk retrieval results",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="re-mine even if the final output dataset already exists",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="delete state + index and start from scratch",
    )
    parser.add_argument(
        "--worker-chunk-idx",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.worker_chunk_idx is not None:
        _worker_main(args)
        return

    output = Path(args.output)
    if output.exists() and not args.overwrite:
        logger.info(
            "%s already exists; nothing to do (pass --overwrite to re-mine)", output
        )
        return

    state_dir = Path(args.state_dir)
    chunks_dir = state_dir / "chunks"
    index_folder = Path(args.index_folder)
    index_name = "mining"

    if args.fresh:
        logger.info("--fresh: clearing %s and %s", state_dir, index_folder)
        shutil.rmtree(state_dir, ignore_errors=True)
        shutil.rmtree(index_folder, ignore_errors=True)

    documents: list[dict]
    queries: list[str]
    positive_by_query: dict[str, str]

    if (state_dir / "manifest.json").exists():
        manifest, documents, queries, positive_by_query = _load_state(state_dir)
        expected_manifest = _manifest(args, len(documents), len(queries))
        core_keys = (
            "version",
            "model",
            "corpus_docs",
            "queries",
            "seed",
            "document_length",
            "query_length",
            "top_k",
            "skip_top",
            "negatives",
            "query_chunk_size",
            "n_documents",
            "n_queries",
        )
        stale = {
            k
            for k in core_keys
            if manifest.get(k) != expected_manifest.get(k)
        }
        if stale:
            raise RuntimeError(
                f"state dir {state_dir} does not match current args on {sorted(stale)}; "
                "pass --fresh to rebuild"
            )
        if manifest.get("retrieve_batch_size") != expected_manifest["retrieve_batch_size"]:
            manifest["retrieve_batch_size"] = expected_manifest["retrieve_batch_size"]
            _atomic_write_json(state_dir / "manifest.json", manifest)
        logger.info(
            "resuming from %s (%s docs, %s queries, %s chunks done)",
            state_dir,
            len(documents),
            len(queries),
            _next_chunk_idx(chunks_dir),
        )
    else:
        documents, queries, positive_by_query = _build_corpus_and_queries(args)
        expected_manifest = _manifest(args, len(documents), len(queries))
        _save_state(state_dir, expected_manifest, documents, queries, positive_by_query)
        chunks_dir.mkdir(parents=True, exist_ok=True)

    if not _index_ready(index_folder, index_name):
        logger.info("building index (%s documents)", len(documents))
        retriever = ColBERTRetriever(
            model_name_or_path=args.model,
            documents=documents,
            index_folder=str(index_folder),
            index_name=index_name,
            batch_size=args.batch_size,
            document_length=args.document_length,
            query_length=args.query_length,
            override_index=True,
            encode_documents=True,
        )
        del retriever, documents
        gc.collect()
        clear_cuda()
        if not _index_ready(index_folder, index_name):
            raise RuntimeError("index build finished but index.voyager is missing")
        logger.info("index saved at %s/%s", index_folder, index_name)
    else:
        logger.info("reusing saved index at %s/%s", index_folder, index_name)

    chunks_dir.mkdir(parents=True, exist_ok=True)
    _spawn_retrieval_chunks(args, queries, chunks_dir)
    _assemble_output(args, chunks_dir, output)


if __name__ == "__main__":
    main()
