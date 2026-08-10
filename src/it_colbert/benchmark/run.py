"""run italian ir comparison across sota baselines."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from pylate import evaluation

from it_colbert.benchmark.datasets import (
    IR_METRICS,
    RetrievalSplit,
    load_mldr_italian,
    load_mmarco_italian_dev,
)
from it_colbert.benchmark.retrievers import (
    BM25Retriever,
    ColBERTRetriever,
    DenseRetriever,
    clear_cuda,
)

logger = logging.getLogger(__name__)


# published full-corpus mmarco-it mrr@10 from literature (not re-run here)
PUBLISHED_MMARCO_IT_MRR10 = {
    "BM25": {"mrr@10": 0.153, "source": "Bonifacio et al. 2021 / Jina table"},
    "mColBERT": {"mrr@10": 0.292, "source": "Bonifacio et al. 2021"},
    "mono-mT5": {"mrr@10": 0.303, "source": "Bonifacio et al. 2021"},
    "mE5-base": {"mrr@10": 0.280, "source": "ColBERT-XM model card (Wang et al.)"},
    "ColBERT-XM": {"mrr@10": 0.265, "source": "Louis et al. 2024 / Jina table"},
    "jina-colbert-v2": {"mrr@10": 0.337, "source": "Jina-ColBERT-v2 model card"},
}


@dataclass
class ModelSpec:
    name: str
    kind: str  # bm25 | dense | colbert
    model_id: str | None = None
    query_prompt: str | None = None
    doc_prompt: str | None = None
    max_seq_length: int | None = 512
    document_length: int = 180
    query_length: int = 32
    # for xmod / colbert-xm style models (e.g. "it_IT")
    language: str | None = None
    notes: str = ""


DEFAULT_MODELS: list[ModelSpec] = [
    ModelSpec(name="BM25", kind="bm25", notes="lexical baseline"),
    ModelSpec(
        name="Italian-ModernBERT-base (mean-pool)",
        kind="dense",
        model_id="DeepMount00/Italian-ModernBERT-base",
        max_seq_length=512,
        notes="starting encoder; no ir fine-tune",
    ),
    ModelSpec(
        name="Italian-ModernBERT-mmarco-mnrl",
        kind="dense",
        model_id="nickprock/Italian-ModernBERT-base-embed-mmarco-mnrl",
        max_seq_length=512,
        notes="same base, dense mnrl on mmarco",
    ),
    ModelSpec(
        name="multilingual-e5-base",
        kind="dense",
        model_id="intfloat/multilingual-e5-base",
        query_prompt="query: ",
        doc_prompt="passage: ",
        max_seq_length=512,
        notes="strong multilingual dense sota",
    ),
    ModelSpec(
        name="multilingual-e5-large",
        kind="dense",
        model_id="intfloat/multilingual-e5-large",
        query_prompt="query: ",
        doc_prompt="passage: ",
        max_seq_length=512,
        notes="strong multilingual dense sota",
    ),
    ModelSpec(
        name="bge-m3",
        kind="dense",
        model_id="BAAI/bge-m3",
        max_seq_length=512,
        notes="multilingual dense sota (dense mode)",
    ),
    ModelSpec(
        name="Ita-Search",
        kind="dense",
        model_id="DeepMount00/Ita-Search",
        query_prompt="Represent this search query for finding relevant passages: ",
        doc_prompt="Represent this passage for retrieval: ",
        max_seq_length=512,
        notes="italian-specialized dense (qwen3-emb ft)",
    ),
    ModelSpec(
        name="jina-colbert-v2",
        kind="colbert",
        model_id="jinaai/jina-colbert-v2",
        document_length=512,
        query_length=32,
        notes="multilingual late-interaction sota",
    ),
    ModelSpec(
        name="mLateOn",
        kind="colbert",
        model_id="lightonai/mLateOn",
        document_length=512,
        query_length=32,
        notes="lighton multilingual pylate colbert (mmbert); includes italian",
    ),
    ModelSpec(
        name="ColBERT-XM",
        kind="colbert",
        model_id="antoinelouis/colbert-xm",
        document_length=512,
        query_length=32,
        language="it_IT",
        notes=(
            "louis et al. xmod colbert; published full-corpus mmarco-it mrr@10=0.265; "
            "stanford colbert-ai checkpoint — may need custom load vs pylate"
        ),
    ),
    ModelSpec(
        name="ItColBERT (ours)",
        kind="colbert",
        model_id="outputs/final",
        document_length=512,
        query_length=32,
        notes="ItColBERT; italian modernbert + pylate; trained doc_len=180",
    ),
    ModelSpec(
        name="ItColBERT (doc256)",
        kind="colbert",
        model_id="outputs/final_doc256",
        document_length=512,
        query_length=32,
        notes="retrain ablation; document_length=256 train+infer",
    ),
    ModelSpec(
        name="ItColBERT (fullkd)",
        kind="colbert",
        model_id="outputs/final_fullkd",
        document_length=512,
        query_length=32,
        notes="full lighton italian kd from phase1; train doc180, infer 512",
    ),
    ModelSpec(
        name="ItColBERT (merge-80k-fullkd)",
        kind="colbert",
        model_id="outputs/final_merge_80k_fullkd",
        document_length=512,
        query_length=32,
        notes=(
            "parameter average 0.5*final(80k kd)+0.5*final_fullkd; "
            "jacolbert-style merge for dual-bench; infer 512"
        ),
    ),
    ModelSpec(
        name="ItColBERT (v2)",
        kind="colbert",
        model_id="outputs/final_v2",
        document_length=512,
        query_length=32,
        notes=(
            "v2 recipe: mnrl init, lr 1e-5, bs1024/mini32 contrastive with mined "
            "hard negatives, proportional kd mixture, ir-metric checkpoint selection"
        ),
    ),
    ModelSpec(
        name="ItColBERT (mixkd)",
        kind="colbert",
        model_id="outputs/final_mixkd",
        document_length=512,
        query_length=32,
        notes=(
            "mix kd: lighton 400k + mmarco-it hn (bge-reranker scores); "
            "train doc180, infer 512"
        ),
    ),
]


@dataclass
class BenchmarkConfig:
    output_dir: str = "outputs/benchmark"
    index_root: str = "outputs/benchmark/indexes"
    benchmarks: list[str] = field(
        default_factory=lambda: ["mldr-it", "mmarco-it"]
    )
    mmarco_max_corpus_docs: int = 200_000
    mmarco_max_queries: int | None = None
    mldr_split: str = "test"
    top_k: int = 100
    dense_batch_size: int = 64
    colbert_batch_size: int = 32
    models: list[ModelSpec] = field(default_factory=lambda: list(DEFAULT_MODELS))
    only_models: list[str] | None = None
    # length-match every colbert model instead of comparing at each spec's own
    # length; the old defaults indexed jina at 180 tokens and ours at 512, which
    # silently handicapped the strongest baseline on long-doc MLDR
    colbert_document_length: int | None = None
    # long-doc mode: index chunks of this many characters and max-pool per document
    chunk_chars: int = 0
    chunk_overlap_chars: int = 0


def _evaluate_run(
    split: RetrievalSplit,
    scores: list[list[dict]],
    top_k: int,
) -> dict[str, float]:
    # pylate evaluate expects queries as list of qids matching qrels keys order
    qids = list(split.queries.keys())
    # align scores with qids order used for retrieval
    metrics = [
        m
        for m in IR_METRICS
        if not (m.startswith("recall@") and int(m.split("@")[1]) > top_k)
        and not (m.startswith("ndcg@") and int(m.split("@")[1]) > top_k)
        and not (m.startswith("hits@") and int(m.split("@")[1]) > top_k)
    ]
    if f"recall@{top_k}" not in metrics and top_k not in (1, 5, 10, 100):
        metrics.append(f"recall@{top_k}")
    return evaluation.evaluate(
        scores=scores,
        qrels=split.qrels,
        queries=qids,
        metrics=metrics,
    )


def _retrieve_for_model(
    spec: ModelSpec,
    split: RetrievalSplit,
    cfg: BenchmarkConfig,
) -> list[list[dict]]:
    qids = list(split.queries.keys())
    qtexts = [split.queries[qid] for qid in qids]
    if spec.kind == "bm25":
        retr = BM25Retriever(split.documents)
        return retr.retrieve(qtexts, k=cfg.top_k)

    if spec.kind == "dense":
        assert spec.model_id is not None
        retr = DenseRetriever(
            model_name=spec.model_id,
            documents=split.documents,
            batch_size=cfg.dense_batch_size,
            query_prompt=spec.query_prompt,
            doc_prompt=spec.doc_prompt,
            max_seq_length=spec.max_seq_length,
        )
        scores = retr.retrieve(qtexts, k=cfg.top_k)
        del retr
        clear_cuda()
        return scores

    if spec.kind == "colbert":
        assert spec.model_id is not None
        safe = (
            split.name
            + "_"
            + spec.name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        )
        doc_len = cfg.colbert_document_length or spec.document_length
        retr = ColBERTRetriever(
            model_name_or_path=spec.model_id,
            documents=split.documents,
            index_folder=str(Path(cfg.index_root) / safe),
            index_name=safe,
            batch_size=cfg.colbert_batch_size,
            document_length=doc_len,
            query_length=spec.query_length,
            language=spec.language,
            override_index=True,
            chunk_chars=cfg.chunk_chars,
            chunk_overlap_chars=cfg.chunk_overlap_chars,
        )
        scores = retr.retrieve(qtexts, k=cfg.top_k)
        del retr
        clear_cuda()
        return scores

    raise ValueError(f"unknown kind: {spec.kind}")


def _load_split(name: str, cfg: BenchmarkConfig) -> RetrievalSplit:
    if name == "mldr-it":
        return load_mldr_italian(split=cfg.mldr_split)
    if name == "mmarco-it":
        return load_mmarco_italian_dev(
            max_corpus_docs=cfg.mmarco_max_corpus_docs,
            max_queries=cfg.mmarco_max_queries,
        )
    raise ValueError(f"unknown benchmark: {name}")


def run_benchmark(cfg: BenchmarkConfig) -> dict[str, Any]:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    Path(cfg.index_root).mkdir(parents=True, exist_ok=True)

    models = cfg.models
    if cfg.only_models:
        wanted = {n.lower() for n in cfg.only_models}
        models = [
            m
            for m in models
            if m.name.lower() in wanted
            or m.name in cfg.only_models
            or (m.model_id and m.model_id.lower() in wanted)
            or any(w in m.name.lower() for w in wanted)
        ]

    results_path = out_dir / "results.json"
    payload: dict[str, Any]
    if results_path.exists():
        payload = json.loads(results_path.read_text(encoding="utf-8"))
        payload.setdefault("protocol", {})
        payload.setdefault("published_mmarco_it_mrr10", PUBLISHED_MMARCO_IT_MRR10)
        payload.setdefault("results", {})
        logger.info("resuming from %s", results_path)
    else:
        payload = {
            "protocol": {
                "metrics": IR_METRICS,
                "top_k": cfg.top_k,
                "mmarco_max_corpus_docs": cfg.mmarco_max_corpus_docs,
                "mmarco_note": (
                    "pooled corpus = all qrel positives + reservoir sample; "
                    "relative comparison across models. for absolute published "
                    "full-corpus mrr@10 see published_mmarco_it_mrr10."
                ),
                "mldr_note": "full italian mldr corpus (~10k docs), official test split",
                "miracl_note": "miracl has no italian language split; not used",
            },
            "published_mmarco_it_mrr10": PUBLISHED_MMARCO_IT_MRR10,
            "results": {},
        }

    # refresh protocol metadata on each run
    payload["protocol"].update(
        {
            "metrics": IR_METRICS,
            "top_k": cfg.top_k,
            "mmarco_max_corpus_docs": cfg.mmarco_max_corpus_docs,
            "colbert_document_length": cfg.colbert_document_length,
            "chunk_chars": cfg.chunk_chars,
            "chunk_overlap_chars": cfg.chunk_overlap_chars,
            "bm25_analyzer": "italian: lowercase + stopwords + snowball stem",
        }
    )

    for bench in cfg.benchmarks:
        logger.info("=== benchmark %s ===", bench)
        split = _load_split(bench, cfg)
        prev = payload["results"].get(bench, {}).get("models", {})
        bench_results: dict[str, Any] = {
            "n_queries": len(split.queries),
            "n_docs": len(split.documents),
            "models": dict(prev),
        }
        for spec in models:
            existing = bench_results["models"].get(spec.name)
            if (
                existing
                and existing.get("error") is None
                and existing.get("metrics")
            ):
                logger.info("skip completed %s / %s", bench, spec.name)
                continue

            logger.info("--- %s / %s ---", bench, spec.name)
            t0 = time.time()
            try:
                scores = _retrieve_for_model(spec, split, cfg)
                metrics = _evaluate_run(split, scores, cfg.top_k)
                err = None
            except Exception as exc:
                logger.exception("failed on %s / %s", bench, spec.name)
                metrics = {}
                err = str(exc)
            elapsed = time.time() - t0
            bench_results["models"][spec.name] = {
                "kind": spec.kind,
                "model_id": spec.model_id,
                "notes": spec.notes,
                # record the length each model actually saw; comparing colbert
                # models indexed at different lengths is not a fair ranking
                "effective_length": (
                    (cfg.colbert_document_length or spec.document_length)
                    if spec.kind == "colbert"
                    else spec.max_seq_length
                ),
                "chunk_chars": cfg.chunk_chars if spec.kind == "colbert" else 0,
                "metrics": metrics,
                "seconds": round(elapsed, 1),
                "error": err,
            }
            payload["results"][bench] = bench_results
            results_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            clear_cuda()

        payload["results"][bench] = bench_results

    results_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("wrote %s", results_path)
    return payload
