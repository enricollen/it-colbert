"""tiny italian maxsim retrieval demo."""

from __future__ import annotations

import argparse
import logging

import torch
from pylate import models, rank

logger = logging.getLogger(__name__)

DEMO_DOCS = [
    "Roma è la capitale d'Italia e ospita il Colosseo e la Città del Vaticano.",
    "Milano è un importante centro finanziario e della moda nel nord Italia.",
    "La pizza napoletana è un patrimonio culturale immateriale dell'UNESCO.",
    "Il Duomo di Firenze è un capolavoro dell'architettura gotica e rinascimentale.",
    "Venezia è famosa per i suoi canali, le gondole e il Carnevale.",
    "L'intelligenza artificiale sta trasformando i sistemi di ricerca e il RAG.",
    "Il ColBERT usa late interaction e MaxSim per recuperare documenti rilevanti.",
    "La Sicilia è la più grande isola del Mediterraneo e fa parte d'Italia.",
]


def run_infer(
    model_name_or_path: str,
    query: str,
    top_k: int = 3,
    document_length: int = 256,
    query_length: int = 32,
) -> list[tuple[str, float]]:
    logging.basicConfig(level=logging.INFO)
    model = models.ColBERT(
        model_name_or_path=model_name_or_path,
        document_length=document_length,
        query_length=query_length,
    )

    docs_embeddings = model.encode(
        DEMO_DOCS,
        batch_size=8,
        is_query=False,
        show_progress_bar=False,
    )
    query_embeddings = model.encode(
        [query],
        batch_size=1,
        is_query=True,
        show_progress_bar=False,
    )

    # pylate rank.rerank expects lists of embeddings
    try:
        results = rank.rerank(
            documents_ids=[list(range(len(DEMO_DOCS)))],
            queries_embeddings=query_embeddings,
            documents_embeddings=[docs_embeddings],
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        # results[0] is list of {id, score}
        scored = [
            (DEMO_DOCS[int(item["id"])], float(item["score"]))
            for item in results[0][:top_k]
        ]
    except Exception:
        # fallback: manual maxsim if rank api differs
        scored = _manual_maxsim(query_embeddings[0], docs_embeddings, DEMO_DOCS, top_k)

    for i, (doc, score) in enumerate(scored, 1):
        print(f"{i}. score={score:.4f} | {doc}")
    return scored


def _manual_maxsim(query_emb, doc_embs, docs: list[str], top_k: int):
    import numpy as np

    q = np.asarray(query_emb)
    scores = []
    for i, d in enumerate(doc_embs):
        d = np.asarray(d)
        # maxsim: sum over query tokens of max cosine vs doc tokens
        qn = q / (np.linalg.norm(q, axis=-1, keepdims=True) + 1e-9)
        dn = d / (np.linalg.norm(d, axis=-1, keepdims=True) + 1e-9)
        sim = qn @ dn.T
        scores.append((docs[i], float(sim.max(axis=1).sum())))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]


def main() -> None:
    parser = argparse.ArgumentParser(description="italian colbert inference demo")
    parser.add_argument(
        "--model",
        default="outputs/phase2/final",
        help="path or hub id of the colbert model",
    )
    parser.add_argument(
        "--query",
        default="Qual è la capitale d'Italia?",
        help="italian query string",
    )
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()
    run_infer(args.model, args.query, top_k=args.top_k)


if __name__ == "__main__":
    main()
