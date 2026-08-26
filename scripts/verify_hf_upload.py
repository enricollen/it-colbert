#!/usr/bin/env python3
"""Sanity-check the published Hugging Face model end-to-end.

Downloads enricollen/ItColBERT straight from the Hub (not a local checkpoint),
runs it against a handful of fake Italian query/document sets with an obvious
correct answer, and asserts the top-ranked document is the right one. This is
a smoke test for "does the uploaded repo actually work", not an accuracy
benchmark -- see outputs/benchmark for that.
"""

from __future__ import annotations

import sys

from pylate import models, rank

MODEL_ID = "enricollen/ItColBERT"

CASES = [
    {
        "query": "Qual è la capitale d'Italia?",
        "docs": {
            "capital": "Roma è la capitale d'Italia.",
            "distractor_geo": "Milano è la capitale economica del Paese.",
            "distractor_unrelated": "Il gatto ha dormito tutto il pomeriggio sul divano.",
        },
        "expect_top": "capital",
    },
    {
        "query": "Chi ha dipinto la Gioconda?",
        "docs": {
            "answer": "La Gioconda è stata dipinta da Leonardo da Vinci.",
            "distractor_art": "Michelangelo ha affrescato la Cappella Sistina.",
            "distractor_unrelated": "La ricetta della carbonara prevede uova, guanciale e pecorino.",
        },
        "expect_top": "answer",
    },
    {
        "query": "Quanti pianeti ci sono nel sistema solare?",
        "docs": {
            "answer": "Nel sistema solare ci sono otto pianeti, da Mercurio a Nettuno.",
            "distractor_space": "La Luna è l'unico satellite naturale della Terra.",
            "distractor_unrelated": "Il campionato di calcio italiano si chiama Serie A.",
        },
        "expect_top": "answer",
    },
    {
        "query": "Qual è la lingua ufficiale del Brasile?",
        "docs": {
            "answer": "In Brasile la lingua ufficiale è il portoghese.",
            "distractor_lang": "In Argentina si parla principalmente spagnolo.",
            "distractor_unrelated": "Il Colosseo si trova nel centro di Roma.",
        },
        "expect_top": "answer",
    },
]


def main() -> int:
    print(f"Loading {MODEL_ID} from the Hugging Face Hub (not a local path)...")
    model = models.ColBERT(model_name_or_path=MODEL_ID)
    print("Loaded OK.\n")

    failures = 0
    for i, case in enumerate(CASES, 1):
        labels = list(case["docs"].keys())
        texts = list(case["docs"].values())

        query_embeddings = model.encode([case["query"]], is_query=True, show_progress_bar=False)
        doc_embeddings = model.encode(texts, is_query=False, show_progress_bar=False)

        reranked = rank.rerank(
            documents_ids=[labels],
            queries_embeddings=query_embeddings,
            documents_embeddings=[doc_embeddings],
        )[0]

        print(f"[{i}] query: {case['query']}")
        for item in reranked:
            marker = " <-- top" if item is reranked[0] else ""
            print(f"    {item['id']:<22} score={item['score']:.3f}{marker}")

        top_label = reranked[0]["id"]
        ok = top_label == case["expect_top"]
        status = "PASS" if ok else "FAIL"
        print(f"    expected top: {case['expect_top']!r} -> {status}\n")
        if not ok:
            failures += 1

    total = len(CASES)
    print(f"{total - failures}/{total} cases ranked the correct document first.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
