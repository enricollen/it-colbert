"""dense, colbert and bm25 retrievers for italian ir benchmarks."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


def scores_to_pylate(
    query_ids: list[str],
    ranked_ids: list[list[str]],
    ranked_scores: list[list[float]],
) -> list[list[dict]]:
    """convert ranked lists to pylate/ranx score format."""
    out: list[list[dict]] = []
    for docs, scores in zip(ranked_ids, ranked_scores):
        out.append(
            [{"id": str(did), "score": float(sc)} for did, sc in zip(docs, scores)]
        )
    return out


class BM25Retriever:
    def __init__(self, documents: list[dict], tokenizer=None):
        self.doc_ids = [d["id"] for d in documents]
        self.tokenizer = tokenizer or (lambda t: t.lower().split())
        tokenized = [self.tokenizer(d["text"]) for d in documents]
        self.bm25 = BM25Okapi(tokenized)

    def retrieve(self, queries: list[str], k: int = 100) -> list[list[dict]]:
        ranked_ids, ranked_scores = [], []
        for q in queries:
            scores = self.bm25.get_scores(self.tokenizer(q))
            top = np.argpartition(scores, -min(k, len(scores)))[-k:]
            top = top[np.argsort(scores[top])[::-1]]
            ranked_ids.append([self.doc_ids[i] for i in top])
            ranked_scores.append([float(scores[i]) for i in top])
        return scores_to_pylate([], ranked_ids, ranked_scores)


class DenseRetriever:
    """single-vector dense retrieval with faiss inner product on l2-normalized vecs."""

    def __init__(
        self,
        model_name: str,
        documents: list[dict],
        batch_size: int = 64,
        query_prompt: str | None = None,
        doc_prompt: str | None = None,
        max_seq_length: int | None = None,
        normalize: bool = True,
    ):
        import faiss

        self.query_prompt = query_prompt
        self.doc_prompt = doc_prompt
        self.normalize = normalize
        self.doc_ids = [d["id"] for d in documents]
        logger.info("loading dense model %s", model_name)
        try:
            self.model = SentenceTransformer(model_name, trust_remote_code=True)
        except Exception as exc:
            # raw encoder checkpoints (e.g. italian-modernbert-base) need mean pooling
            logger.warning(
                "sentence-transformers load failed (%s); wrapping as mean-pool encoder",
                exc,
            )
            from sentence_transformers import models as st_models

            word = st_models.Transformer(model_name, max_seq_length=max_seq_length or 512)
            pool = st_models.Pooling(word.get_word_embedding_dimension(), pooling_mode="mean")
            self.model = SentenceTransformer(modules=[word, pool])
        if max_seq_length is not None:
            self.model.max_seq_length = max_seq_length

        texts = [
            (self.doc_prompt + d["text"]) if self.doc_prompt else d["text"]
            for d in documents
        ]
        logger.info("encoding %s documents with %s", len(texts), model_name)
        emb = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
        )
        emb = np.asarray(emb, dtype=np.float32)
        self.index = faiss.IndexFlatIP(emb.shape[1])
        self.index.add(emb)

    def retrieve(self, queries: list[str], k: int = 100) -> list[list[dict]]:
        qtexts = [
            (self.query_prompt + q) if self.query_prompt else q for q in queries
        ]
        qemb = self.model.encode(
            qtexts,
            batch_size=64,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
        )
        qemb = np.asarray(qemb, dtype=np.float32)
        scores, idxs = self.index.search(qemb, k)
        ranked_ids, ranked_scores = [], []
        for row_scores, row_idxs in zip(scores, idxs):
            ids, scs = [], []
            for sc, ix in zip(row_scores, row_idxs):
                if ix < 0:
                    continue
                ids.append(self.doc_ids[ix])
                scs.append(float(sc))
            ranked_ids.append(ids)
            ranked_scores.append(scs)
        return scores_to_pylate([], ranked_ids, ranked_scores)


class ColBERTRetriever:
    """pylate colbert retrieval.

    for corpora <= brute_force_limit uses exact maxsim (avoids voyager crashes on
    long multi-vector docs). larger corpora use voyager/plaid ann.
    """

    def __init__(
        self,
        model_name_or_path: str,
        documents: list[dict],
        index_folder: str,
        index_name: str,
        batch_size: int = 32,
        document_length: int = 180,
        query_length: int = 32,
        language: str | None = None,
        override_index: bool = True,
        brute_force_limit: int = 25_000,
    ):
        from pylate import models

        self.doc_ids = [d["id"] for d in documents]
        self.use_bruteforce = len(documents) <= brute_force_limit
        logger.info("loading colbert model %s", model_name_or_path)
        self.model = models.ColBERT(
            model_name_or_path=model_name_or_path,
            document_length=document_length,
            query_length=query_length,
            trust_remote_code=True,
        )
        # xmod / colbert-xm: activate italian language adapters when available
        if language:
            applied = False
            for obj in (self.model, getattr(self.model, "model", None)):
                if obj is None:
                    continue
                set_lang = getattr(obj, "set_default_language", None)
                if callable(set_lang):
                    set_lang(language)
                    applied = True
                    break
                auto = getattr(obj, "auto_model", None)
                set_lang = getattr(auto, "set_default_language", None) if auto else None
                if callable(set_lang):
                    set_lang(language)
                    applied = True
                    break
            if applied:
                logger.info("set colbert default language=%s", language)
            else:
                logger.warning(
                    "language=%s requested but model has no set_default_language",
                    language,
                )
        logger.info(
            "encoding %s documents with colbert (bruteforce=%s)",
            len(documents),
            self.use_bruteforce,
        )
        self.doc_emb = self.model.encode(
            [d["text"] for d in documents],
            batch_size=batch_size,
            is_query=False,
            show_progress_bar=True,
        )
        self.retriever = None
        if not self.use_bruteforce:
            from pylate import indexes
            from pylate import retrieve as pylate_retrieve

            self.index = None
            last_err: Exception | None = None
            for factory in (
                lambda: indexes.PLAID(
                    index_folder=index_folder,
                    index_name=index_name,
                    override=override_index,
                ),
                lambda: indexes.Voyager(
                    index_folder=index_folder,
                    index_name=index_name,
                    override=override_index,
                ),
            ):
                try:
                    self.index = factory()
                    break
                except Exception as exc:  # noqa: BLE001
                    last_err = exc
                    logger.warning("index backend failed (%s); trying next", exc)
            if self.index is None:
                raise RuntimeError(f"no colbert index backend available: {last_err}")
            self.index.add_documents(
                documents_ids=self.doc_ids,
                documents_embeddings=self.doc_emb,
            )
            self.retriever = pylate_retrieve.ColBERT(index=self.index)

    def retrieve(self, queries: list[str], k: int = 100) -> list[list[dict]]:
        q_emb = self.model.encode(
            queries,
            batch_size=32,
            is_query=True,
            show_progress_bar=True,
        )
        if self.use_bruteforce:
            return self._bruteforce_retrieve(q_emb, k=k)
        return self.retriever.retrieve(queries_embeddings=q_emb, k=k)

    def _bruteforce_retrieve(
        self,
        queries_embeddings: list,
        k: int = 100,
        doc_chunk: int = 256,
    ) -> list[list[dict]]:
        """exact late-interaction maxsim against the full in-memory corpus."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ranked_ids: list[list[str]] = []
        ranked_scores: list[list[float]] = []
        n_docs = len(self.doc_emb)

        # pre-pad all docs once into chunks on cpu
        chunk_tensors: list[tuple[torch.Tensor, torch.Tensor, int, int]] = []
        for start in range(0, n_docs, doc_chunk):
            end = min(start + doc_chunk, n_docs)
            chunk = self.doc_emb[start:end]
            max_len = max(int(np.asarray(d).shape[0]) for d in chunk)
            dim = int(np.asarray(chunk[0]).shape[-1])
            padded = torch.zeros((len(chunk), max_len, dim), dtype=torch.float32)
            mask = torch.zeros((len(chunk), max_len), dtype=torch.bool)
            for i, d in enumerate(chunk):
                arr = np.asarray(d, dtype=np.float32)
                padded[i, : arr.shape[0]] = torch.from_numpy(arr)
                mask[i, : arr.shape[0]] = True
            chunk_tensors.append((padded, mask, start, end))

        for qi, qe in enumerate(queries_embeddings):
            q = torch.as_tensor(qe, dtype=torch.float32, device=device)
            if q.ndim != 2:
                q = q.reshape(q.shape[-2], q.shape[-1])
            all_scores = torch.empty(n_docs, dtype=torch.float32, device=device)
            for padded, mask, start, end in chunk_tensors:
                docs = padded.to(device, non_blocking=True)
                dmask = mask.to(device, non_blocking=True)
                # (n, tq, td)
                sim = torch.einsum("qd,ntd->nqt", q, docs)
                sim = sim.masked_fill(~dmask.unsqueeze(1), float("-inf"))
                token_max = sim.max(dim=-1).values  # (n, tq)
                token_max = torch.nan_to_num(token_max, nan=0.0, neginf=0.0)
                all_scores[start:end] = token_max.sum(dim=-1)
            topk = min(k, n_docs)
            vals, idxs = torch.topk(all_scores, k=topk)
            ranked_ids.append([self.doc_ids[int(i)] for i in idxs.tolist()])
            ranked_scores.append([float(v) for v in vals.tolist()])
            if (qi + 1) % 20 == 0:
                logger.info("bruteforce retrieve %s/%s queries", qi + 1, len(queries_embeddings))
        return scores_to_pylate([], ranked_ids, ranked_scores)


def build_base_modernbert_dense(
    documents: list[dict],
    batch_size: int = 64,
    max_seq_length: int = 512,
) -> DenseRetriever:
    """mean-pooled italian modernbert base (pretrained lm, no ir fine-tune)."""
    return DenseRetriever(
        model_name="DeepMount00/Italian-ModernBERT-base",
        documents=documents,
        batch_size=batch_size,
        max_seq_length=max_seq_length,
        normalize=True,
    )


def clear_cuda() -> None:
    # force gc before empty_cache; cuda teardown can otherwise hang on exit
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:  # noqa: BLE001
            pass
