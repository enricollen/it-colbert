"""dense, colbert and bm25 retrievers for italian ir benchmarks."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

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


def maxsim_topk(
    doc_embeddings: list,
    query_embeddings: list,
    doc_ids: list[str],
    k: int = 100,
    doc_chunk: int = 256,
    log_every: int = 0,
    consume: bool = False,
) -> tuple[list[list[str]], list[list[float]]]:
    """exact late-interaction maxsim of every query against the whole corpus.

    kept separate from ColBERTRetriever so the training-time IR evaluator can
    score a checkpoint without building a second model/index.

    `consume` frees each source embedding as it is copied into the padded
    tensors, so peak host memory is one copy of the corpus instead of two. that
    matters in long-document mode: MLDR-it chunked to full coverage is ~25M
    vectors (~13gb fp32), and holding the padded copy alongside the originals
    does not fit in 27gb. only pass it when the caller has no further use for
    `doc_embeddings` — the list is emptied in place.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_docs = len(doc_embeddings)
    ranked_ids: list[list[str]] = []
    ranked_scores: list[list[float]] = []
    # only a python list can release its elements one at a time; a stacked
    # ndarray/tensor owns one buffer and freeing it piecewise is impossible
    consume = consume and isinstance(doc_embeddings, list)

    # pre-pad the corpus once into fixed chunks on cpu, then stream to gpu
    chunk_tensors: list[tuple[torch.Tensor, torch.Tensor, int, int]] = []
    for start in range(0, n_docs, doc_chunk):
        end = min(start + doc_chunk, n_docs)
        chunk = doc_embeddings[start:end]
        max_len = max(int(np.asarray(d).shape[0]) for d in chunk)
        dim = int(np.asarray(chunk[0]).shape[-1])
        padded = torch.zeros((len(chunk), max_len, dim), dtype=torch.float32)
        mask = torch.zeros((len(chunk), max_len), dtype=torch.bool)
        for i, d in enumerate(chunk):
            arr = np.asarray(d, dtype=np.float32)
            padded[i, : arr.shape[0]] = torch.from_numpy(arr)
            mask[i, : arr.shape[0]] = True
        chunk_tensors.append((padded, mask, start, end))
        if consume:
            del chunk
            for i in range(start, end):
                doc_embeddings[i] = None

    for qi, qe in enumerate(query_embeddings):
        q = torch.as_tensor(qe, dtype=torch.float32, device=device)
        if q.ndim != 2:
            q = q.reshape(q.shape[-2], q.shape[-1])
        all_scores = torch.empty(n_docs, dtype=torch.float32, device=device)
        for padded, mask, start, end in chunk_tensors:
            docs = padded.to(device, non_blocking=True)
            dmask = mask.to(device, non_blocking=True)
            sim = torch.einsum("qd,ntd->nqt", q, docs)  # (n, tq, td)
            sim = sim.masked_fill(~dmask.unsqueeze(1), float("-inf"))
            token_max = sim.max(dim=-1).values  # (n, tq)
            token_max = torch.nan_to_num(token_max, nan=0.0, neginf=0.0)
            all_scores[start:end] = token_max.sum(dim=-1)
        topk = min(k, n_docs)
        vals, idxs = torch.topk(all_scores, k=topk)
        ranked_ids.append([doc_ids[int(i)] for i in idxs.tolist()])
        ranked_scores.append([float(v) for v in vals.tolist()])
        if log_every and (qi + 1) % log_every == 0:
            logger.info("maxsim %s/%s queries", qi + 1, len(query_embeddings))
    return ranked_ids, ranked_scores


def chunk_long_documents(
    documents: list[dict],
    chunk_chars: int,
    overlap_chars: int = 0,
) -> tuple[list[dict], list[str]]:
    """split long documents into overlapping chunks for max-pooled scoring.

    a colbert encoder truncates at document_length tokens, so on MLDR-style
    corpora everything past the first ~512 tokens is invisible. chunking and
    taking the best chunk per document recovers that tail without retraining.

    returns the chunk documents plus the parent doc id of each chunk.
    """
    if chunk_chars <= 0:
        return documents, [d["id"] for d in documents]
    step = max(1, chunk_chars - max(0, overlap_chars))
    chunks: list[dict] = []
    parents: list[str] = []
    for doc in documents:
        text = doc["text"]
        if len(text) <= chunk_chars:
            chunks.append(doc)
            parents.append(doc["id"])
            continue
        for i, start in enumerate(range(0, len(text), step)):
            piece = text[start : start + chunk_chars]
            if not piece.strip():
                continue
            chunks.append({"id": f"{doc['id']}::chunk{i}", "text": piece})
            parents.append(doc["id"])
            if start + chunk_chars >= len(text):
                break
    logger.info(
        "chunked %s documents into %s chunks (chunk_chars=%s, overlap=%s)",
        len(documents),
        len(chunks),
        chunk_chars,
        overlap_chars,
    )
    return chunks, parents


def maxpool_chunks_to_documents(
    ranked: list[list[dict]],
    chunk_to_parent: dict[str, str],
    k: int,
) -> list[list[dict]]:
    """collapse a chunk-level ranking to a document ranking by best chunk score."""
    out: list[list[dict]] = []
    for row in ranked:
        best: dict[str, float] = {}
        for hit in row:
            parent = chunk_to_parent.get(hit["id"], hit["id"])
            score = float(hit["score"])
            if score > best.get(parent, float("-inf")):
                best[parent] = score
        ordered = sorted(best.items(), key=lambda kv: -kv[1])[:k]
        out.append([{"id": did, "score": sc} for did, sc in ordered])
    return out


ITALIAN_STOPWORDS = {
    "a", "ad", "agli", "ai", "al", "all", "alla", "alle", "allo", "anche",
    "che", "chi", "ci", "coi", "col", "come", "con", "cui", "da", "dai", "dal",
    "dall", "dalla", "dalle", "dallo", "degli", "dei", "del", "dell", "della",
    "delle", "dello", "di", "e", "ed", "gli", "ha", "hai", "hanno", "ho", "i",
    "il", "in", "io", "la", "le", "lei", "li", "lo", "loro", "lui", "ma", "mi",
    "ne", "negli", "nei", "nel", "nell", "nella", "nelle", "nello", "noi",
    "non", "o", "per", "perché", "più", "quale", "quanto", "quello", "questo",
    "sei", "si", "sia", "siamo", "sono", "su", "sugli", "sui", "sul", "sull",
    "sulla", "sulle", "sullo", "ti", "tra", "tu", "tuo", "un", "una", "uno",
    "vi", "voi", "è",
}


def make_italian_bm25_tokenizer() -> "Callable[[str], list[str]]":
    """italian analyzer for bm25: lowercase, strip punctuation, stopwords, stem.

    the previous `text.lower().split()` left punctuation glued to tokens and did
    no stemming, which understates the lexical baseline on an inflected language.
    falls back to stopword removal only if `snowballstemmer` is not installed.
    """
    import re

    token_re = re.compile(r"\w+", flags=re.UNICODE)
    try:
        import snowballstemmer

        stem = snowballstemmer.stemmer("italian").stemWord
    except Exception:  # noqa: BLE001
        logger.warning("snowballstemmer missing; bm25 runs unstemmed (weaker baseline)")

        def stem(word: str) -> str:
            return word

    def tokenize(text: str) -> list[str]:
        return [
            stem(tok)
            for tok in token_re.findall(text.lower())
            if tok not in ITALIAN_STOPWORDS and len(tok) > 1
        ]

    return tokenize


class BM25Retriever:
    def __init__(self, documents: list[dict], tokenizer=None):
        self.doc_ids = [d["id"] for d in documents]
        self.tokenizer = tokenizer or make_italian_bm25_tokenizer()
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
        encode_documents: bool = True,
        brute_force_limit: int = 25_000,
        chunk_chars: int = 0,
        chunk_overlap_chars: int = 0,
    ):
        from pylate import models

        # long-doc mode: index chunks, then max-pool chunk scores back per document
        self.chunk_to_parent: dict[str, str] | None = None
        if chunk_chars > 0:
            chunks, parents = chunk_long_documents(
                documents, chunk_chars=chunk_chars, overlap_chars=chunk_overlap_chars
            )
            self.chunk_to_parent = {
                c["id"]: p for c, p in zip(chunks, parents, strict=True)
            }
            documents = chunks

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

        self.retriever = None
        self.doc_emb = None
        if not encode_documents:
            self.use_bruteforce = False
            self._load_index_only(
                index_folder=index_folder,
                index_name=index_name,
            )
            return

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
        if not self.use_bruteforce:
            self._build_index(
                index_folder=index_folder,
                index_name=index_name,
                override_index=override_index,
            )

    def _load_index_only(self, index_folder: str, index_name: str) -> None:
        from pylate import indexes
        from pylate import retrieve as pylate_retrieve

        index_path = Path(index_folder) / index_name / "index.voyager"
        if not index_path.exists():
            raise FileNotFoundError(f"no saved index at {index_path}")
        logger.info("loading saved colbert index from %s (skip document encoding)", index_path)
        self.index = indexes.Voyager(
            index_folder=index_folder,
            index_name=index_name,
            override=False,
        )
        self.retriever = pylate_retrieve.ColBERT(index=self.index)

    def _build_index(
        self,
        index_folder: str,
        index_name: str,
        override_index: bool,
    ) -> None:
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
        # indexed embeddings live on disk/in the index backend; drop the
        # in-memory copy so retrieval over large corpora does not oom
        self.doc_emb = None
        import gc

        gc.collect()

    def retrieve(
        self,
        queries: list[str],
        k: int = 100,
        retrieve_batch_size: int = 5,
    ) -> list[list[dict]]:
        q_emb = self.model.encode(
            queries,
            batch_size=32,
            is_query=True,
            show_progress_bar=True,
        )
        # over-fetch when chunking so max-pooling still yields k distinct documents
        fetch_k = k * 4 if self.chunk_to_parent else k
        if self.use_bruteforce:
            # retrieve() is called once per model per benchmark, so the raw
            # embeddings are dead after this call — let maxsim free them as it
            # pads rather than holding two copies of a chunked corpus
            ranked_ids, ranked_scores = maxsim_topk(
                self.doc_emb, q_emb, self.doc_ids, k=fetch_k, log_every=20, consume=True
            )
            self.doc_emb = None
            ranked = scores_to_pylate([], ranked_ids, ranked_scores)
        else:
            ranked = self.retriever.retrieve(
                queries_embeddings=q_emb,
                k=fetch_k,
                batch_size=retrieve_batch_size,
            )
        if self.chunk_to_parent:
            ranked = maxpool_chunks_to_documents(ranked, self.chunk_to_parent, k=k)
        return ranked


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
