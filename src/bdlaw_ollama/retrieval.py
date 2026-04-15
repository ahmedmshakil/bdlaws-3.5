from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from .config import AppConfig
from .utils import compact_whitespace, ensure_dir, iter_jsonl, load_pickle, normalize_digits, save_pickle, write_jsonl


def _lazy_import_faiss():
    try:
        import faiss
    except ModuleNotFoundError as exc:  # pragma: no cover - dependency path
        raise RuntimeError("Missing dependency `faiss-cpu`. Install project dependencies before building the index.") from exc

    return faiss


def _lazy_import_sentence_transformer():
    try:
        from sentence_transformers import SentenceTransformer
    except ModuleNotFoundError as exc:  # pragma: no cover - dependency path
        raise RuntimeError("Missing dependency `sentence-transformers`. Install project dependencies before building the index.") from exc

    return SentenceTransformer


def _embedding_device() -> str:
    try:
        import torch
    except ModuleNotFoundError:  # pragma: no cover - dependency path
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _embedding_batch_size(device: str) -> int:
    return 8 if device == "cuda" else 32


def _lazy_import_numpy():
    try:
        import numpy as np
    except ModuleNotFoundError as exc:  # pragma: no cover - dependency path
        raise RuntimeError("Missing dependency `numpy`. Install project dependencies before running retrieval commands.") from exc

    return np


def _lazy_import_bm25():
    try:
        from rank_bm25 import BM25Okapi
    except ModuleNotFoundError as exc:  # pragma: no cover - dependency path
        raise RuntimeError("Missing dependency `rank-bm25`. Install project dependencies before building the index.") from exc

    return BM25Okapi


def normalize_search_text(text: str) -> str:
    text = normalize_digits(text).lower()
    return compact_whitespace(text)


def tokenize(text: str) -> list[str]:
    return [token for token in normalize_search_text(text).split() if token]


def reciprocal_rank_fusion(rankings: list[list[int]], k: int = 60) -> dict[int, float]:
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, item_id in enumerate(ranking, start=1):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank)
    return scores


def load_chunks(config: AppConfig) -> list[dict[str, Any]]:
    return list(iter_jsonl(config.path("processed_dir") / "chunks.jsonl"))


def build_index(config: AppConfig) -> dict[str, Any]:
    chunks = load_chunks(config)
    if not chunks:
        raise RuntimeError("No chunks found. Run `bdlaw prepare-data` first.")

    np = _lazy_import_numpy()
    BM25Okapi = _lazy_import_bm25()
    SentenceTransformer = _lazy_import_sentence_transformer()
    faiss = _lazy_import_faiss()

    model_name = config.project["models"]["embedding_model"]
    device = _embedding_device()
    model = SentenceTransformer(model_name, device=device)
    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(
        texts,
        batch_size=_embedding_batch_size(device),
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    embeddings = np.asarray(embeddings, dtype="float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    bm25 = BM25Okapi([tokenize(text) for text in texts])

    retrieval_dir = ensure_dir(config.path("retrieval_dir"))
    faiss.write_index(index, str(retrieval_dir / "faiss.index"))
    save_pickle(retrieval_dir / "bm25.pkl", bm25)
    np.save(retrieval_dir / "dense.npy", embeddings)
    write_jsonl(retrieval_dir / "chunk_metadata.jsonl", chunks)
    return {"chunks": len(chunks), "embedding_dim": int(embeddings.shape[1]), "retrieval_dir": str(retrieval_dir)}


@lru_cache(maxsize=2)
def load_index(retrieval_dir: str, embedding_model: str) -> dict[str, Any]:
    faiss = _lazy_import_faiss()
    SentenceTransformer = _lazy_import_sentence_transformer()
    retrieval_path = Path(retrieval_dir)
    device = _embedding_device()
    model = SentenceTransformer(embedding_model, device=device)
    return {
        "faiss": faiss.read_index(str(retrieval_path / "faiss.index")),
        "bm25": load_pickle(retrieval_path / "bm25.pkl"),
        "chunks": list(iter_jsonl(retrieval_path / "chunk_metadata.jsonl")),
        "model": model,
    }


def retrieve_chunks(question: str, config: AppConfig, top_k: int | None = None) -> dict[str, Any]:
    np = _lazy_import_numpy()
    retrieval_cfg = config.rag["retrieval"]
    use_top_k = top_k or int(retrieval_cfg["top_k"])
    loaded = load_index(str(config.path("retrieval_dir")), config.project["models"]["embedding_model"])
    model = loaded["model"]
    faiss_index = loaded["faiss"]
    bm25 = loaded["bm25"]
    metadata = loaded["chunks"]

    query_embedding = np.asarray(
        model.encode(
            [question],
            batch_size=1,
            show_progress_bar=False,
            normalize_embeddings=True,
        ),
        dtype="float32",
    )
    dense_scores, dense_ids = faiss_index.search(query_embedding, use_top_k)
    dense_scores = dense_scores[0]
    dense_ids = dense_ids[0]

    bm25_scores_all = bm25.get_scores(tokenize(question))
    bm25_sorted = np.argsort(bm25_scores_all)[::-1][:use_top_k]

    fused = reciprocal_rank_fusion(
        [dense_ids.tolist(), bm25_sorted.tolist()],
        k=int(retrieval_cfg["rrf_k"]),
    )

    dense_score_map = {int(idx): float(score) for idx, score in zip(dense_ids.tolist(), dense_scores.tolist()) if idx >= 0}
    bm25_score_map = {int(idx): float(bm25_scores_all[idx]) for idx in bm25_sorted.tolist()}

    ranked = sorted(fused.items(), key=lambda item: item[1], reverse=True)[: int(retrieval_cfg["rerank_top_k"])]
    selected: list[dict[str, Any]] = []
    for idx, fused_score in ranked:
        row = dict(metadata[idx])
        row["dense_score"] = dense_score_map.get(idx, 0.0)
        row["bm25_score"] = bm25_score_map.get(idx, 0.0)
        row["fused_score"] = float(fused_score)
        selected.append(row)

    best_dense = max((row["dense_score"] for row in selected), default=0.0)
    best_bm25 = max((row["bm25_score"] for row in selected), default=0.0)
    threshold_dense = float(retrieval_cfg["dense_score_threshold"])
    threshold_bm25 = float(retrieval_cfg["bm25_min_score"])
    grounded = best_dense >= threshold_dense or best_bm25 >= threshold_bm25

    return {
        "question": question,
        "grounded": grounded,
        "best_dense_score": best_dense,
        "best_bm25_score": best_bm25,
        "chunks": selected[: int(retrieval_cfg["max_context_chunks"])],
        "all_ranked_chunks": selected,
    }
