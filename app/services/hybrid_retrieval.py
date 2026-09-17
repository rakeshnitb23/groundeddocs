import uuid
from typing import Literal

from sqlalchemy.orm import Session

from app.services.bm25_retrieval import bm25_retrieve
from app.services.vector_store import retrieve_similar_chunks

RRF_K = 60
DEFAULT_N = 60

Mode = Literal["dense", "bm25", "hybrid"]
_VALID_MODES = ("dense", "bm25", "hybrid")


def _dense_ranked_ids(db: Session, user_id: uuid.UUID, query: str, n: int) -> list[str]:
    chunks = retrieve_similar_chunks(db=db, user_id=user_id, query=query, top_k=n)
    return [str(c.id) for c in chunks]


def _bm25_ranked_ids(db: Session, user_id: uuid.UUID, query: str, n: int) -> list[str]:
    hits = bm25_retrieve(db=db, user_id=user_id, query=query, k=n)
    return [hit["chunk_id"] for hit in hits]


def _rrf_fuse(dense_ids: list[str], bm25_ids: list[str], k_rrf: int = RRF_K) -> dict[str, dict]:
    """
    RRF over ranks, never raw scores — dense (cosine distance) and BM25 scores
    live on incomparable scales, so only rank position feeds the fused score.
    A chunk missing from one list simply doesn't get that list's term; it is
    never penalized for being absent, only rewarded for being present.
    """
    fused: dict[str, dict] = {}

    for rank, chunk_id in enumerate(dense_ids, start=1):
        entry = fused.setdefault(chunk_id, {"rrf_score": 0.0, "sources": []})
        entry["rrf_score"] += 1.0 / (k_rrf + rank)
        entry["sources"].append("dense")

    for rank, chunk_id in enumerate(bm25_ids, start=1):
        entry = fused.setdefault(chunk_id, {"rrf_score": 0.0, "sources": []})
        entry["rrf_score"] += 1.0 / (k_rrf + rank)
        entry["sources"].append("bm25")

    return fused


def hybrid_retrieve(
    db: Session,
    user_id: uuid.UUID,
    query: str,
    k: int = 5,
    mode: Mode = "hybrid",
    n: int = DEFAULT_N,
) -> list[dict]:
    if mode not in _VALID_MODES:
        raise ValueError(f"Unknown mode: {mode!r}, expected one of {_VALID_MODES}")

    # Over-retrieve only matters when fusing two lists; a single-source mode
    # just wants its own top-k, so there's no reason to over-fetch for it.
    fetch_n = n if mode == "hybrid" else k

    dense_ids = _dense_ranked_ids(db, user_id, query, fetch_n) if mode in ("dense", "hybrid") else []
    bm25_ids = _bm25_ranked_ids(db, user_id, query, fetch_n) if mode in ("bm25", "hybrid") else []

    fused = _rrf_fuse(dense_ids, bm25_ids)
    ranked = sorted(fused.items(), key=lambda item: item[1]["rrf_score"], reverse=True)

    return [
        {"chunk_id": chunk_id, "rrf_score": data["rrf_score"], "sources": data["sources"]}
        for chunk_id, data in ranked[:k]
    ]
