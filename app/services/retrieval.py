import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.chunk import Chunk
from app.services.hybrid_retrieval import DEFAULT_N, hybrid_retrieve
from app.services.reranking import rerank

# Size of the hybrid pool handed to the reranker. The reranker only ever
# reorders this pool — it never searches the corpus — so this should match
# Slice 2's own over-retrieve width, not the final k the caller wants.
RERANK_POOL_SIZE = DEFAULT_N


def _load_chunk_texts(db: Session, user_id: uuid.UUID, chunk_ids: list[str]) -> dict[str, str]:
    ids = [uuid.UUID(cid) for cid in chunk_ids]
    stmt = select(Chunk.id, Chunk.content).where(
        Chunk.user_id == user_id,
        Chunk.id.in_(ids),
    )
    rows = db.execute(stmt).all()
    return {str(row.id): row.content for row in rows}


def retrieve(
    db: Session,
    user_id: uuid.UUID,
    query: str,
    k: int = 5,
    mode: str = "hybrid",
) -> list[dict]:
    if mode != "hybrid_rerank":
        return hybrid_retrieve(db, user_id, query, k=k, mode=mode)

    pool = hybrid_retrieve(db, user_id, query, k=RERANK_POOL_SIZE, mode="hybrid")
    if not pool:
        return []

    # Re-filtering by user_id here (not just trusting pool's chunk_ids) means
    # isolation holds even if hybrid_retrieve's guarantee were ever broken.
    texts = _load_chunk_texts(db, user_id, [hit["chunk_id"] for hit in pool])
    candidates = [
        {"chunk_id": hit["chunk_id"], "text": texts[hit["chunk_id"]]}
        for hit in pool
        if hit["chunk_id"] in texts
    ]

    return rerank(query, candidates, top_n=k)
