import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.chunk import Chunk
from app.services.embeddings import get_embedding


def embed_chunks_for_document(
    db: Session,
    document_id: uuid.UUID,
    user_id: uuid.UUID,
) -> int:
    stmt = (
        select(Chunk)
        .where(
            Chunk.document_id == document_id,
            Chunk.user_id == user_id,
            Chunk.embedding.is_(None),
        )
        .order_by(Chunk.chunk_index)
    )
    chunks = db.execute(stmt).scalars().all()

    count = 0
    for chunk in chunks:
        chunk.embedding = get_embedding(chunk.content)
        count += 1

    db.commit()
    return count


def retrieve_similar_chunks(
    db: Session,
    user_id: uuid.UUID,
    query: str,
    top_k: int = 5,
) -> list[Chunk]:
    query_embedding = get_embedding(query)

    stmt = (
        select(Chunk)
        .where(
            Chunk.user_id == user_id,
            Chunk.embedding.is_not(None),
        )
        .order_by(Chunk.embedding.cosine_distance(query_embedding))
        .limit(top_k)
    )
    return list(db.execute(stmt).scalars().all())
