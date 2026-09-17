import re
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import bm25s
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models.chunk import Chunk

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


@dataclass
class _CachedIndex:
    fingerprint: tuple[int, Optional[datetime]]
    chunk_ids: list[uuid.UUID]
    retriever: bm25s.BM25


# Process-memory cache, one BM25 index per user. No explicit invalidation hook
# anywhere else in the codebase: correctness comes from the fingerprint check
# in _get_or_build_index, not from remembering to clear this on every mutation.
_cache: dict[uuid.UUID, _CachedIndex] = {}


def _fingerprint(db: Session, user_id: uuid.UUID) -> tuple[int, Optional[datetime]]:
    stmt = select(func.count(Chunk.id), func.max(Chunk.created_at)).where(
        Chunk.user_id == user_id
    )
    count, max_created_at = db.execute(stmt).one()
    return (count, max_created_at)


def _get_or_build_index(db: Session, user_id: uuid.UUID) -> _CachedIndex:
    fingerprint = _fingerprint(db, user_id)

    cached = _cache.get(user_id)
    if cached is not None and cached.fingerprint == fingerprint:
        return cached

    stmt = select(Chunk.id, Chunk.content).where(Chunk.user_id == user_id)
    rows = db.execute(stmt).all()

    chunk_ids = [row.id for row in rows]
    texts = [row.content for row in rows]

    corpus_tokens = bm25s.tokenize(texts, stopwords="en", show_progress=False)
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens, show_progress=False)

    cached = _CachedIndex(fingerprint=fingerprint, chunk_ids=chunk_ids, retriever=retriever)
    _cache[user_id] = cached
    return cached


def bm25_retrieve(
    db: Session,
    user_id: uuid.UUID,
    query: str,
    k: int = 5,
) -> list[dict]:
    cached = _get_or_build_index(db, user_id)
    if not cached.chunk_ids:
        return []

    k = min(k, len(cached.chunk_ids))
    query_tokens = bm25s.tokenize([query], stopwords="en", show_progress=False)
    results, scores = cached.retriever.retrieve(query_tokens, k=k, show_progress=False)

    return [
        {"chunk_id": str(cached.chunk_ids[idx]), "score": float(score)}
        for idx, score in zip(results[0], scores[0])
    ]
