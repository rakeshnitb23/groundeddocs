"""
Manual Slice 1 check: run one query through both retrievers and print both
top-k lists side by side. Point this at a query built around a rare token
(an ID, an error code, a specific proper noun) that appears in one of your
already-uploaded and already-embedded documents.

Usage:
    python check_bm25.py "what is invoice INV-88213-XZ for?"
"""

import sys

from app.api.documents import TEMP_USER_ID
from app.core.database import SessionLocal
from app.services.bm25_retrieval import bm25_retrieve
from app.services.vector_store import retrieve_similar_chunks


def main(query: str, top_k: int = 5) -> None:
    db = SessionLocal()
    try:
        print(f"Query: {query!r}\n")

        print("--- Dense (pgvector cosine) ---")
        for chunk in retrieve_similar_chunks(db=db, user_id=TEMP_USER_ID, query=query, top_k=top_k):
            print(f"  {chunk.id}  {chunk.content[:80]!r}")

        print("\n--- Sparse (BM25) ---")
        for hit in bm25_retrieve(db=db, user_id=TEMP_USER_ID, query=query, k=top_k):
            print(f"  {hit['chunk_id']}  score={hit['score']:.3f}")
    finally:
        db.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(1)
    main(sys.argv[1])
