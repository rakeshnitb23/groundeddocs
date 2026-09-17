"""
Manual Slice 2 check: run one rare-identifier query through dense-only,
BM25-only, and hybrid (RRF-fused) retrieval, and print all three lists.

Expected shape: the gold chunk ranks poorly (or is absent) in the dense
list, ranks well in the BM25 list, and still surfaces in the final hybrid
top-k on the strength of its BM25 rank alone.

Usage:
    python check_hybrid.py "3,976" ba8ace2e-1869-4f57-8daf-35f33b57842f
"""

import sys

from app.api.documents import TEMP_USER_ID
from app.core.database import SessionLocal
from app.services.hybrid_retrieval import DEFAULT_N, _bm25_ranked_ids, _dense_ranked_ids, hybrid_retrieve


def _rank_of(chunk_id: str, ranked_ids: list[str]) -> str:
    return str(ranked_ids.index(chunk_id) + 1) if chunk_id in ranked_ids else f"not in top {len(ranked_ids)}"


def main(query: str, gold_chunk_id: str = "", k: int = 5) -> None:
    db = SessionLocal()
    try:
        print(f"Query: {query!r}\n")

        dense_ids = _dense_ranked_ids(db, TEMP_USER_ID, query, DEFAULT_N)
        bm25_ids = _bm25_ranked_ids(db, TEMP_USER_ID, query, DEFAULT_N)

        print(f"--- Dense top-{DEFAULT_N} ---")
        for cid in dense_ids:
            print(f"  {cid}")

        print(f"\n--- BM25 top-{DEFAULT_N} ---")
        for cid in bm25_ids:
            print(f"  {cid}")

        print(f"\n--- Hybrid (RRF-fused) top-{k} ---")
        hybrid_hits = hybrid_retrieve(db, TEMP_USER_ID, query, k=k, mode="hybrid")
        for hit in hybrid_hits:
            print(f"  {hit['chunk_id']}  rrf_score={hit['rrf_score']:.5f}  sources={hit['sources']}")

        if gold_chunk_id:
            hybrid_ids = [h["chunk_id"] for h in hybrid_hits]
            print(f"\nGold chunk {gold_chunk_id}:")
            print(f"  dense rank:  {_rank_of(gold_chunk_id, dense_ids)}")
            print(f"  bm25 rank:   {_rank_of(gold_chunk_id, bm25_ids)}")
            print(f"  hybrid rank: {_rank_of(gold_chunk_id, hybrid_ids)}")
    finally:
        db.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(1)
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "")
