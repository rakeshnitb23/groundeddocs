"""
Manual Slice 3 check: run one query through hybrid retrieval, then through
the cross-encoder reranker on top of that same pool, and print both orders.

Expected shape: the same chunk_ids appear in both lists (recall of the pool
is unchanged), but the order differs, and the gold chunk (if given) should
move up or stay at the top after reranking.

Usage:
    python check_rerank.py "3,976" ba8ace2e-1869-4f57-8daf-35f33b57842f
"""

import sys

from app.api.documents import TEMP_USER_ID
from app.core.database import SessionLocal
from app.services.hybrid_retrieval import hybrid_retrieve
from app.services.retrieval import RERANK_POOL_SIZE, retrieve


def _rank_of(chunk_id: str, ids: list[str]) -> str:
    return str(ids.index(chunk_id) + 1) if chunk_id in ids else f"not in top {len(ids)}"


def main(query: str, gold_chunk_id: str = "", k: int = 8) -> None:
    db = SessionLocal()
    try:
        print(f"Query: {query!r}\n")

        hybrid_pool = hybrid_retrieve(db, TEMP_USER_ID, query, k=RERANK_POOL_SIZE, mode="hybrid")
        hybrid_ids = [h["chunk_id"] for h in hybrid_pool]

        print(f"--- Hybrid order (pool of {len(hybrid_ids)}) ---")
        for cid in hybrid_ids[:k]:
            print(f"  {cid}")

        reranked = retrieve(db, TEMP_USER_ID, query, k=k, mode="hybrid_rerank")
        reranked_ids = [r["chunk_id"] for r in reranked]

        print(f"\n--- Reranked order (top {k}) ---")
        for r in reranked:
            print(f"  {r['chunk_id']}  rerank_score={r['rerank_score']:.4f}")

        same_set = set(hybrid_ids[:len(reranked_ids)]) if len(hybrid_ids) >= len(reranked_ids) else None
        print(f"\nAll reranked IDs came from the hybrid pool: {set(reranked_ids).issubset(set(hybrid_ids))}")

        if gold_chunk_id:
            print(f"\nGold chunk {gold_chunk_id}:")
            print(f"  hybrid rank:   {_rank_of(gold_chunk_id, hybrid_ids)}")
            print(f"  reranked rank: {_rank_of(gold_chunk_id, reranked_ids)}")
    finally:
        db.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(1)
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "")
