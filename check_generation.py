"""
Manual Slice 5 check, two parts:

1. Server-side citation validation (deterministic, no LLM involved): a
   fabricated citation with a chunk_id outside the retrieved set, and a
   citation whose quote doesn't actually occur in its own chunk, must both
   be rejected on their own — this isn't something we can reliably provoke
   a real model into doing on cue, so it's tested directly.

2. End to end against real data: an answerable question should get an
   answer with only valid citations; a genuinely insufficient-evidence case
   should abstain at the generation layer itself (independent of Slice 4's
   tau gate, which is exercised separately in check_abstention.py).

Usage:
    python check_generation.py
"""

from app.api.documents import TEMP_USER_ID
from app.core.database import SessionLocal
from app.services.generation import Citation, generate_answer, validate_citations
from app.services.retrieval import retrieve


def check_citation_validation() -> None:
    print("--- Part 1: server-side citation validation (no LLM) ---")
    chunks = [
        {"chunk_id": "real-chunk-1", "text": "The sample size was 3,976 participants."},
        {"chunk_id": "real-chunk-2", "text": "Conscientiousness increased in young adulthood."},
    ]

    citations = [
        Citation(chunk_id="real-chunk-1", quote="The sample size was 3,976 participants."),
        Citation(chunk_id="fake-chunk-999", quote="This chunk_id was never retrieved."),
        Citation(chunk_id="real-chunk-2", quote="This exact sentence is not in that chunk."),
    ]

    valid = validate_citations(chunks, citations)
    print(f"  input citations:  {[c.chunk_id for c in citations]}")
    print(f"  accepted:         {valid}")

    assert valid == ["real-chunk-1"], "expected only the genuinely valid citation to survive"
    print("  PASS: fake chunk_id rejected, unsupported quote rejected, real one kept\n")


def check_end_to_end() -> None:
    print("--- Part 2: end to end against real data ---")
    db = SessionLocal()
    try:
        answerable_q = "In which age period did people increase most in social dominance, conscientiousness, and emotional stability?"
        candidates = retrieve(db, TEMP_USER_ID, answerable_q, k=8, mode="hybrid_rerank")
        chunks = [{"chunk_id": c["chunk_id"], "text": c["text"]} for c in candidates]
        valid_ids = {c["chunk_id"] for c in chunks}

        result = generate_answer(answerable_q, chunks)
        print(f"  answerable question -> status: {result['status']}")
        print(f"  cited_chunk_ids: {result['cited_chunk_ids']}")
        assert result["status"] == "answered"
        assert set(result["cited_chunk_ids"]).issubset(valid_ids), "citation escaped the retrieved set"
        print("  PASS: answered with citations, all within the retrieved set\n")

        # A "partial" eval question: real retrieved chunks, but evidence is
        # deliberately thin/ambiguous — this tests generate_answer's OWN
        # abstain decision, not the tau gate (which this call bypasses).
        partial_q = "Does this study prove that psychotherapy can change personality traits?"
        candidates = retrieve(db, TEMP_USER_ID, partial_q, k=8, mode="hybrid_rerank")
        chunks = [{"chunk_id": c["chunk_id"], "text": c["text"]} for c in candidates]

        result = generate_answer(partial_q, chunks)
        print(f"  partial-evidence question -> status: {result['status']}")
        if result["status"] == "abstained":
            print("  PASS: generation layer abstained rather than overreach on thin evidence\n")
        else:
            print(f"  cited_chunk_ids: {result['cited_chunk_ids']}")
            print("  (model chose to answer this one — not a failure, just worth reading the answer)\n")
    finally:
        db.close()


if __name__ == "__main__":
    check_citation_validation()
    check_end_to_end()
