"""
Manual Slice 4 check: run one unanswerable question and one answerable
question through the pre-generation gate and confirm each takes the
expected branch — using the calibrated ABSTENTION_TAU from config, with
zero LLM calls made in either case (retrieve_with_gate never calls the
LLM; that's Slice 5's job).

Usage:
    python check_abstention.py
"""

from app.api.documents import TEMP_USER_ID
from app.core.database import SessionLocal
from app.core.config import settings
from app.services.abstention import retrieve_with_gate

UNANSWERABLE_Q = "What is the best 6-week therapy protocol to increase conscientiousness?"
ANSWERABLE_Q = "In which age period did people increase most in social dominance, conscientiousness, and emotional stability?"


def main() -> None:
    print(f"Using ABSTENTION_TAU = {settings.ABSTENTION_TAU}\n")

    db = SessionLocal()
    try:
        print(f"Unanswerable question: {UNANSWERABLE_Q!r}")
        result = retrieve_with_gate(db, TEMP_USER_ID, UNANSWERABLE_Q, k=8)
        print(f"  decision:          {result['decision']}")
        print(f"  top_rerank_score:  {result['top_rerank_score']}")
        print(f"  abstain_reason:    {result['abstain_reason']}")
        assert result["decision"] == "abstain", "expected this to abstain before any LLM call"
        print("  PASS: abstained without calling the LLM\n")

        print(f"Answerable question: {ANSWERABLE_Q!r}")
        result = retrieve_with_gate(db, TEMP_USER_ID, ANSWERABLE_Q, k=8)
        print(f"  decision:          {result['decision']}")
        print(f"  top_rerank_score:  {result['top_rerank_score']}")
        print(f"  chunks passed on:  {len(result['chunks'])}")
        assert result["decision"] == "proceed", "expected this to proceed to generation"
        print("  PASS: proceeded with chunks ready for the (Slice 5) generator")
    finally:
        db.close()


if __name__ == "__main__":
    main()
