import uuid
from typing import Optional

from sqlalchemy.orm import Session

from app.core.config import settings
from app.services.retrieval import retrieve


def should_abstain(top_rerank_score: float, tau: float) -> bool:
    """
    Ignores dense cosine distance and RRF's rank-derived score entirely — RRF
    is not a confidence measure (Slice 2/3) and must never be thresholded.
    Only the cross-encoder's rerank_score, the one signal in this pipeline
    that actually looked at query and chunk together, decides this.
    """
    return top_rerank_score < tau


def retrieve_with_gate(
    db: Session,
    user_id: uuid.UUID,
    question: str,
    k: int = 5,
    tau: Optional[float] = None,
) -> dict:
    """
    Slice 4's ask-path gate: hybrid_rerank retrieval, then abstain-or-proceed.
    Deliberately stops here rather than calling the LLM — wiring this into
    real generation (the prompt, the citation contract) is Slice 5's job, not
    this one's. A caller that gets decision="proceed" is handed exactly the
    chunks it would pass to the generator; a caller that gets "abstain" has
    made zero LLM calls.
    """
    tau = settings.ABSTENTION_TAU if tau is None else tau

    candidates = retrieve(db, user_id, question, k=k, mode="hybrid_rerank")

    if not candidates:
        return {
            "decision": "abstain",
            "abstain_reason": "No candidates retrieved.",
            "top_rerank_score": None,
            "tau": tau,
            "chunks": [],
        }

    top_score = candidates[0]["rerank_score"]

    if should_abstain(top_score, tau):
        return {
            "decision": "abstain",
            "abstain_reason": f"Top rerank score {top_score:.4f} is below tau={tau}.",
            "top_rerank_score": top_score,
            "tau": tau,
            "chunks": [],
        }

    return {
        "decision": "proceed",
        "abstain_reason": None,
        "top_rerank_score": top_score,
        "tau": tau,
        "chunks": candidates,
    }
