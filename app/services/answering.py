import uuid

from sqlalchemy.orm import Session

from app.schemas.answer import AnswerResponse
from app.services.abstention import retrieve_with_gate
from app.services.generation import generate_answer


def answer_question(
    db: Session,
    user_id: uuid.UUID,
    question: str,
    top_k: int = 5,
) -> AnswerResponse:
    gate_result = retrieve_with_gate(db, user_id, question, k=top_k)

    if gate_result["decision"] == "abstain":
        return AnswerResponse(decision="abstain", abstain_reason=gate_result["abstain_reason"])

    chunks = [{"chunk_id": c["chunk_id"], "text": c["text"]} for c in gate_result["chunks"]]
    result = generate_answer(question, chunks)

    if result["status"] == "abstained":
        return AnswerResponse(
            decision="abstain",
            abstain_reason="Model could not produce a fully cited answer from the retrieved chunks.",
        )

    return AnswerResponse(
        decision="answer",
        answer=result["answer"],
        citations=[uuid.UUID(cid) for cid in result["cited_chunk_ids"]],
    )
