import uuid

from openai import OpenAI
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.chunk import Chunk
from app.schemas.answer import AnswerResponse
from app.services.vector_store import retrieve_similar_chunks

client = OpenAI(api_key=settings.OPENAI_API_KEY)

SYSTEM_PROMPT = """
You are a careful assistant that answers questions ONLY using the provided document chunks.

Rules (must follow strictly):
1. You may ONLY use the information present in the given chunks.
2. If the chunks do not contain enough evidence to answer confidently, you MUST abstain.
3. When you answer, every factual claim must be supported by one or more of the provided chunks.
4. Return the IDs of the chunks you actually used as citations.
5. Never invent information or use outside knowledge.

You must respond with a JSON object that matches this schema:
{
  "decision": "answer" or "abstain",
  "answer": "string or null",
  "citations": ["chunk-id-1", "chunk-id-2"],
  "abstain_reason": "string or null"
}
"""


def build_user_prompt(question: str, chunks: list[Chunk]) -> str:
    parts = [f"Question: {question}\n", "Available chunks:"]
    for chunk in chunks:
        parts.append(f"\n--- Chunk ID: {chunk.id} ---\n{chunk.content}\n")
    parts.append(
        "\nBased only on the chunks above, either answer the question with citations "
        "or abstain if the evidence is insufficient."
    )
    return "\n".join(parts)


def answer_question(
    db: Session,
    user_id: uuid.UUID,
    question: str,
    top_k: int = 5,
) -> AnswerResponse:
    chunks = retrieve_similar_chunks(
        db=db,
        user_id=user_id,
        query=question,
        top_k=top_k,
    )

    if not chunks:
        return AnswerResponse(
            decision="abstain",
            abstain_reason="No relevant documents found for this user.",
        )

    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(question, chunks)},
        ],
    )

    parsed = AnswerResponse.model_validate_json(response.choices[0].message.content)
    valid_ids = {c.id for c in chunks}
    parsed.citations = [cid for cid in parsed.citations if cid in valid_ids]

    if parsed.decision == "answer" and not parsed.citations:
        return AnswerResponse(
            decision="abstain",
            abstain_reason="Model produced an answer without valid citations.",
        )

    return parsed
