import re
from typing import Literal, Optional

from pydantic import BaseModel, Field

from app.core.config import settings
from app.core.llm import client

_WHITESPACE_RE = re.compile(r"\s+")


def _normalize_whitespace(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text).strip()

GENERATION_SYSTEM_PROMPT = """
You are a careful assistant that answers questions ONLY using the provided document chunks.

Rules (must follow strictly):
1. Use ONLY the information present in the given chunks. Never use outside knowledge.
2. If the chunks do not contain enough evidence to answer confidently, set status to
   "abstained" instead of guessing.
3. Every factual claim in your answer must be supported by at least one chunk you cite.
4. For each citation, give the exact chunk_id (copied character-for-character from the
   chunks below) and a short quote copied verbatim from that chunk's text that supports
   your claim.
5. Never cite a chunk_id that was not given to you below. Never fabricate a quote —
   if you can't find an exact supporting phrase, don't cite that chunk.

Respond with a JSON object matching exactly this schema:
{
  "status": "answered" or "abstained",
  "answer": "string or null",
  "citations": [{"chunk_id": "string", "quote": "string or null"}]
}
If status is "abstained", answer must be null and citations must be an empty list.
"""

RETRY_NOTE = (
    "Your previous response cited a chunk_id or quote that does not exist in the "
    "chunks provided below. Use only the exact chunk_ids given, with quotes copied "
    "verbatim from that chunk's text, or set status to \"abstained\" if you cannot."
)


class Citation(BaseModel):
    chunk_id: str
    quote: Optional[str] = None


class GenerationOutput(BaseModel):
    status: Literal["answered", "abstained"]
    answer: Optional[str] = None
    citations: list[Citation] = Field(default_factory=list)


def build_prompt(query: str, chunks: list[dict], retry_note: str = "") -> str:
    parts = [f"Question: {query}\n", "Available chunks:"]
    for chunk in chunks:
        parts.append(f"\n--- chunk_id: {chunk['chunk_id']} ---\n{chunk['text']}\n")
    parts.append(
        "\nUsing only the chunks above, answer with citations, or abstain if the "
        "evidence is insufficient."
    )
    if retry_note:
        parts.append(f"\n{retry_note}")
    return "\n".join(parts)


def validate_citations(chunks: list[dict], citations: list[Citation]) -> list[str]:
    """
    Server-side gate on what the model claimed, independent of whether the model
    followed instructions. A chunk_id that isn't in the retrieved set is dropped
    outright — the model cannot cite its way outside what was actually retrieved
    for this user. A quote that doesn't literally occur in its cited chunk is
    dropped too, catching a citation that points at a real chunk_id but invents
    the supporting text. Whitespace is normalized on both sides first: chunk
    text preserves the PDF's original line-wrap newlines, which a model
    naturally flattens to spaces when it quotes them, so comparing raw strings
    would reject faithful quotes over formatting alone.
    """
    text_by_id = {
        chunk["chunk_id"]: _normalize_whitespace(chunk["text"]) for chunk in chunks
    }
    valid_ids: list[str] = []

    for citation in citations:
        if citation.chunk_id not in text_by_id:
            continue
        if citation.quote and _normalize_whitespace(citation.quote) not in text_by_id[citation.chunk_id]:
            continue
        if citation.chunk_id not in valid_ids:
            valid_ids.append(citation.chunk_id)

    return valid_ids


def _call_llm(query: str, chunks: list[dict], retry_note: str = "") -> GenerationOutput:
    prompt = build_prompt(query, chunks, retry_note)
    response = client.chat.completions.create(
        model=settings.CHAT_MODEL,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": GENERATION_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    return GenerationOutput.model_validate_json(response.choices[0].message.content)


def generate_answer(query: str, chunks: list[dict], max_retries: int = 1) -> dict:
    """
    chunks: [{chunk_id, text}] — only what the gate already decided to pass on.
    Does not retrieve, rerank, or look at any score; user_id isolation is the
    caller's responsibility (chunks already belong to that user by the time
    they get here).
    """
    if not chunks:
        return {"status": "abstained", "answer": None, "cited_chunk_ids": []}

    retry_note = ""
    for _ in range(max_retries + 1):
        parsed = _call_llm(query, chunks, retry_note)

        if parsed.status == "abstained":
            return {"status": "abstained", "answer": None, "cited_chunk_ids": []}

        valid_ids = validate_citations(chunks, parsed.citations)
        if valid_ids:
            return {"status": "answered", "answer": parsed.answer, "cited_chunk_ids": valid_ids}

        retry_note = RETRY_NOTE

    return {"status": "abstained", "answer": None, "cited_chunk_ids": []}
