from typing import Optional

from sentence_transformers import CrossEncoder

RERANKER_MODEL = "BAAI/bge-reranker-base"
MAX_LENGTH = 512  # tokens; the tokenizer truncates silently past this, so we flag it ourselves

_model: Optional[CrossEncoder] = None


def _get_model() -> CrossEncoder:
    global _model
    if _model is None:
        _model = CrossEncoder(RERANKER_MODEL, max_length=MAX_LENGTH)
    return _model


def _flag_truncated(model: CrossEncoder, query: str, candidates: list[dict]) -> None:
    tokenizer = model.tokenizer
    for candidate in candidates:
        # encode() returns a plain list[int], not the dict/Encoding union __call__
        # returns, so its length is unambiguous. verbose=False: we're deliberately
        # building an over-length sequence just to measure it, not feeding it to
        # the model, so the tokenizer's own truncation warning here is expected
        # noise, not a real problem.
        length = len(tokenizer.encode(query, candidate["text"], truncation=False, verbose=False))
        if length > MAX_LENGTH:
            print(
                f"[reranker] chunk {candidate['chunk_id']} is {length} tokens "
                f"(query+text), truncated to {MAX_LENGTH} before scoring"
            )


def rerank(query: str, candidates: list[dict], top_n: int = 8) -> list[dict]:
    """
    Cross-encoder rerank over an already-retrieved candidate pool.

    candidates: [{chunk_id, text}, ...] — never searches the corpus itself,
    only reorders what it's handed. Scores query and chunk_text jointly
    (unlike RRF or cosine similarity, which never let the two interact),
    which is what lets it catch things like negation or exact confirmation
    that rank-fusion can't.
    """
    if not candidates:
        return []

    model = _get_model()
    _flag_truncated(model, query, candidates)

    pairs = [(query, c["text"]) for c in candidates]
    scores = model.predict(pairs)

    scored = [
        {"chunk_id": c["chunk_id"], "text": c["text"], "rerank_score": float(score)}
        for c, score in zip(candidates, scores)
    ]
    scored.sort(key=lambda item: item["rerank_score"], reverse=True)
    return scored[:top_n]
