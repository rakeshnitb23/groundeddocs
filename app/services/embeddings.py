from app.core.config import settings
from app.core.llm import client


def get_embedding(text: str) -> list[float]:
    text = text.replace("\n", " ").strip()
    if not text:
        raise ValueError("Cannot embed empty text")

    response = client.embeddings.create(
        input=text,
        model=settings.EMBEDDING_MODEL,
    )
    return response.data[0].embedding
