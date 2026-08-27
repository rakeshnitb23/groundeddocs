from openai import OpenAI

from app.core.config import settings

client = OpenAI(api_key=settings.OPENAI_API_KEY)

EMBEDDING_MODEL = "text-embedding-3-small"


def get_embedding(text: str) -> list[float]:
    text = text.replace("\n", " ").strip()
    if not text:
        raise ValueError("Cannot embed empty text")

    response = client.embeddings.create(
        input=text,
        model=EMBEDDING_MODEL,
    )
    return response.data[0].embedding
