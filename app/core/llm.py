from openai import OpenAI

from app.core.config import settings

# OpenRouter is OpenAI-compatible; the key in OPENAI_API_KEY is an OpenRouter key.
client = OpenAI(
    api_key=settings.OPENAI_API_KEY,
    base_url=settings.OPENAI_BASE_URL,
)
