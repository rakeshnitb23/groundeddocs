from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    DATABASE_URL: str
    OPENAI_API_KEY: str
    OPENAI_BASE_URL: str = "https://openrouter.ai/api/v1"
    CHAT_MODEL: str = "openai/gpt-4o"
    EMBEDDING_MODEL: str = "openai/text-embedding-3-small"

    class Config:
        env_file = ".env"


settings = Settings()
