from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    DATABASE_URL: str
    OPENAI_API_KEY: str
    OPENAI_BASE_URL: str = "https://openrouter.ai/api/v1"
    CHAT_MODEL: str = "openai/gpt-4o"
    EMBEDDING_MODEL: str = "openai/text-embedding-3-small"
    # Pre-generation abstention gate (Slice 4): below this top rerank_score,
    # the ask path abstains without calling the LLM. Single source of truth —
    # everything reads this instead of hardcoding a threshold. Provisional
    # default; see eval/calibrate_tau.py and eval/results/tau_calibration.json
    # for how this was derived and why it should be re-run as the eval set grows.
    ABSTENTION_TAU: float = 0.8641

    class Config:
        env_file = ".env"


settings = Settings()
