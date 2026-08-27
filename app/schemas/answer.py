import uuid
from typing import Literal, Optional

from pydantic import BaseModel, Field


class AnswerResponse(BaseModel):
    decision: Literal["answer", "abstain"]
    answer: Optional[str] = None
    citations: list[uuid.UUID] = Field(default_factory=list)
    abstain_reason: Optional[str] = None


class QuestionRequest(BaseModel):
    question: str
    top_k: int = 5
