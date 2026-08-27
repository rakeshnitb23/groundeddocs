import uuid
from datetime import datetime

from pydantic import BaseModel


class DocumentOut(BaseModel):
    id: uuid.UUID
    user_id: uuid.UUID
    original_filename: str
    content_hash: str
    storage_path: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True
