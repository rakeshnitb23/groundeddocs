import hashlib
import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.document import Document
from app.services.storage import save_raw_pdf


def compute_content_hash(file_bytes: bytes) -> str:
    return hashlib.sha256(file_bytes).hexdigest()


def ingest_document(
    db: Session,
    user_id: uuid.UUID,
    filename: str,
    file_bytes: bytes,
) -> Document:
    content_hash = compute_content_hash(file_bytes)

    stmt = select(Document).where(
        Document.user_id == user_id,
        Document.content_hash == content_hash,
    )
    existing = db.execute(stmt).scalar_one_or_none()
    if existing:
        return existing

    storage_path = save_raw_pdf(file_bytes, filename)
    document = Document(
        user_id=user_id,
        original_filename=filename,
        content_hash=content_hash,
        storage_path=storage_path,
    )
    db.add(document)
    db.commit()
    db.refresh(document)
    return document
