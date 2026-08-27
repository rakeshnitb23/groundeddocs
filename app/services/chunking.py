import hashlib
import uuid
from pathlib import Path

import fitz
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.chunk import Chunk
from app.models.document import Document


def compute_text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def simple_chunk_text(text: str, chunk_size: int = 600, overlap: int = 80) -> list[str]:
    text = text.strip()
    if not text:
        return []

    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
        start = max(end - overlap, 0)
        if start >= end:
            start = end
    return chunks


def create_chunks_for_document(
    db: Session,
    document_id: uuid.UUID,
    user_id: uuid.UUID,
) -> list[Chunk]:
    document = db.get(Document, document_id)
    if not document:
        raise ValueError("Document not found")
    if document.user_id != user_id:
        raise ValueError("Document does not belong to this user")

    pdf_path = Path(document.storage_path)
    if not pdf_path.exists():
        raise ValueError("Raw PDF file not found on disk")

    doc = fitz.open(pdf_path)
    full_text = "\n".join(page.get_text() for page in doc)
    doc.close()

    text_chunks = simple_chunk_text(full_text)
    created: list[Chunk] = []

    for index, chunk_text in enumerate(text_chunks):
        content_hash = compute_text_hash(chunk_text)
        stmt = select(Chunk).where(
            Chunk.document_id == document_id,
            Chunk.content_hash == content_hash,
        )
        existing = db.execute(stmt).scalar_one_or_none()
        if existing:
            continue

        chunk = Chunk(
            document_id=document_id,
            user_id=user_id,
            content_hash=content_hash,
            content=chunk_text,
            chunk_index=index,
        )
        db.add(chunk)
        created.append(chunk)

    db.commit()
    for chunk in created:
        db.refresh(chunk)
    return created
