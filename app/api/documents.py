import uuid

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.schemas.answer import AnswerResponse, QuestionRequest
from app.schemas.document import DocumentOut
from app.services.answering import answer_question
from app.services.chunking import create_chunks_for_document
from app.services.ingestion import ingest_document
from app.services.vector_store import embed_chunks_for_document, retrieve_similar_chunks

router = APIRouter(prefix="/documents", tags=["documents"])

TEMP_USER_ID = uuid.UUID("00000000-0000-0000-0000-000000000001")


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5


@router.post("/upload", response_model=DocumentOut)
async def upload_document(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Empty file")

    return ingest_document(
        db=db,
        user_id=TEMP_USER_ID,
        filename=file.filename,
        file_bytes=file_bytes,
    )


@router.post("/{document_id}/chunks")
def create_chunks(
    document_id: uuid.UUID,
    db: Session = Depends(get_db),
):
    try:
        chunks = create_chunks_for_document(
            db=db,
            document_id=document_id,
            user_id=TEMP_USER_ID,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return [
        {
            "id": str(c.id),
            "chunk_index": c.chunk_index,
            "content_hash": c.content_hash,
            "content_preview": c.content[:120],
        }
        for c in chunks
    ]


@router.post("/{document_id}/embed")
def embed_document_chunks(
    document_id: uuid.UUID,
    db: Session = Depends(get_db),
):
    count = embed_chunks_for_document(
        db=db,
        document_id=document_id,
        user_id=TEMP_USER_ID,
    )
    return {"embedded_chunks": count}


@router.post("/retrieve")
def retrieve(
    body: QueryRequest,
    db: Session = Depends(get_db),
):
    chunks = retrieve_similar_chunks(
        db=db,
        user_id=TEMP_USER_ID,
        query=body.query,
        top_k=body.top_k,
    )
    return [
        {
            "id": str(c.id),
            "document_id": str(c.document_id),
            "chunk_index": c.chunk_index,
            "content": c.content,
            "content_hash": c.content_hash,
        }
        for c in chunks
    ]


@router.post("/ask", response_model=AnswerResponse)
def ask_question(
    body: QuestionRequest,
    db: Session = Depends(get_db),
):
    return answer_question(
        db=db,
        user_id=TEMP_USER_ID,
        question=body.question,
        top_k=body.top_k,
    )
