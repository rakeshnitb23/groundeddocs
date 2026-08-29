import json
import uuid
from pathlib import Path

from app.core.database import SessionLocal
from app.models.chunk import Chunk

TEMP_USER_ID = uuid.UUID("00000000-0000-0000-0000-000000000001")
OUT = Path("eval/chunks.json")


def main():
    db = SessionLocal()
    try:
        chunks = (
            db.query(Chunk)
            .filter(Chunk.user_id == TEMP_USER_ID)
            .order_by(Chunk.document_id, Chunk.chunk_index)
            .all()
        )
        payload = [
            {
                "id": str(c.id),
                "document_id": str(c.document_id),
                "chunk_index": c.chunk_index,
                "content": c.content,
            }
            for c in chunks
        ]
    finally:
        db.close()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {len(payload)} chunks to {OUT}")


if __name__ == "__main__":
    main()