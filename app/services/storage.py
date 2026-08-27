import uuid
from pathlib import Path

UPLOAD_DIR = Path("storage/raw_documents")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def save_raw_pdf(file_bytes: bytes, original_filename: str) -> str:
    suffix = Path(original_filename).suffix or ".pdf"
    filename = f"{uuid.uuid4()}{suffix}"
    path = UPLOAD_DIR / filename
    path.write_bytes(file_bytes)
    return str(path)
