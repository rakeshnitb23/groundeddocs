GroundedDocs — deterministic grounded baseline

1. Copy .env.example to .env and set OPENAI_API_KEY
2. docker compose up -d
3. pip install -r requirements.txt
4. uvicorn app.main:app --reload

Flow:
POST /documents/upload
POST /documents/{document_id}/chunks
POST /documents/{document_id}/embed
POST /documents/retrieve
POST /documents/ask
