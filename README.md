# GroundedDocs

A deterministic, citation-grounded question-answering API for PDF documents. Upload a
PDF, and ask questions about it — the model is constrained to answer only from
retrieved chunks, cite the chunks it used, and **abstain** rather than guess when the
evidence isn't there.

Built with FastAPI, Postgres + [pgvector](https://github.com/pgvector/pgvector), and
the OpenAI API.

## Why "grounded"

Most RAG demos hand the LLM some context and hope for the best. This one enforces
grounding at the API layer instead of trusting the model's word for it:

- The system prompt requires the model to return structured JSON (`decision`,
  `answer`, `citations`, `abstain_reason`) rather than free text.
- Every citation the model returns is checked against the actual chunk IDs that were
  retrieved for that query — hallucinated citation IDs are stripped out.
- If the model answers but produces zero *valid* citations after that check, the
  response is downgraded to an abstain rather than served as a confident answer.
- If retrieval finds no relevant chunks at all, the model is never called — the API
  abstains immediately.

## How it works

```
PDF upload → content-hash dedup → store raw file
           → extract text (PyMuPDF) → fixed-size overlapping chunks → hash-dedup
           → embed each chunk (OpenAI text-embedding-3-small) → pgvector column
Question   → embed query → cosine-distance top-k retrieval (pgvector)
           → GPT-4o, constrained to the retrieved chunks, JSON-schema output
           → citations validated against retrieved chunk IDs → answer or abstain
```

Documents and chunks are deduplicated by SHA-256 content hash, so re-uploading the
same file or re-chunking an already-processed document is a no-op rather than
creating duplicates.

## API

| Endpoint | Description |
|---|---|
| `POST /documents/upload` | Upload a PDF file (multipart). Deduplicates by content hash. |
| `POST /documents/{document_id}/chunks` | Extract text and split it into overlapping chunks. |
| `POST /documents/{document_id}/embed` | Generate embeddings for any un-embedded chunks. |
| `POST /documents/retrieve` | `{ "query": str, "top_k": int }` — return the top-k most similar chunks. |
| `POST /documents/ask` | `{ "question": str, "top_k": int }` — retrieve + grounded answer with citations, or abstain. |

Interactive docs are available at `/docs` once the server is running.

## Tech stack

- **FastAPI** + **Pydantic** — API layer and request/response schemas
- **PostgreSQL** + **pgvector** — relational storage and vector similarity search (cosine distance)
- **SQLAlchemy** — ORM
- **PyMuPDF (fitz)** — PDF text extraction
- **OpenAI API** — `text-embedding-3-small` for embeddings, `gpt-4o` for grounded answering
- **Docker Compose** — local Postgres/pgvector instance

## Getting started

1. Copy `.env.example` to `.env` and fill in `DATABASE_URL` and `OPENAI_API_KEY`.
2. Start Postgres with pgvector:
   ```
   docker compose up -d
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Run the API:
   ```
   uvicorn app.main:app --reload
   ```
5. Try the flow against `http://localhost:8000/docs`:
   - `POST /documents/upload` a PDF
   - `POST /documents/{document_id}/chunks`
   - `POST /documents/{document_id}/embed`
   - `POST /documents/ask` with a question about the PDF's contents

## Current limitations / next steps

This is a baseline focused on getting the grounding behavior right end-to-end, not a
production-ready service yet:

- Requests run under a single hardcoded placeholder user — there's no real
  authentication/authorization layer.
- Chunking is fixed-size character splitting with overlap, not
  structure-aware (headings, tables, etc.).
- No automatic pipeline trigger — upload, chunk, and embed are separate calls a
  client must chain together.

## License

No license file yet — all rights reserved by default until one is added.
