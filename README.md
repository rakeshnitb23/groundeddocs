# GroundedDocs

A private document Q&A API that only answers when it can back the answer with a real
citation from the user's own documents — and abstains when it can't.

## Why this exists

Most "upload a PDF, chat with it" demos look convincing in a five-minute walkthrough
and then fall apart in ways that only show up once you actually try to trust the
output:

- The UI shows a "source," but the cited passage doesn't actually support the claim
  the model made.
- The model answers anyway when the documents don't contain enough evidence, because
  answering looks more impressive than refusing.
- Re-uploading or re-processing a file quietly creates duplicate chunks and duplicate
  embedding cost.
- "It answered something" gets treated as success, when the real question is whether
  the answer was grounded.
- Multi-user document isolation gets bolted on later instead of designed in from the
  first schema.

I wanted to build something that treats those as the actual product requirements,
not as edge cases to clean up eventually. GroundedDocs is a small, honest baseline
for a document assistant that is designed around trust first: faithful citations,
calibrated refusal, idempotent ingestion, and per-user isolation, before anything
else.

## Design decisions

These are the calls I made and why — the reasoning is the point of this project as
much as the code is.

**Citations get checked, not trusted.** The model is forced into a structured JSON
response (`decision`, `answer`, `citations`, `abstain_reason`) instead of free text
with inline references. After the model responds, every citation is checked against
the actual chunk IDs that were retrieved for that query — anything else gets
stripped. If an "answer" ends up with zero valid citations after that check, it's
downgraded to an abstain before it's ever returned. I didn't want to rely on the
model's word that a citation is real; the API verifies it.

**Abstention happens at generation time, not as a retrieval-score cutoff.** Cosine
top-k will always return *something* — the top 5 closest chunks exist even when none
of them actually answer the question. A fixed distance threshold felt like it would
either abstain too eagerly on legitimately relevant chunks or leak through irrelevant
ones, depending on the corpus. Instead, the retrieved chunks are handed to the model
with an explicit instruction to abstain if they don't support the question, and the
abstain path is the default when retrieval returns nothing at all. This can improve
later with a real threshold or re-ranker, but I didn't want to fake precision I
hadn't measured yet.

**Idempotency by content hash, at both the document and chunk level.** Documents are
deduplicated by a SHA-256 hash of the raw file bytes; chunks are deduplicated by a
hash of their own text. Re-uploading the same PDF, or re-running chunking on a
document that's already been chunked, is a no-op instead of creating duplicate rows
and paying for duplicate embeddings. This was a deliberate schema-level decision
(unique constraints on `(user_id, content_hash)` and `(document_id, content_hash)`),
not something patched on after finding duplicates in the data.

**User isolation lives in the schema, not in application logic I have to remember to
apply.** Every document and chunk row carries a `user_id`, and retrieval filters on
it directly in the query. I'd rather cross-user leakage be structurally impossible
than correct "as long as every endpoint remembers to filter."

**Schema changes go through Alembic, not `create_all` on startup.** The API assumes
the database is already migrated. `alembic upgrade head` creates the `vector`
extension and the tables; startup only seeds the temporary user. That keeps schema
changes reviewable and repeatable instead of implicit.

**HNSW indexing, hybrid search, and re-ranking are deliberately deferred, not
missing by accident.** The current corpus is small enough that a flat cosine scan
over the chunk table is fine. Adding an approximate index or a re-ranking pass now
would be optimizing a problem I don't have yet. It's a documented next step, not a
gap I'm hoping nobody notices.

**Chunking is naive character-splitting, and I know it.** Fixed-size chunks with
overlap are cheap to reason about and easy to verify, but they don't respect
headings, tables, or multi-column layout. I chose to get the grounding and
abstention behavior right first, on top of a chunking strategy simple enough to not
be a confounding variable, rather than solve chunking and trust at the same time.

## How it works

```
PDF upload → content-hash dedup → store raw file
           → extract text (PyMuPDF) → fixed-size overlapping chunks → hash-dedup
           → embed each chunk (OpenAI text-embedding-3-small) → pgvector column
Question   → embed query → cosine-distance top-k retrieval, filtered by user_id
           → GPT-4o, constrained to the retrieved chunks, JSON-schema output
           → citations validated against retrieved chunk IDs → answer or abstain
```

### Data model

- `users` — `id`
- `documents` — `id`, `user_id`, `original_filename`, `content_hash` (raw PDF
  bytes), `storage_path`, timestamps. Unique on `(user_id, content_hash)`.
- `chunks` — `id`, `document_id`, `user_id`, `content_hash` (chunk text), `content`,
  `chunk_index`, `embedding` (1536-dim vector), timestamps. Unique on
  `(document_id, content_hash)`.

Hierarchy: User → Document → Chunk → Vector.

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
- **Alembic** — versioned schema migrations (`alembic upgrade head`)
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
4. Apply database migrations (creates the `vector` extension and tables):
   ```
   alembic upgrade head
   ```
5. Run the API:
   ```
   uvicorn app.main:app --reload
   ```
6. Try the flow against `http://localhost:8000/docs`:
   - `POST /documents/upload` a PDF
   - `POST /documents/{document_id}/chunks`
   - `POST /documents/{document_id}/embed`
   - `POST /documents/ask` with a question about the PDF's contents

## Status: what's actually verified vs. what's designed-for

The pipeline above runs end to end, but I want to be specific about what's proven
and what's still a claim I intend to back up rather than something already checked:

**Built and working:** upload → chunk → embed → retrieve → grounded answer/abstain,
with citation validation and user-scoped retrieval, all exercised manually through
`/docs`.

**Designed for, not yet test-verified:** the dedup logic hasn't been run through an
automated test that re-uploads and re-chunks the same file and asserts zero
duplicate rows — I've relied on the unique constraints and manual checks so far.

**Not built yet:** an evaluation harness. The plan is a frozen set of questions
(answerable / partially answerable / unanswerable from the corpus) scored on
citation-support rate, abstention precision and recall, and retrieval quality —
reported separately from each other, not folded into one "accuracy" number. Without
that, "grounded" is currently a property I've engineered for and spot-checked, not
one I've measured.

**Known accepted gap:** chunking, as noted above — naive character splitting will
underperform on tables, multi-column PDFs, and long headers/footers. This is a
tracked tradeoff, not an oversight.

There's also no request pipeline stitching upload → chunk → embed together
automatically yet — a client has to call all three in sequence.

## A note on how this was built

The architecture, the schema, and every tradeoff described above — idempotency by
content hash, abstention as a generation-time decision instead of a distance
threshold, isolation enforced in the schema, deferring HNSW and re-ranking, accepting
naive chunking for now — were decisions I made, and each one took actual back-and-forth
to land on. I used AI tooling to help with implementation and scaffolding, but the
reasoning above is mine, not generated.

## License

No license file yet — all rights reserved by default until one is added.
