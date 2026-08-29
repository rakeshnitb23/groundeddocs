from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from app.core.config import settings
from app.core.llm import client

QUESTIONS_IN = Path("eval/questions_personality_meta.json")
CHUNKS_IN = Path("eval/chunks.json")
QUESTIONS_OUT = Path("eval/questions_with_gold.proposed.json")

JUDGE_PROMPT = """You are labeling retrieval evidence.
Question type: {qtype}
Question: {question}

Chunk ID: {chunk_id}
Chunk text:
{chunk_text}

Does this chunk contain evidence that can support an answer to the question?
Do not answer the question.
Reply as JSON only:
{{"relevant": true or false, "span": "short quote or empty string"}}
"""


def judge_chunk(question: str, qtype: str, chunk: dict) -> tuple[bool, str]:
    resp = client.chat.completions.create(
        model=settings.CHAT_MODEL,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "user",
                "content": JUDGE_PROMPT.format(
                    qtype=qtype,
                    question=question,
                    chunk_id=chunk["id"],
                    chunk_text=chunk["content"][:4000],
                ),
            }
        ],
    )
    raw = resp.choices[0].message.content
    if not raw:
        return False, ""
    data = json.loads(raw)
    return bool(data.get("relevant")), data.get("span") or ""


def main():
    questions = json.loads(QUESTIONS_IN.read_text())
    chunks = json.loads(CHUNKS_IN.read_text())

    out = []
    for q in questions:
        print(f"Judging {q['id']} ({q['type']}) against {len(chunks)} chunks")
        proposed = []
        notes = []

        if q["type"] == "unanswerable":
            out.append(
                {
                    "id": q["id"],
                    "question": q["question"],
                    "type": q["type"],
                    "gold_chunk_ids": [],
                    "notes": "unanswerable — gold must stay empty after your review",
                }
            )
            continue

        for i, chunk in enumerate(chunks, start=1):
            if i % 50 == 0:
                print(f"  {q['id']}: judged {i}/{len(chunks)}")
            try:
                relevant, span = judge_chunk(q["question"], q["type"], chunk)
            except Exception as e:
                print(f"  skip {chunk['id']}: {e}")
                continue
            if relevant:
                proposed.append(chunk["id"])
                notes.append(f"{chunk['id']}: {span[:180]}")

        out.append(
            {
                "id": q["id"],
                "question": q["question"],
                "type": q["type"],
                "gold_chunk_ids": proposed[:5],
                "notes": " | ".join(notes[:5]),
            }
        )

    QUESTIONS_OUT.write_text(json.dumps(out, indent=2))
    print(f"Wrote proposed gold to {QUESTIONS_OUT}")
    print("NOW review it. Copy to eval/questions_with_gold.json after you accept/reject IDs.")


if __name__ == "__main__":
    main()
