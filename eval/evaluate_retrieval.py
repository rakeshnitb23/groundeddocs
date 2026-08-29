import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import requests

QUESTIONS_PATH = Path("eval/questions_with_gold.json")
OUT_PATH = Path("eval/results/retrieval_baseline.json")
BASE_URL = "http://127.0.0.1:8000"
K = 10


def dense_retriever(question: str, k: int = K) -> list:
    resp = requests.post(
        f"{BASE_URL}/documents/retrieve",
        json={"query": question, "top_k": k},
        timeout=60,
    )
    resp.raise_for_status()
    return [item["id"] for item in resp.json()]


def recall_at_k(gold, ranked, k):
    if not gold:
        return None
    hit = set(gold) & set(ranked[:k])
    return len(hit) / len(gold)


def precision_at_k(gold, ranked, k):
    if not gold:
        return None
    top = ranked[:k]
    if not top:
        return 0.0
    return len(set(gold) & set(top)) / len(top)


def mrr(gold, ranked):
    if not gold:
        return None
    gold_set = set(gold)
    for i, cid in enumerate(ranked, start=1):
        if cid in gold_set:
            return 1.0 / i
    return 0.0


def evaluate_retrieval(retriever_fn, questions, k=K):
    rows = []
    for q in questions:
        if q["type"] == "unanswerable":
            continue
        gold = q.get("gold_chunk_ids") or []
        if not gold:
            continue
        ranked = retriever_fn(q["question"], k)
        rows.append(
            {
                "id": q["id"],
                "type": q["type"],
                "gold_chunk_ids": gold,
                "retrieved_ids": ranked,
                "recall@k": recall_at_k(gold, ranked, k),
                "precision@k": precision_at_k(gold, ranked, k),
                "mrr": mrr(gold, ranked),
            }
        )

    n = len(rows) or 1
    return {
        "k": k,
        "n_scored_questions": len(rows),
        "recall@k": round(sum(r["recall@k"] for r in rows) / n, 3),
        "precision@k": round(sum(r["precision@k"] for r in rows) / n, 3),
        "mrr": round(sum(r["mrr"] for r in rows) / n, 3),
        "per_question": rows,
    }


def main():
    if not QUESTIONS_PATH.exists():
        raise SystemExit(
            f"{QUESTIONS_PATH} is missing. Copy the proposed file and review gold IDs first:\n"
            "  cp eval/questions_with_gold.proposed.json eval/questions_with_gold.json"
        )
    questions = json.loads(QUESTIONS_PATH.read_text())
    summary = evaluate_retrieval(dense_retriever, questions, k=K)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(summary, indent=2))
    print(
        json.dumps(
            {
                k: summary[k]
                for k in ["k", "n_scored_questions", "recall@k", "precision@k", "mrr"]
            },
            indent=2,
        )
    )
    print(f"Wrote {OUT_PATH}")
    print("Freeze these numbers as the Week 2 dense-only baseline.")


if __name__ == "__main__":
    main()
