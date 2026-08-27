import json
import time
from collections import Counter
from pathlib import Path

import requests

BASE_URL = "http://127.0.0.1:8000"
QUESTIONS_PATH = Path("eval/questions_personality_meta.json")
RESULTS_PATH = Path("eval/results/raw_results.json")
SUMMARY_PATH = Path("eval/results/summary.json")
TOP_K = 5


def expected_decision(qtype: str) -> str:
    if qtype == "unanswerable":
        return "abstain"
    if qtype == "answerable":
        return "answer"
    return "either"  # partial


def classify_case(qtype: str, decision: str) -> str:
    if qtype == "answerable" and decision == "answer":
        return "correct_answer"
    if qtype == "answerable" and decision == "abstain":
        return "false_refusal"
    if qtype == "unanswerable" and decision == "abstain":
        return "correct_abstain"
    if qtype == "unanswerable" and decision == "answer":
        return "false_answer"
    if qtype == "partial" and decision in {"answer", "abstain"}:
        return "partial_ok_for_now"
    return "unexpected"


def main():
    questions = json.loads(QUESTIONS_PATH.read_text())
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for q in questions:
        start = time.perf_counter()
        try:
            resp = requests.post(
                f"{BASE_URL}/documents/ask",
                json={"question": q["question"], "top_k": TOP_K},
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()
            error = None
        except Exception as e:
            data = {
                "decision": None,
                "answer": None,
                "citations": [],
                "abstain_reason": None,
            }
            error = str(e)

        latency_ms = round((time.perf_counter() - start) * 1000, 1)
        decision = data.get("decision")
        citations = data.get("citations") or []

        row = {
            "id": q["id"],
            "type": q["type"],
            "question": q["question"],
            "expected": expected_decision(q["type"]),
            "decision": decision,
            "answer": data.get("answer"),
            "citations": citations,
            "citation_count": len(citations),
            "abstain_reason": data.get("abstain_reason"),
            "latency_ms": latency_ms,
            "case": classify_case(q["type"], decision) if decision else "request_failed",
            "error": error,
        }
        rows.append(row)
        print(f"{q['id']}  type={q['type']:13}  decision={decision}  case={row['case']}")

    RESULTS_PATH.write_text(json.dumps(rows, indent=2))

    counts = Counter(r["case"] for r in rows)
    answerable = [r for r in rows if r["type"] == "answerable"]
    unanswerable = [r for r in rows if r["type"] == "unanswerable"]
    answered = [r for r in rows if r["decision"] == "answer"]

    n_ans = len(answerable) or 1
    n_unans = len(unanswerable) or 1
    n_answered = len(answered) or 1

    summary = {
        "n_questions": len(rows),
        "case_counts": dict(counts),
        "answer_rate_on_answerable": round(
            sum(1 for r in answerable if r["decision"] == "answer") / n_ans, 3
        ),
        "abstention_recall": round(
            sum(1 for r in unanswerable if r["decision"] == "abstain") / n_unans, 3
        ),
        "false_answer_rate": round(
            sum(1 for r in unanswerable if r["decision"] == "answer") / n_unans, 3
        ),
        "false_refusal_rate": round(
            sum(1 for r in answerable if r["decision"] == "abstain") / n_ans, 3
        ),
        "answers_with_at_least_one_citation": round(
            sum(1 for r in answered if r["citation_count"] > 0) / n_answered, 3
        ),
        "avg_latency_ms": round(sum(r["latency_ms"] for r in rows) / len(rows), 1),
        "note": (
            "Citation presence is automatic. Citation FAITHFULNESS "
            "(does the cited text support the claim) still needs a manual pass "
            "on eval/results/raw_results.json."
        ),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))

    print("\n=== Slice 5 baseline ===")
    print(json.dumps(summary, indent=2))
    print(f"\nWrote {RESULTS_PATH}")
    print(f"Wrote {SUMMARY_PATH}")


if __name__ == "__main__":
    main()