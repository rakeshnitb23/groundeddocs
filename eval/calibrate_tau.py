"""
Slice 4 calibration: run every labeled eval question through hybrid_rerank
retrieval, log its top rerank_score, then sweep tau to see which threshold
would have correctly abstained on unanswerable questions without incorrectly
refusing answerable ones.

Uses the "type" label from eval/questions_personality_meta.json (answerable /
partial / unanswerable), NOT the gold_chunk_ids in questions_with_gold.json —
those are explicitly marked as unverified placeholders in that file and are
irrelevant to this calibration anyway (abstention only needs the label and
the top score, not gold chunk identity).

Usage:
    python eval/calibrate_tau.py
"""

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from app.api.documents import TEMP_USER_ID
from app.core.database import SessionLocal
from app.services.retrieval import retrieve

QUESTIONS_PATH = Path("eval/questions_personality_meta.json")
OUT_PATH = Path("eval/results/tau_calibration.json")
K = 8  # top of the reranked list; only rows[0]["rerank_score"] is used here


def log_scores(db, questions: list[dict]) -> list[dict]:
    rows = []
    for q in questions:
        candidates = retrieve(db, TEMP_USER_ID, q["question"], k=K, mode="hybrid_rerank")
        top = candidates[0] if candidates else None
        row = {
            "question_id": q["id"],
            "label": q["type"],
            "top_chunk_id": top["chunk_id"] if top else None,
            "top_rerank_score": top["rerank_score"] if top else None,
        }
        rows.append(row)
        print(f"{row['question_id']:5} label={row['label']:12} top_rerank_score={row['top_rerank_score']}")
    return rows


def _confusion(scored: list[dict], tau: float) -> dict:
    """
    Positive class = "should abstain" = label == unanswerable.
      TP: unanswerable, correctly abstains            (correct_abstain)
      FN: unanswerable, incorrectly answers            (false_answer  -- unsafe)
      FP: answerable,   incorrectly abstains            (false_refusal -- annoying)
      TN: answerable,   correctly answers               (correct_answer)
    """
    tp = fn = fp = tn = 0
    for r in scored:
        would_abstain = r["top_rerank_score"] < tau
        is_unanswerable = r["label"] == "unanswerable"
        if is_unanswerable and would_abstain:
            tp += 1
        elif is_unanswerable and not would_abstain:
            fn += 1
        elif not is_unanswerable and would_abstain:
            fp += 1
        else:
            tn += 1
    return {"tp": tp, "fn": fn, "fp": fp, "tn": tn}


def sweep_tau(rows: list[dict]) -> list[dict]:
    # partial is excluded: there's no "correct" abstain/answer expectation for
    # it (see eval/run_eval.py's own classify_case, which treats it the same way).
    scored = [r for r in rows if r["top_rerank_score"] is not None and r["label"] != "partial"]
    scores = sorted({r["top_rerank_score"] for r in scored})
    if not scores:
        return []

    # Midpoints between consecutive observed scores, plus one tau below the
    # minimum (abstains on nothing) and one above the maximum (abstains on
    # everything), so the sweep covers every distinct decision boundary.
    candidate_taus = [scores[0] - 0.01]
    candidate_taus += [(a + b) / 2 for a, b in zip(scores, scores[1:])]
    candidate_taus += [scores[-1] + 0.01]

    sweep = []
    for tau in candidate_taus:
        c = _confusion(scored, tau)
        precision = c["tp"] / (c["tp"] + c["fp"]) if (c["tp"] + c["fp"]) else None
        recall = c["tp"] / (c["tp"] + c["fn"]) if (c["tp"] + c["fn"]) else None
        sweep.append(
            {
                "tau": round(tau, 4),
                "abstention_precision": round(precision, 3) if precision is not None else None,
                "abstention_recall": round(recall, 3) if recall is not None else None,
                "false_answers": c["fn"],
                "false_refusals": c["fp"],
                "correct_abstains": c["tp"],
                "correct_answers": c["tn"],
            }
        )
    return sweep


def choose_tau(sweep: list[dict]) -> dict:
    # False answers (hallucination risk) are strictly worse than false
    # refusals (annoying but safe) for a system whose whole premise is
    # "abstain rather than guess" — so first minimize false answers, then
    # false refusals, then prefer the smallest (least aggressive) tau.
    return min(sweep, key=lambda s: (s["false_answers"], s["false_refusals"], s["tau"]))


def main():
    questions = json.loads(QUESTIONS_PATH.read_text())
    db = SessionLocal()
    try:
        rows = log_scores(db, questions)
    finally:
        db.close()

    sweep = sweep_tau(rows)
    n_partial = sum(1 for r in rows if r["label"] == "partial")
    n_missing = sum(1 for r in rows if r["top_rerank_score"] is None)
    n_scored = len(rows) - n_partial - n_missing

    result = {
        "rows": rows,
        "sweep": sweep,
        "n_questions": len(rows),
        "n_scored_answerable_or_unanswerable": n_scored,
        "n_partial_excluded_from_metrics": n_partial,
        "n_missing_scores": n_missing,
    }

    if not sweep:
        result["chosen_tau"] = None
        result["note"] = (
            "Calibration incomplete: no answerable/unanswerable question had a "
            "rerank score to sweep over."
        )
    else:
        chosen = choose_tau(sweep)
        result["chosen_tau"] = chosen["tau"]
        result["chosen_tau_stats"] = chosen
        result["selection_rule"] = (
            "Minimize false_answers first, then false_refusals, then prefer the "
            "smallest (least aggressive) tau among ties."
        )
        result["caveat"] = (
            f"Calibrated on only {n_scored} labeled answerable/unanswerable questions "
            "against a single ingested document. This is a small-sample estimate, not "
            "a statistically robust threshold — re-run this script as more eval "
            "questions and documents are added, and update ABSTENTION_TAU in "
            "app/core/config.py accordingly."
        )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(result, indent=2))

    print("\n=== Tau sweep ===")
    for s in sweep:
        print(s)
    print(f"\nChosen tau: {result.get('chosen_tau')}")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
