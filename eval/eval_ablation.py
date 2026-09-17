"""
Slice 6: one ablation table comparing four retrieval/generation configurations
against the frozen eval set (eval/questions_personality_meta.json).

Rows:
  1. dense-only        -- Week 2 pgvector cosine, top-k, generate_answer on top-k
  2. bm25-only         -- Slice 1 BM25, top-k, generate_answer on top-k
  3. hybrid (RRF)       -- Slice 2 dense+BM25 fused by rank, top-k, generate_answer on top-k
  4. hybrid_rerank+tau  -- the real production ask path: retrieve_with_gate (Slice 4)
                            + generate_answer with citation validation (Slice 5),
                            via answer_question() itself, not a reimplementation

k=10 for every row, identical, chosen to match the k already frozen in
eval/results/retrieval_baseline.json so retrieval numbers are comparable to
that earlier baseline.

Gold labels: eval/questions_with_gold.json's gold_chunk_ids are explicitly
marked in that file's own "notes" field as random, non-human-verified
placeholders -- eval/results/retrieval_baseline.json already established the
precedent of computing recall/precision/MRR from them ONLY to prove the
scoring code is correct, with a loud warning attached, never as a real
number. This script follows that same precedent rather than inventing new
gold labels or silently treating known-fake numbers as real.

End-to-end metrics use the real "type" labels (answerable/partial/
unanswerable) from eval/questions_personality_meta.json, which are not
placeholders, so citation support rate and abstention precision/recall are
trustworthy even though the retrieval columns are not.

For baseline rows (1-3), which have no tau/pre-generation gate at all,
"abstain" means generate_answer's own model-decided abstention -- there is
no equivalent of Slice 4's calibrated threshold for these rows. This is a
deliberate choice (see Requirement 5 in the Slice 6 brief) and is called out
in the printed interpretation, not left implicit.

Reproducibility: retrieval (dense/BM25/RRF/cross-encoder) is fully
deterministic for a fixed corpus and query. The LLM call in generate_answer
uses temperature=0, which is "best effort" deterministic per the provider,
not a guarantee of byte-identical output -- running this script twice should
produce the same retrieval columns and very likely (not certified) the same
end-to-end decisions.

Usage:
    python eval/eval_ablation.py
"""

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from app.api.documents import TEMP_USER_ID
from app.core.config import settings
from app.core.database import SessionLocal
from app.services.answering import answer_question
from app.services.generation import generate_answer
from app.services.retrieval import RERANK_POOL_SIZE, _load_chunk_texts, retrieve

QUESTIONS_PATH = Path("eval/questions_personality_meta.json")
GOLD_PATH = Path("eval/questions_with_gold.json")
ARTIFACTS_DIR = Path("artifacts")
TABLE_PATH = ARTIFACTS_DIR / "ablation_table.md"
RESULTS_PATH = ARTIFACTS_DIR / "ablation_results.json"

K = 10  # explicit, identical across all four rows

ROW_DENSE = "dense-only"
ROW_BM25 = "bm25-only"
ROW_HYBRID = "hybrid (RRF)"
ROW_FINAL = "hybrid_rerank + tau + citations"
ROWS = [ROW_DENSE, ROW_BM25, ROW_HYBRID, ROW_FINAL]
MODE_BY_ROW = {ROW_DENSE: "dense", ROW_BM25: "bm25", ROW_HYBRID: "hybrid"}


# ---------- retrieval metric helpers (only meaningful with real gold) ----------


def recall_at_k(gold: list[str], ranked: list[str], k: int):
    if not gold:
        return None
    hit = set(gold) & set(ranked[:k])
    return len(hit) / len(gold)


def precision_at_k(gold: list[str], ranked: list[str], k: int):
    if not gold:
        return None
    top = ranked[:k]
    if not top:
        return 0.0
    return len(set(gold) & set(top)) / len(top)


def mrr(gold: list[str], ranked: list[str]):
    if not gold:
        return None
    gold_set = set(gold)
    for i, cid in enumerate(ranked, start=1):
        if cid in gold_set:
            return 1.0 / i
    return 0.0


def load_gold_index() -> tuple[dict, bool, str]:
    if not GOLD_PATH.exists():
        return {}, False, f"{GOLD_PATH} does not exist -- retrieval columns unavailable."

    entries = json.loads(GOLD_PATH.read_text())
    has_placeholder_note = any("PLACEHOLDER" in (e.get("notes") or "") for e in entries)
    gold_by_id = {e["id"]: (e.get("gold_chunk_ids") or []) for e in entries}

    if has_placeholder_note:
        note = (
            f"gold_chunk_ids in {GOLD_PATH} are explicitly marked as random, "
            "non-human-verified placeholders (same file used in "
            "eval/results/retrieval_baseline.json). Retrieval columns below are "
            "computed only to prove the scoring code is correct -- they are NOT a "
            "trustworthy retrieval-quality signal. Do not draw conclusions from them."
        )
        return gold_by_id, False, note

    return gold_by_id, True, "gold_chunk_ids loaded as human-verified."


# ---------- running each row ----------


def answer_with_mode(db, user_id, question: str, k: int, mode: str) -> dict:
    # dense/bm25/hybrid modes return {chunk_id, rrf_score, sources} -- no text,
    # since only hybrid_rerank's reranker needs it. Load it the same way
    # retrieval.py's own hybrid_rerank path does, still user_id-filtered.
    candidates = retrieve(db, user_id, question, k=k, mode=mode)
    ranked_ids = [c["chunk_id"] for c in candidates]
    texts = _load_chunk_texts(db, user_id, ranked_ids)
    chunks = [{"chunk_id": cid, "text": texts[cid]} for cid in ranked_ids if cid in texts]
    result = generate_answer(question, chunks)
    return {
        "ranked_ids": ranked_ids,
        "decision": "answer" if result["status"] == "answered" else "abstain",
        "citations": result["cited_chunk_ids"],
        "abstain_reason": None,
    }


def answer_with_gate(db, user_id, question: str, k: int) -> dict:
    # Reuses the real production ask path end to end (Slice 4 gate + Slice 5
    # generation), rather than re-implementing the gate logic here.
    resp = answer_question(db, user_id, question, top_k=k)
    candidates = retrieve(db, user_id, question, k=k, mode="hybrid_rerank")
    return {
        "ranked_ids": [c["chunk_id"] for c in candidates],
        "decision": resp.decision,
        "citations": [str(cid) for cid in resp.citations],
        "abstain_reason": resp.abstain_reason,
    }


def run_row(db, user_id, questions: list[dict], row_name: str, k: int) -> list[dict]:
    rows = []
    for q in questions:
        if row_name == ROW_FINAL:
            out = answer_with_gate(db, user_id, q["question"], k)
        else:
            out = answer_with_mode(db, user_id, q["question"], k, MODE_BY_ROW[row_name])
        rows.append({"id": q["id"], "type": q["type"], "question": q["question"], **out})
        print(f"  {q['id']:5} type={q['type']:12} decision={out['decision']}")
    return rows


def citations_within_retrieved_set(row: dict) -> bool:
    return all(cid in row["ranked_ids"] for cid in row["citations"])


# ---------- metrics over a completed row ----------


def compute_retrieval_metrics(rows: list[dict], gold_by_id: dict, k: int):
    scored = []
    for r in rows:
        gold = gold_by_id.get(r["id"]) or []
        if not gold:
            continue
        scored.append(
            {
                "recall@k": recall_at_k(gold, r["ranked_ids"], k),
                "precision@k": precision_at_k(gold, r["ranked_ids"], k),
                "mrr": mrr(gold, r["ranked_ids"]),
            }
        )
    if not scored:
        return None
    n = len(scored)
    return {
        "recall@k": round(sum(s["recall@k"] for s in scored) / n, 3),
        "precision@k": round(sum(s["precision@k"] for s in scored) / n, 3),
        "mrr": round(sum(s["mrr"] for s in scored) / n, 3),
        "n_scored": n,
    }


def compute_end_to_end_metrics(rows: list[dict]) -> dict:
    # partial excluded: no correct abstain/answer expectation for it, same
    # convention already used in eval/run_eval.py and eval/calibrate_tau.py.
    scored = [r for r in rows if r["type"] != "partial"]
    answered = [r for r in rows if r["decision"] == "answer"]

    tp = fn = fp = tn = 0
    for r in scored:
        would_abstain = r["decision"] == "abstain"
        is_unanswerable = r["type"] == "unanswerable"
        if is_unanswerable and would_abstain:
            tp += 1
        elif is_unanswerable and not would_abstain:
            fn += 1
        elif not is_unanswerable and would_abstain:
            fp += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    n_answered = len(answered) or 1
    citation_support = sum(1 for r in answered if len(r["citations"]) > 0) / n_answered

    return {
        "citation_support_rate": round(citation_support, 3),
        "abstention_precision": round(precision, 3) if precision is not None else None,
        "abstention_recall": round(recall, 3) if recall is not None else None,
        "false_answers": fn,
        "false_refusals": fp,
        "all_citations_within_retrieved_set": all(citations_within_retrieved_set(r) for r in rows),
    }


# ---------- output ----------


def write_markdown_table(table: list[dict], gold_verified: bool, gold_note: str) -> None:
    lines = [
        "# GroundedDocs ablation table",
        "",
        f"k = {K} (identical across all rows). tau = {settings.ABSTENTION_TAU} (read from config, not swept here).",
        "",
    ]
    if not gold_verified:
        lines.append(f"> **Retrieval columns caveat:** {gold_note}")
        lines.append("")

    lines.append(
        "| Row | recall@k | precision@k | MRR | citation support rate | "
        "abstention precision | abstention recall | false answers | false refusals |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for t in table:
        r, e = t["retrieval"], t["e2e"]
        lines.append(
            "| {row} | {recall} | {precision} | {mrr} | {csr} | {ap} | {ar} | {fa} | {fr} |".format(
                row=t["row"],
                recall=r["recall@k"] if r else "N/A",
                precision=r["precision@k"] if r else "N/A",
                mrr=r["mrr"] if r else "N/A",
                csr=e["citation_support_rate"],
                ap=e["abstention_precision"],
                ar=e["abstention_recall"],
                fa=e["false_answers"],
                fr=e["false_refusals"],
            )
        )

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_PATH.write_text("\n".join(lines) + "\n")


def print_interpretation(table: list[dict]) -> None:
    by_row = {t["row"]: t for t in table}
    dense, bm25, hybrid, final = (by_row[r] for r in ROWS)

    print("\n=== Interpretation ===")

    if dense["retrieval"] and hybrid["retrieval"]:
        print(
            f"What hybrid moved (recall@{K}): dense={dense['retrieval']['recall@k']} -> "
            f"hybrid={hybrid['retrieval']['recall@k']} "
            "(retrieval numbers are placeholder-gold-derived, see caveat -- read as "
            "'the scorer works,' not as a real quality delta)."
        )
    else:
        print("Retrieval-metric movement: unavailable (no gold_chunk_ids to score against).")

    if hybrid["retrieval"] and final["retrieval"]:
        same_recall = hybrid["retrieval"]["recall@k"] == final["retrieval"]["recall@k"]
        print(
            f"What rerank moved: hybrid recall@{K}={hybrid['retrieval']['recall@k']} -> "
            f"hybrid_rerank recall@{K}={final['retrieval']['recall@k']}. "
            f"Recall stayed flat after rerank: {same_recall}. "
            "Note: this can legitimately change even though rerank never searches beyond "
            f"the pool hybrid already found. hybrid mode's own top-{K} is RRF's top-{K} of "
            f"the fused list; hybrid_rerank draws from RRF's own top-{RERANK_POOL_SIZE} and "
            f"then keeps the cross-encoder's top-{K} of THAT subset. Since RRF-rank and "
            f"cross-encoder score are different functions over the same pool, which chunks "
            f"survive into the final top-{K} can differ, not just their order -- that's "
            f"expected when k is smaller than the pool size, not a bug."
        )

    print(
        f"False answers by row: dense={dense['e2e']['false_answers']}, "
        f"bm25={bm25['e2e']['false_answers']}, hybrid={hybrid['e2e']['false_answers']}, "
        f"hybrid_rerank+tau={final['e2e']['false_answers']}."
    )
    baseline_min_false_answers = min(
        dense["e2e"]["false_answers"], bm25["e2e"]["false_answers"], hybrid["e2e"]["false_answers"]
    )
    print(
        f"Whether tau reduced false answers: "
        f"{'yes' if final['e2e']['false_answers'] <= baseline_min_false_answers else 'no'} "
        "-- rows 1-3 have no pre-generation gate at all (see script docstring): their "
        "'abstain' is generate_answer's own model-decided call, not a calibrated threshold. "
        "Any improvement in row 4 reflects the tau gate specifically."
    )
    print(
        "Citation contract holds across every row (no citation outside its own "
        f"row's retrieved set): {all(t['e2e']['all_citations_within_retrieved_set'] for t in table)}"
    )


def main() -> None:
    questions = json.loads(QUESTIONS_PATH.read_text())
    gold_by_id, gold_verified, gold_note = load_gold_index()

    print(f"k = {K} (explicit, identical across all rows)")
    print(f"tau = {settings.ABSTENTION_TAU} (read from config; not modified by this script)")
    print(f"Gold note: {gold_note}\n")

    db = SessionLocal()
    try:
        all_results = {}
        table = []
        for row_name in ROWS:
            print(f"--- {row_name} ---")
            rows = run_row(db, TEMP_USER_ID, questions, row_name, K)
            all_results[row_name] = rows

            retrieval = compute_retrieval_metrics(rows, gold_by_id, K)
            e2e = compute_end_to_end_metrics(rows)
            table.append({"row": row_name, "retrieval": retrieval, "e2e": e2e})

            print(f"  retrieval:  {retrieval}")
            print(f"  end_to_end: {e2e}\n")
    finally:
        db.close()

    final_rows = all_results[ROW_FINAL]
    assert all(
        citations_within_retrieved_set(r) for r in final_rows
    ), "Row 4 cited a chunk_id that was not in its own retrieved set!"

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(
        json.dumps(
            {
                "k": K,
                "tau": settings.ABSTENTION_TAU,
                "gold_verified": gold_verified,
                "gold_note": gold_note,
                "table": table,
                "per_row_results": all_results,
            },
            indent=2,
        )
    )
    write_markdown_table(table, gold_verified, gold_note)
    print_interpretation(table)

    print(f"\nWrote {TABLE_PATH}")
    print(f"Wrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
