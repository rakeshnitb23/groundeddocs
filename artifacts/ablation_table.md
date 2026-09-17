# GroundedDocs ablation table

k = 10 (identical across all rows). tau = 0.8641 (read from config, not swept here).

> **Retrieval columns caveat:** gold_chunk_ids in eval/questions_with_gold.json are explicitly marked as random, non-human-verified placeholders (same file used in eval/results/retrieval_baseline.json). Retrieval columns below are computed only to prove the scoring code is correct -- they are NOT a trustworthy retrieval-quality signal. Do not draw conclusions from them.

| Row | recall@k | precision@k | MRR | citation support rate | abstention precision | abstention recall | false answers | false refusals |
|---|---|---|---|---|---|---|---|---|
| dense-only | 0.083 | 0.017 | 0.054 | 1.0 | 0.8 | 0.8 | 1 | 1 |
| bm25-only | 0.083 | 0.017 | 0.024 | 1.0 | 0.571 | 0.8 | 1 | 3 |
| hybrid (RRF) | 0.083 | 0.017 | 0.05 | 1.0 | 0.8 | 0.8 | 1 | 1 |
| hybrid_rerank + tau + citations | 0.042 | 0.008 | 0.042 | 1.0 | 0.625 | 1.0 | 0 | 3 |
