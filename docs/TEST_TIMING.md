# Test run timing: `run_50k_v3` (1,732,544 test S1)

Command (resumes from checkpoints):

```bash
./venv/Scripts/python.exe run_pipeline.py --mode test --run-name run_50k_v3 --device cuda --cpu-threads 12
```

Estimates come from timings measured in this project's own logs on the RTX 4050 laptop (6 GB):
`run_100k_v2` test, the `run_10_v1` cache build and the two `run_50k_v3` test attempts.
Actual times vary by about ±20% with GPU memory pressure and background load.

## 1. Overview (what is left)

| Stage | What happens | Status | Estimated time |
| --- | --- | --- | --- |
| Step 1: load test S1 ids | 1,732,544 ids | fast | < 5 s |
| Step 2: **blocking**, France | 259k S1 vs 1.4M targets | ✅ done (11 chunks checkpointed) | 0 (resumes in seconds) |
| Step 2: **blocking**, India S2 | 810k S1 vs 2.3M targets | ✅ done (56.3M raw pairs checkpointed) | 0 |
| Step 2: **blocking**, India S3 | 810k S1 vs 2.4M targets (rival table already cached) | ⏳ remaining | 20–30 min |
| Step 2: **blocking**, US S2 + S3 | 663k S1 vs 1.9M + 1.9M targets (+ build 2 rival tables) | ⏳ remaining | 45–55 min |
| Step 2: ranking cap + saving chunks (IN, US) | ranker picks the best 80 per S1, writes 60 chunk files | ⏳ remaining | 8–12 min |
| Step 3: **scoring**, 71 chunks | features + context + stage 1 + stage 2 prediction | ⏳ remaining | 40–50 min |
| Step 4: **decision** + write files | exclusivity + thresholds (t1 0.44, t2 0.92), 2 TSVs | ⏳ remaining | 1–2 min |
| Step 5: validator | checks `matching_results.tsv` / `candidate_pairs.tsv` | ⏳ remaining | 4–5 min |
| **Total remaining** | | | **≈ 2 h – 2 h 30 min** |

## 2. Blocking in detail (per country and source)

| Country / source | S1 queries | Targets | Rival table (reverse search) | Top-K retrieval (41 / 33 query chunks) | Ranking cap (CPU ranker) | Total |
| --- | --- | --- | --- | --- | --- | --- |
| FR, S2 + S3 | 259,452 | 703k + 732k | cached | done | done | ✅ done |
| IN, S2 | 809,986 | 2,312,565 | cached | done | done | ✅ done |
| IN, S3 | 809,986 | 2,405,000 | cached (built in last run, 8 min) | 41 chunks × 20–35 s ≈ 15–24 min | 2–4 min | **20–30 min** |
| US, S2 | 663,106 | 1,871,330 | 9–11 min (first time) | 33 chunks × 15–25 s ≈ 8–14 min | 2–3 min | **20–28 min** |
| US, S3 | 663,106 | 1,945,701 | 9–11 min (first time) | 33 chunks × 15–25 s ≈ 8–14 min | 2–3 min | **20–28 min** |
| Save chunks (IN 33 + US 27 files) | | | | | | 5–8 min |

The ranker now runs on the CPU (a few seconds per chunk), so the GPU is used only for the sparse top-K search.

## 3. Scoring in detail (per 25k-S1 chunk, about 1.9M candidate pairs)

| Sub-step | Runs on | Per chunk | × 71 chunks |
| --- | --- | --- | --- |
| Load chunk + candidate list lines | disk / CPU | 2–3 s | 2–4 min |
| Pairwise features (95 columns, RapidFuzz, 12 processes) | CPU | 15–20 s | 18–24 min |
| Context features (rival + distinctive-token, from cached tables) | CPU | 5–8 s | 6–9 min |
| **Prediction**: stage 1 XGBoost | GPU | 2–3 s | 2–4 min |
| Stacking features + **prediction**: stage 2 XGBoost | CPU + GPU | 4–6 s | 5–7 min |
| Write chunk checkpoint (`scoring/*.parquet` + `.cand.tsv`) + free memory | disk | 1 s | 1–2 min |
| **Total scoring** | | **≈ 30–40 s** | **≈ 40–50 min** |

Before the first chunk, the cached rival tables are loaded once per country (about 1–2 min in total).

## 4. Final steps

| Step | Estimated time |
| --- | --- |
| Assemble `candidate_pairs.tsv` (about 1.8 GB) + kept pairs (prob ≥ 0.29) | 2–3 min |
| Decision layer (target exclusivity + t1 / t2) + write `matching_results.tsv` | 1 min |
| `utils/validate_submission.py` | 4–5 min |

## 5. What you should see in the log

| Log line | Meaning |
| --- | --- |
| `[Blocking] FR: resumed from checkpoint (11 chunks)` | France reused |
| `S1 -> S2 (IN): resumed ... pairs from checkpoint` | India S2 reused |
| `[Context] rival table S2/US: ... targets in ...s` | one-time US reverse-search build |
| `[GPU] '<field>' moved to CPU for the rest of this pass` | GPU memory tight: continues on CPU (slower, not a crash) |
| `test chunks scored: n/71` | scoring progress |
| `PASS - no blocking issues found. Safe to submit.` | upload `runs/run_50k_v3/output/matching_results.tsv` |

If the run stops for any reason (closed terminal, sleep, crash), rerun the same command. It resumes at the
last finished source or chunk.
