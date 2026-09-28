# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** [Your Team Name]  
**Team Members:** Saket Jha, [other team members]  
**Submission Date:** 27 September 2026

---

## 1. Executive Summary

We solve entity resolution as **blocking + learned pairwise matching + a global decision layer**. For every
Source 1 business, candidates are retrieved from Source 2 and Source 3 with exact normalized keys,
GPU char-3-gram TF-IDF nearest neighbours and a reverse search (target → best S1). A learned ranker keeps
the best 80 candidates, which reaches a blocking recall ceiling of 0.988. A two-stage XGBoost model scores
each pair using 120 string, structure and "context" features. Stage 2 re-scores each candidate using the
other candidates of the same S1. A decision layer then assigns every target to at most one S1 and applies
two thresholds.

The key ideas were:
- context features that look at the whole dataset, not just the pair: rival S1 businesses and target
  duplicate structure;
- a learned transliteration dictionary for native-script Indian names;
- distractor-aware training and thresholds.

Final: validation Macro F0.5 **0.9835**, leaderboard **0.96964**.

---

## 2. Methodology

### 2.1 Problem Analysis

Findings from EDA and error analysis on the training ground truth (2.2M S1, 10.3M S2+S3 records, US and
India):

- **Structure.** Each S1 has 0–5 S2 and 0–6 S3 matches (mean 3.46 links). 5.6% of S1 are singletons.
  Every S2/S3 record belongs to **at most one** S1 (0 of 7.6M owned targets have two owners).
- **Name noise:**
  - legal-suffix swaps (Pvt Ltd / Private Limited, LLC / Inc, SARL / SAS);
  - word-order transpositions;
  - typos and "leet" digits ("c1ub");
  - DBA / trade names that are random strings ("calobrixxylo") or initials ("EA" for "Editions Amis");
  - web-style names ("consortciesarl.com");
  - about 28% of India S2 names written in Devanagari, Tamil and other native scripts.
- **Address noise:**
  - abbreviations (Rd / Road, R. / Rue);
  - component reordering;
  - missing components: 3.3% of targets have **no address at all**;
  - leading zeros ("0018"), "N°", "#" prefixes;
  - house numbers perturbed even on true matches (7114 → 7112, 17255 → 7255).
- **Distractors.** 26% of training targets belong to no S1. These are look-alikes of a real business:
  same street, a nearby house number, often an extra legal suffix or a door number. They are **single
  records**: only 1.2% share (name key, house number) with another S2/S3 record, vs 32% of real records.
- **Test shift:**
  - France is unseen in training.
  - The test set has 5.8 targets per S1 vs 4.7 in training, i.e. about 40% distractors vs 26%.
  - Our test predictions showed about 2× more second-tier candidates in the 0.8–0.995 probability band
    than validation. Leaderboard probes confirmed that **stricter-than-validation thresholds score
    higher**.

### 2.2 Solution Strategy

**Approach Type:** Blocking + two-stage gradient-boosted classifier (stacking) + global assignment
decision layer (hybrid, with graph-style exclusivity).

**Core Innovation:** Context features that use the whole dataset rather than the two records alone:
1. **Rival S1 businesses.** Reverse retrieval of every target against the full S1 file: does another S1
   explain this target better?
2. **Target duplicate structure.** Is this candidate a single perturbed copy (a distractor) or one of
   several records of a real business?

Together with a learned transliteration dictionary and distractor-aware weighting and thresholds, these
target exactly the error modes found in the analysis.

---

## 3. Candidate Generation (Blocking)

Code: `src/text_norm.py`, `src/blocking.py`, `src/gpu_topk.py`. Blocking is always done within the same
country; country is treated as an open label set.

**Normalization** (cached once per file):
- anyascii transliteration and accent folding;
- a **learned transliteration dictionary** (592 native-script → Latin word mappings mined from training
  ground-truth pairs; count ≥ 3, share ≥ 0.6);
- legal-form and honorific removal to get a "core" name;
- a sorted-token key and a phonetic skeleton;
- canonical addresses (street-type abbreviations for US/IN/FR, state names → codes);
- canonical house numbers.

**Blocking keys and retrieval:**
- Exact keys: full name, core name, sorted name key, skeleton key, exact address, name-prefix + house
  number, first core words + state, house number + state, skeleton + state. Keys shared by more than 200
  targets are skipped.
- **TF-IDF nearest neighbours on the GPU** (CuPy / cuSPARSE, CPU fallback via `sparse_dot_topn`): char
  3-grams on the core name (top 15), the address (top 20) and name+address (top 15), with cosine ≥ 0.2.
  n-grams in more than 1% of targets are dropped.
- **Reverse retrieval:** each target's 5 best S1 in the whole S1 file (char + IDF-word views) are added as
  candidates. This recovers about 31% of the true links that forward retrieval misses.
- **Learned cap ranker:** an XGBoost ranker trained on uncapped candidates of 5,000 training S1 keeps the
  best **80 candidates per S1** across S2+S3.

**Candidate pairs generated:** 138,357,667 test pairs (79.9 per S1 over 1,732,544 S1). Validation: 839,197
pairs for 10,500 S1.

**How we ensured true matches were not lost:**
- The blocking recall ceiling is measured on every validation run: **0.988** of true links are in the
  candidate set.
- Retrieval is multi-view (names, addresses, combined, exact keys, reverse), so a record missing one field
  or badly misspelt is still found by another view.
- The cap is learned, not rule-ordered. Recall at the cap was measured against a heuristic cap.

---

## 4. Matching Model

Code: `src/features.py`, `src/context.py`, `src/stacking.py`, `src/model.py`. 120 stage-1 features, plus
20 stacking features in stage 2.

**Features used:**

- **Name features** (full name, core name, skeleton; all RapidFuzz, vectorized):
  - ratio, token-set, token-sort, WRatio, partial ratio, Jaro-Winkler;
  - exact equality of name / core / key / skeleton;
  - token Jaccard and overlap;
  - first- and last-word match, 4-char prefix;
  - no-space ratio (handles "etsopera.com");
  - legal-form agreement / conflict / one-missing;
  - name frequency among targets and among S1 ("is this name shared by other businesses?").
- **Address features:**
  - ratio, token-set, token-sort, partial;
  - exact match, Jaccard, length ratio;
  - number-set Jaccard, subset and fuzzy number match;
  - house number match / conflict / edit similarity / relative difference;
  - state and city match or conflict, city contained in the target address;
  - missing-address flags.
- **Other:**
  - Retrieval signals: TF-IDF cosines (name, skeleton, address, combined), candidate rank and gap to the
    best candidate per view, number of candidates, target source, native-script flag, a one-hot flag for
    every blocking rule that fired, and reverse-retrieval rank.
  - **Context: distinctive-token evidence.** IDF-weighted name and address cosine / Jaccard, and the share
    of S1 IDF mass missing from the target and vice versa (a shared city word counts little, a mismatched
    rare word counts a lot).
  - **Context: rival S1 businesses.** Reverse retrieval of every target against the whole S1 file. The
    model sees the margin between this S1 and the target's best other S1, the number of better / close
    rivals, and the rival's name, address and house evidence.
  - **Context: target duplicate counts** (new in the final run). How many other S2/S3 records share
    (name key, house number), the exact address, (name key, city) or (skeleton, house number), in total and
    in the other source only. Distractors are single records, real businesses usually have several. This
    became a top-10 stage-1 feature (`t_dup_nkh`).
  - **Stage 2 (stacking).** Out-of-fold stage-1 probability, its rank and gap within the S1 group, group
    max / second / sum / counts above 0.5 and 0.9, the top-3 sibling candidates' probability and their
    name / address similarity to this candidate, sibling support, and the max probability of siblings with
    the same address or core name.

**Model type:**
- XGBoost (GPU, `hist`): 2400 trees, depth 8, learning rate 0.03, subsample 0.8, colsample 0.6,
  `scale_pos_weight` 2.
- Stage 1 is trained on all candidates of 59,500 training S1 (4.76M pairs).
- Stage 2 is trained on the same pairs plus the stacking features built from **5-fold GroupKFold
  out-of-fold** stage-1 predictions (grouped by S1, so there is no leakage).
- **Distractor-aware weighting:** negatives whose target has no owner in the ground truth are weighted
  ×1.9, the ratio of test to training distractor density.
- We also ran Optuna (30 trials in an earlier run, 20 trials for `run_70k_opt`). The tuned parameters did
  not beat the defaults on validation or on the leaderboard (0.96937 vs 0.96964).

**Threshold selection method:**
1. **Target exclusivity:** every S2/S3 record is kept only for its highest-probability S1 (ground-truth
   property).
2. **Two thresholds:** an S1's best remaining candidate is accepted if prob ≥ t1, every other candidate if
   prob ≥ t2.
3. (t1, t2) is grid-searched on validation to maximize Macro F0.5 with false matches counted ×2
   (test-calibrated). The validation optimum was t1 0.26 / t2 0.92.
4. Because the test set has about 2× more distractors, the final thresholds were set stricter, using saved
   test scores re-thresholded without re-scoring (`--mode decide`):

   | t1 / t2 | Leaderboard |
   | --- | --- |
   | 0.26 / 0.92 | 0.96755 |
   | 0.60 / 0.97 | 0.96958 |
   | **0.75 / 0.985 (final)** | **0.96964** |

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):**

  | Metric | Value |
  | --- | --- |
  | Validation Macro F0.5 (10,500 held-out S1) | **0.9835** |
  | Precision | 0.9984 |
  | Recall | 0.9555 |
  | Blocking recall ceiling | 0.988 |
  | Macro F0.5, India / US | 0.9814 / 0.9850 |
  | Stage 1 only | 0.9816 |
  | Leaderboard (public) | **0.96964** |

- **Common false positives (wrong merges):**
  - Look-alike distractors: the same business name, often with an added legal form ("Signature Ultra" vs
    "The Signature Ultra Inc", "Silvassa Infotech Ltd" vs "Silvassa Infotech Pvt Ltd"), on the same street
    with a different or added house / door number ("1727" vs "1728 Calle Rancho Grande", "door no 59").
  - Records of a neighbouring business at the same address with a random DBA name.
  - These make up most of the high-probability errors on validation (40 of 48 negatives above 0.92) and
    are about twice as frequent on test. This motivated the duplicate features, the ×1.9 distractor weight
    and the stricter test thresholds.
- **Common false negatives (missed matches):**
  - **Records with no address** (51% of the missed links, although only 3.8% of all true links). When the
    name is unique among S1 the model finds them (99.8%). When 2–60 S1 businesses share the name, the
    record is inherently ambiguous.
  - True matches whose house number was perturbed (digit dropped, ±1–2), random DBA / initial names, and
    very partial Indian addresses.
  - Blocking misses (1.2% of links).

---

## 6. Conclusion

A carefully normalized, multi-view blocking stage (0.988 recall ceiling) combined with a two-stage XGBoost
matcher reaches 0.9835 Macro F0.5 on validation. The most valuable features were those that look beyond
the pair: rival S1 businesses, distinctive-token IDF evidence and target duplicate structure. The main
lesson: the test set contains many more look-alike distractors than training. Precision-oriented choices
(distractor weighting, duplicate features, exclusivity, stricter thresholds) consistently improved the
leaderboard, while larger training samples and hyperparameter tuning did not.

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/`:

| Path | Purpose |
| --- | --- |
| `run_pipeline.py` | Entry point (`--mode train / val / test / decide`) |
| `config.py` | All settings (paths, blocking, model, thresholds) |
| `src/text_norm.py` | Normalization, transliteration dictionary |
| `src/blocking.py`, `src/gpu_topk.py` | Candidate generation, GPU top-K, learned cap ranker |
| `src/features.py` | Pairwise features |
| `src/context.py` | Rival-S1, IDF-evidence and target-duplicate features |
| `src/stacking.py` | Stage-2 features |
| `src/model.py`, `src/tuner.py` | XGBoost / LightGBM wrapper, Optuna |
| `src/metrics.py` | Macro F0.5, exclusivity, (t1, t2) sweep |
| `src/pipeline.py` | Resumable train / val / test / decide orchestration |
| `utils/validate_submission.py` | Official validator |
| `README.md` | Setup and full documentation |
| `requirements.txt` | Pinned dependencies (Python 3.11, CUDA 12 GPU) |

Reproduce both output files (data in `dataset/` as described in the README):

```bash
python run_pipeline.py --mode train  --sample-size 70000 --train-split 0.85 --run-name run_70k_v1 --model-type xgboost --device cuda --cpu-threads 12
python run_pipeline.py --mode val    --run-name run_70k_v1 --device cuda --cpu-threads 12
python run_pipeline.py --mode test   --run-name run_70k_v1 --device cuda --cpu-threads 12
python run_pipeline.py --mode decide --t1 0.75 --t2 0.985 --run-name run_70k_v1 --cpu-threads 12
```

The commands produce these files:
- `runs/run_70k_v1/output/decide_t1_0.75_t2_0.98/matching_results.tsv`
- `runs/run_70k_v1/output/candidate_pairs.tsv`

Hardware used: RTX 4050 laptop (6 GB), 16 GB RAM, 20 cores. Training takes about 45 min, validation
about 3 min, and the test about 1 h 45 min from scratch. Every stage is checkpointed and resumable. All
models are XGBoost (Apache 2.0); no external data or APIs are used.

### B. Additional Results

| Run | Change | Validation F0.5 | Leaderboard |
| --- | --- | --- | --- |
| run_100k_v1 | Blocking v2 + XGBoost + stacking | 0.9693 | 0.948 |
| run_100k_v2 | + context features (rival S1, IDF evidence) | 0.9745 | 0.953 → 0.9597 (stricter thresholds) |
| run_50k_v3 | + transliteration dictionary, reverse retrieval, legal-form features, distractor weighting | 0.9832 | 0.9687 |
| **run_70k_v1** | + target duplicate features, 70k sample | **0.9835** | **0.96964** |
| run_70k_opt | Optuna-tuned XGBoost | 0.9831 | 0.96937 |

Final submission: 1,732,544 S1 rows, 5,669,854 predicted links, 100,099 S1 with no match (5.8%).

Stage-1 top features:
1. `rev_rank`
2. `rv_margin`
3. `rv_n_close`
4. `num_s1_subset`
5. `rv_n_better`
6. `rule_rev_knn`
7. `rule_exact_name`
8. `num_jaccard`
9. `legal_conflict`
10. `t_dup_nkh`
