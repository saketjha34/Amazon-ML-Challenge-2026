# Business Entity Resolution — Amazon ML Challenge 2026

For every **Source 1** business record, find all records in **Source 2** and **Source 3** that describe
the same real-world business. The pipeline trains, validates and produces the two submission files
(`matching_results.tsv`, `candidate_pairs.tsv`). The score is **Macro F0.5 per Source 1 entity**
(precision counts twice as much as recall; a correctly empty "singleton" scores 1.0).

The challenge statement is in [docs/PROBLEM_STATEMENT.md](docs/PROBLEM_STATEMENT.md).

| Run | Validation F0.5 | Precision | Recall | Blocking recall ceiling | Leaderboard |
| --- | --- | --- | --- | --- | --- |
| `run_50k_v2` | 0.9684 | 0.9928 | 0.9317 | 0.9718 | — |
| `run_100k_v1` (XGBoost + stacking) | 0.9693 | 0.9945 | 0.9312 | 0.9709 | **0.948** |
| `run_100k_v2` (+ context features) | 0.9745 | 0.9970 | 0.9409 | 0.971 | 0.953 (0.9597 with stricter thresholds) |
| `run_50k_v3` (v3 approach, below) | 0.9832 | 0.9982 | 0.9546 | 0.9880 | 0.9687 |
| **`run_70k_v1`** (v3 + target-duplicate features, **final**) | **0.9835** | 0.9984 | 0.9555 | 0.9882 | 0.96755 (val thresholds) → 0.96958 (0.60/0.97) → **0.96964 (t1 0.75 / t2 0.985, submitted)** |
| `run_70k_opt` (Optuna-tuned XGBoost) | 0.9831 | 0.9986 | 0.9551 | 0.9882 | 0.96937 (0.75/0.985) |

### Reproduce the final submission

```bash
# 1) train on the first 70,000 Source 1 businesses (85% train / 15% validation), ~45 min
python run_pipeline.py --mode train --sample-size 70000 --train-split 0.85 --run-name run_70k_v1 --model-type xgboost --device cuda --cpu-threads 12
# 2) validate: thresholds + metrics/metrics.json, ~3 min
python run_pipeline.py --mode val --run-name run_70k_v1 --device cuda --cpu-threads 12
# 3) test: blocking + scoring of all 1.73M test S1 -> output/candidate_pairs.tsv + test scores, ~45 min
#    (~1 h 45 min when the test blocking is not reused from an earlier run)
python run_pipeline.py --mode test --run-name run_70k_v1 --device cuda --cpu-threads 12
# 4) final decision thresholds (stricter than validation; chosen on the leaderboard), ~5 min
python run_pipeline.py --mode decide --t1 0.75 --t2 0.985 --run-name run_70k_v1 --cpu-threads 12
```
Final files: `runs/run_70k_v1/output/decide_t1_0.75_t2_0.98/matching_results.tsv` (the folder name rounds
0.985) and `runs/run_70k_v1/output/candidate_pairs.tsv`.

**v4 addition (in `run_70k_v1`): target duplicate features** (`src/context.py`, `DUP_COLUMNS`). Distractor
records (S2/S3 records that belong to no S1 business) are single perturbed copies: only 1.2% of them share
(name key, house number) with another S2/S3 record and 4.6% share the exact address, vs 32% / 43% of real
records. Six counts per candidate (other records sharing name key + house number, exact address, name key +
city, name skeleton + house number, and the first two counted in the other source only) let the model
separate a look-alike distractor from a real record. The test set has ~2x more distractors per S1 than
training, which is also why stricter-than-validation thresholds score higher on the leaderboard.

### v3 approach (what changed and why)

Measured on `run_100k_v2`, a perfect model on its candidates scores only 0.989 on validation, and the
leaderboard reacts to thresholds as if the test set has many more distractors than validation. v3 fixes
the measured holes:

| Change | Where | Evidence |
| --- | --- | --- |
| **Learned transliteration dictionary** for native-script names ("फ्यूचर इंडस्ट्रीज" → "future industries"), learned from training ground truth (normalization v3) | `src/text_norm.py` | native-script true pairs with name token-set ≥ 90: 5.7% → 95.7% |
| **Reverse retrieval**: each target's best 5 S1 in the whole S1 file become candidates (`rev_knn`, `rev_rank`) | `src/blocking.py` | recovers 31.5% of the true links plain retrieval never finds |
| **Legal-form agreement + Jaro-Winkler features** (LLC vs Inc transfers to SARL vs SAS) | `src/features.py` | France look-alikes differ mainly by legal form |
| **Distractor-aware training**: negatives on targets owned by no S1 weighted ×1.9 (`ORPHAN_NEG_WEIGHT`) | `src/pipeline.py` | test ≈ 40% unowned targets vs 26% in training |
| **Test-calibrated thresholds**: sweep counts each false match ×2 (`THRESHOLD_FP_WEIGHT`); the plain validation optimum is also stored | `src/metrics.py` | stricter thresholds: 0.953 → 0.957 → 0.9597 on the leaderboard |

Turn pieces off with `--orphan-weight 1.0` (retrain needed), `--fp-weight 1.0` (re-run `val` only), or
`BLOCKING_PARAMS["reverse_k"] = 0`.

**Contents**
1. [Setup from scratch](#1-setup-from-scratch)
2. [Running the pipeline](#2-running-the-pipeline)
3. [How it works: the full pipeline in detail](#3-how-it-works-the-full-pipeline-in-detail)
4. [Context features (rival businesses) in detail](#4-context-features-rival-businesses-in-detail)
5. [Configuration reference (`config.py`)](#5-configuration-reference-configpy)
6. [Phase reference: train / validation / test](#6-phase-reference-train--validation--test)
7. [Checkpoints, resuming, and the run folder](#7-checkpoints-resuming-and-the-run-folder)
8. [Command-line flags](#8-command-line-flags)
9. [Resources, timings, and troubleshooting](#9-resources-timings-and-troubleshooting)
10. [Working as a team (git)](#10-working-as-a-team-git)
11. [Repository layout](#11-repository-layout)

---

## 1. Setup from scratch

### 1.1 What you need

| | Minimum | Used for development |
| --- | --- | --- |
| OS | Windows 10/11 or Linux | Windows 11 |
| Python | **3.11** (3.12+ is untested) | 3.11.1 |
| RAM | 16 GB | 16 GB |
| GPU | Optional: NVIDIA with a CUDA 12.x driver | RTX 4050 Laptop, 6 GB |
| CPU | 4+ cores | 20 threads |
| Free disk | **~60 GB** | raw data ~3 GB, parquet ~3 GB, caches ~22 GB, each full run 15–30 GB |

Without a GPU everything still runs on the CPU (add `--device cpu`), just more slowly.

### 1.2 Install the tools

1. **Python 3.11**: https://www.python.org/downloads/ (Windows: tick *"Add python.exe to PATH"*).
2. **Git**: https://git-scm.com/downloads
3. **NVIDIA driver** (GPU only): any recent driver supporting CUDA 12. Check with `nvidia-smi`.
   You do **not** need to install the CUDA Toolkit separately; `cupy-cuda12x[ctk]` bundles it.
4. **Jupyter** (to run the preprocessing notebook): installed in step 1.4 below, or use Kaggle.

### 1.3 Get the code and create the environment

```bash
git clone <this-repo-url>
cd Amazon-ML-Challenge-2026

python -m venv venv
# Windows (cmd / PowerShell)
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
pip install notebook          # only needed to run data-preprocessing.ipynb locally
```

`requirements.txt` (pinned):

| Package | Version | Used for |
| --- | --- | --- |
| numpy / pandas / pyarrow / scipy | 2.4.6 / 3.0.6 / 25.0.1 / 1.17.1 | data handling, sparse matrices |
| scikit-learn | 1.9.1 | hashing vectorizers, GroupKFold |
| duckdb | 1.5.5 | fast parquet queries / joins |
| RapidFuzz | 3.14.6 | string similarity features |
| anyascii | 0.3.3 | transliteration of any script to ASCII |
| sparse-dot-topn | 1.2.0 | CPU top-K sparse cosine retrieval |
| xgboost / lightgbm | 3.2.0 / 4.7.0 | classifiers (XGBoost on GPU) |
| optuna | 5.0.0 | hyper-parameter tuning |
| tqdm / joblib | 4.70.1 / 1.6.0 | progress bars, parallel workers |
| cupy-cuda12x[ctk] | 14.2.0 | GPU sparse retrieval (optional) |

**No NVIDIA GPU?** Delete the `cupy-cuda12x[ctk]` line before `pip install`, and pass `--device cpu`
to every command.

Check the install:

```bash
python run_pipeline.py --help
python -c "import cupy; print(cupy.cuda.runtime.getDeviceCount(), 'GPU(s)')"   # GPU only
```

### 1.4 Get and prepare the data

The data is **not in git** (`dataset/` is in `.gitignore`: it is tens of GB). Everyone prepares it locally.

**Step A: download** the challenge data from the portal. It contains:
```
student_resource/dataset/train/train_source1.tsv, train_source2.tsv, train_source3.tsv, train_ground_truth.tsv
student_resource/dataset/test/test_source1.tsv,  test_source2.tsv,  test_source3.tsv
```

**Step B: run `data-preprocessing.ipynb`.** It reads the raw TSVs, adds normalized columns and writes six
parquet files plus `metadata.json`. It was written on Kaggle, so **edit two paths** in its first cells
before running it locally:

```python
BASE_DIR = Path(r"C:\path\to\student_resource")        # folder containing dataset/train and dataset/test
SAVE_DIR = Path(r"C:\path\to\Amazon-ML-Challenge-2026\dataset")   # where the parquet files go
```

Then run all cells (`jupyter notebook data-preprocessing.ipynb`, *Run → Run All*). It needs about 8–12 GB RAM.
Each parquet file has these columns: `entity_id, business_name, business_address, country,
name_normalized, name_tokens, name_length, name_token_count, address_normalized, address_tokens,
address_length, address_token_count, address_numbers, house_number, country_normalized, state_code, city`.
`state_code` and `city` are only parsed for US and India. France gets empty values, and the pipeline
does not depend on them.

**Step C: arrange the files exactly like this:**

```
dataset/
├── train_source1_final.parquet     # ~2.2M rows
├── train_source2_final.parquet     # ~5.0M rows
├── train_source3_final.parquet     # ~5.3M rows
├── test_source1_final.parquet      # ~1.73M rows (US, India, France)
├── test_source2_final.parquet      # ~4.9M rows
├── test_source3_final.parquet      # ~5.1M rows
├── train_ground_truth.tsv          # copy from the raw train/ folder
├── metadata.json                   # written by the notebook
└── test/
    ├── test_source1.tsv            # copy the raw test TSVs here: utils/validate_submission.py reads them
    ├── test_source2.tsv
    └── test_source3.tsv
```

Paths live in `config.py` (section 1). You can also point to another folder with `--data-dir <folder>`
(same file names).

### 1.5 Smoke test (about 30–60 min the first time, a few minutes afterwards)

```bash
python run_pipeline.py --mode all --sample-size 500 --run-name smoke --device cuda --cpu-threads 8
```

This trains on 500 Source 1 entities, validates, and writes a 500-entity test submission to
`runs/smoke/output/`. **The first run of any kind builds the one-time caches in `dataset/norm/`**
(normalized text, TF-IDF matrices, target indexes, context tables, about 22 GB), which is why it is slow.
If this finishes with `PASS` from the validator, your setup is complete.

---

## 2. Running the pipeline

Run the phases one at a time with the **same `--run-name`**. Every phase checkpoints its work, so after
a crash or Ctrl+C you just rerun the same command to resume.

```bash
# 1) TRAIN: 100k Source 1 sample, 85% train / 15% validation
python run_pipeline.py --mode train --sample-size 100000 --train-split 0.85 --run-name run_100k_v2 --model-type xgboost --device cuda --cpu-threads 8

# 2) VALIDATE: scores the held-out 15%, picks decision thresholds, writes metrics/metrics.json
python run_pipeline.py --mode val --run-name run_100k_v2 --device cuda --cpu-threads 8

# 3) TEST: all 1.73M test Source 1 entities -> submission files + validator
python run_pipeline.py --mode test --run-name run_100k_v2 --device cuda --cpu-threads 8
```

The submission is written to `runs/<run>/output/matching_results.tsv` (upload this file) and
`runs/<run>/output/candidate_pairs.tsv`.

**Reusing an earlier run (`--init-from`).** When only the model or features change, add
`--init-from <old_run>` to the **train** command. It hard-links the old run's split, blocking output,
features and test blocking checkpoints into the new run folder (no extra disk space), which skips
about 2 hours of blocking. Keep the same `--sample-size` / `--train-split` as the old run so the split is
identical.

```bash
python run_pipeline.py --mode train --sample-size 100000 --train-split 0.85 --run-name run_100k_v3 --init-from run_100k_v2 --device cuda --cpu-threads 8
```

**Hyper-parameter tuning** (needs a finished validation run in the same folder, because it uses the
cached validation features):

```bash
python run_pipeline.py --mode train --sample-size 100000 --train-split 0.85 --run-name run_100k_v2 --tune-hyperparams --n-trials 30 --tune-timeout 75 --device cuda --cpu-threads 8
python run_pipeline.py --mode val --run-name run_100k_v2 --device cuda --cpu-threads 8   # re-pick thresholds afterwards
```

**Validate a submission by hand** (the test phase already does this):

```bash
python utils/validate_submission.py --matching runs/run_100k_v2/output/matching_results.tsv --candidate runs/run_100k_v2/output/candidate_pairs.tsv --test-dir dataset/test
```

Upload only if it prints **PASS**.

---

## 3. How it works: the full pipeline in detail

```
 raw TSV ─► data-preprocessing.ipynb ─► *_final.parquet
                                              │
                         ┌────────────────────┘
                         ▼
 (1) Normalization  src/text_norm.py      → dataset/norm/*_norm_v2.parquet (cached)
 (2) Blocking       src/blocking.py       exact keys + TF-IDF top-K (GPU)  → ≤80 candidates / S1
                    src/gpu_topk.py       learned BlockRanker decides which 80 survive
 (3) Pair features  src/features.py       87 features per (S1, candidate)
 (4) Context        src/context.py        21 features using the WHOLE Source 1 file (rivals, rare words)
 (5) Stage 1        src/model.py          XGBoost (GPU) → p1
 (6) Stage 2        src/stacking.py       XGBoost on features + p1 of sibling candidates → prob
 (7) Decision       src/metrics.py        target exclusivity + two thresholds (t1, t2)
 (8) Output         src/pipeline.py       matching_results.tsv, candidate_pairs.tsv, validator
```

Terminology used below:
- **S1** is a Source 1 record, the reference. It is deduplicated: every real business appears once.
- **Target** is a Source 2 or Source 3 record.
- **Pair** is (S1, candidate target).
- A true link has label 1. On average an S1 has about 3.46 true targets, and about 5.6% of S1 have none
  (these are singletons).
- Each target belongs to **at most one** S1 (verified on the full training ground truth).

### 3.1 Data preprocessing (`data-preprocessing.ipynb`)

The notebook loads the raw TSVs with string dtypes, then adds normalized name and address, token
lists, lengths, numbers found in the address, house number, and, for US and India, `state_code` and
`city`. The results are saved as zstd-compressed parquet. The pipeline reads only these parquet files.

### 3.2 Normalization (`src/text_norm.py`)

This step runs once per source file and is cached as `dataset/norm/<file>_norm_v2.parquet`. It is
parallel over `--cpu-threads` processes. For every record it produces:

| Column | How it is built | Example |
| --- | --- | --- |
| `name_t` | NFKD accent folding → `anyascii` transliteration (Devanagari, Tamil, … → ASCII) → lowercase → `&`→`and` → punctuation removed → leetspeak repaired (`c1ub`→`club`, `m0dern`→`modern`) → ID / phone numbers removed → multi-token legal forms joined (`l l c`→`llc`) | `Roues & Fi1s SARL` → `roues and fils sarl` |
| `name_core` | `name_t` without legal forms and honorifics (US: llc, inc, corp, ltd…; India: pvt, private, limited plus transliterated variants such as *praivet* and *limitet*, m/s, shri; France: sarl, sas, sasu, eurl, sci, cie…; web noise: www, com) | `roues and fils` |
| `name_key` | sorted unique tokens of `name_core` (word order does not matter) | `and fils roues` |
| `name_skel` | phonetic consonant skeleton per token (ph→f, bh→b, voiced→unvoiced, vowels after the first letter dropped, doubles collapsed). Bridges transliteration noise | `marketing` ≈ `marketimg` |
| `addr_t` | transliterated address, abbreviations unified (road→rd, street→st, avenue/av→ave, r→rue, all→allee…), state names → codes (tennessee→tn, maharashtra→mh), leading zeros removed (`018`→`18`) | `018 Place De L'église` → `18 pl de l eglise` |
| `house_c` | house number without leading zeros | `0072` → `72` |
| `is_native` | 1 if the raw name contains non-Latin script | |

Country is an **open set**: nothing is filtered or hard-coded per country. France is processed with the same code.

### 3.3 Blocking / candidate generation (`src/blocking.py`, `src/gpu_topk.py`)

It is impossible to score every S1 against ~10M targets, so blocking proposes plausible candidates. It is
the **recall ceiling**: a true match that is not proposed can never be predicted. The current ceiling is
about 0.971. Blocking runs **per country** (S1 of country X only against targets of country X) and per
target source (S2, S3).

**(a) Exact-key rules** are hash joins on normalized keys. A key shared by more than `key_freq_cap` = 200
targets is skipped, because it is too generic and would flood the candidates.

| Rule | Key | Catches |
| --- | --- | --- |
| `exact_name` | `name_t` | identical names |
| `core_name` | `name_core` | same name, different legal form |
| `name_key` | `name_key` | word-order changes |
| `skel_key` | `name_skel` | transliteration / spelling noise |
| `exact_addr` | `addr_t` (≥ 8 chars) | same address, name changed (DBA / trade name) |
| `prefix4_house` | first 4 letters of name + house number | typos later in the name |
| `cw12_state` | first 2 words of name + state | partial names |
| `num_state` | every address number (house / plot / PIN) + state | short or partial target addresses |
| `skel1_state` | first skeleton token + state | heavy transliteration |

**(b) TF-IDF nearest-neighbour retrieval.** Char 3-gram hashed TF-IDF (2^20 features, `char_wb`,
sublinear tf, IDF from the target file; n-grams present in more than `max_df_frac` of targets are dropped).
For each S1, the top-K most similar targets are retrieved per field:

| Field | Text | top-K |
| --- | --- | --- |
| `name` | `name_core` | 15 |
| `addr` | `addr_t` | 20 |
| `combo` | `name_core + addr_t` | 15 |
| `skel` | `name_skel` | 0 (disabled for retrieval, still used as a cosine feature) |

Neighbours need cosine ≥ `cos_floor` = 0.2. Retrieval runs on the **GPU** (CuPy / cuSPARSE sparse
matrix product, `src/gpu_topk.py`). Batches are sized from free GPU memory; if a batch fails with out
of memory it is split in half and retried, and it falls back to the CPU (`sparse_dot_topn`) if needed.
A background thread computes the next chunk's GPU retrieval while the CPU does the key joins, so GPU and
CPU work overlap.

**(c) Merge and cap.** All pairs from all rules are merged, keeping a **multi-hot rule bitmask** (one bit
per rule) and the exact cosines of all four fields. Each S1 then keeps its best
`MAX_CANDIDATES_PER_S1` = 80 candidates across S2 + S3, ranked by the **BlockRanker**:
- A small GPU XGBoost model (300 trees, depth 6) over 17 cheap signals: 4 cosines, a missing-address
  flag, a heuristic score and the 13 rule bits.
- It is trained during `train` (step 3a) on the **uncapped** candidates of 5,000 training S1
  (`RANKER_TRAIN_S1`).
- It raised recall@80 from 0.9468 (hand-made heuristic) to 0.9699.
- It is saved as `models/block_ranker.joblib`.

**(d) Materialization.** Each kept pair gets both records' normalized fields plus signals computed from
the whole files:
- `target_core_freq`: how many targets share the target's core name.
- `target_core_s1freq` / `s1_core_s1freq`: how many S1 businesses use that name.
- `other_s1_same_name_house` / `other_s1_same_name_addr`: whether **another** S1 has the target's exact
  name plus house number or address.

In test mode, blocking is written to disk per country / source / 25k-S1 chunk
(see [section 7](#7-checkpoints-resuming-and-the-run-folder)).

### 3.4 Pairwise features (`src/features.py`, 87 columns)

These are computed with RapidFuzz `cpdist` (multi-threaded) and vectorized NumPy. Batches run in parallel
processes, with task size capped at 25k rows (large pickles break Windows pipes).

| Group | Features |
| --- | --- |
| Name similarity | `name_ratio/tokenset/tokensort/wratio`, `core_ratio/tokenset/partial`, `skel_ratio/tokenset`, `nospace_ratio/partial` (spaces removed: `obccollege` vs `obc college`) |
| Exact flags | `name_exact`, `core_exact`, `key_exact`, `skel_exact`, `addr_exact` |
| Name tokens | `core_jaccard`, `core_overlap_ratio`, `core_len_ratio`, `core_token_diff`, `first_word_match`, `last_word_match`, `core_prefix_4` |
| Address | `addr_ratio/tokenset/tokensort/partial`, `addr_jaccard`, `addr_overlap_ratio`, `addr_len_ratio` |
| Numbers | `num_jaccard`, `num_any_overlap`, `num_s1_subset`, `has_nums_both`, `num_fuzzy` |
| House number | `house_match`, `has_house_both`, `house_conflict`, `house_edit_sim`, `house_num_reldiff`, `house_prefix` (house numbers carry digit noise: 2169 vs 2171) |
| Location | `state_match`, `has_state_both`, `state_conflict`, `city_match`, `has_city_both`, `city_in_target_addr` |
| Retrieval | `name_cos`, `skel_cos`, `addr_cos`, `combo_cos`, `block_score` |
| Within-S1 context | `rank_block/addr/name`, `gap_block/addr/name` (distance to the S1's best candidate), `n_cands`, `n_cands_same_source`, `n_sib_same_addr`, `n_sib_same_core` |
| Missing data | `s1_addr_empty`, `target_addr_empty`, `name_only_evidence` |
| Name rarity | `target_core_logfreq`, `rare_name_evidence`, `s1freq_target_name_log`, `s1freq_own_name_log`, `name_share_prior` (a name used by exactly one S1 business almost always belongs to it) |
| Competing S1 | `other_s1_same_name_house`, `other_s1_same_name_addr` |
| Other | `is_source3`, `target_native`, `rule_<rule>` bits |

There is **no country feature on purpose**: France is not in training, so the model must not depend on
the country.

### 3.5 Context features (`src/context.py`, 21 columns)

These are pair features that need the **whole Source 1 file**. See [section 4](#4-context-features-rival-businesses-in-detail).

### 3.6 Model: stage 1 and stage 2 (`src/model.py`, `src/stacking.py`)

- **Stage 1**: XGBoost on the GPU over all pairwise and context features. Params are in
  [section 5](#5-configuration-reference-configpy). `scale_pos_weight` = 2 because about 4% of pairs are
  positive. It predicts `p1`.
- **Out-of-fold predictions**: 5-fold `GroupKFold` grouped by S1, so no S1 is in both a fold's train and
  predict parts. Each training pair gets a `p1` from a model that never saw its S1, and stage 2 therefore
  never trains on leaked scores. Each fold is checkpointed.
- **Stage 2**: a second XGBoost on the stage-1 features plus 20 **stacking features** computed within each
  S1's candidate group. The final probability is `prob`.
  - `p1`, `p1_rank`, `p1_max`, `p1_gap`, `p1_second`, `p1_sum`, `p1_n_gt50`, `p1_n_gt90`
  - `sib{1,2,3}_p1/core_sim/addr_sim`: the 3 most confident other candidates and how similar they are
    to this one.
  - `sib_support`, `sib_same_addr_max_p1`, `sib_same_core_max_p1`.

  The idea: the true matches of one S1 are variants of the same business, so they resemble each other.
  A candidate similar to other confident candidates is likely a match; an isolated one that competes
  with a stronger, different sibling is likely not.
- `--model-type lightgbm` uses LightGBM on the CPU. `ensemble` trains XGBoost (GPU) and LightGBM (CPU)
  concurrently and blends them 65% / 35%.

### 3.7 Decision layer (`src/metrics.py`)

1. **Exact override**: a pair with identical `name_t` and identical `addr_t` gets `prob` = 1.0.
2. **Target exclusivity** (`EXCLUSIVE_TARGETS`): each target is kept only for the S1 with its
   highest `prob`. This follows from the one-owner-per-target structure of the data.
3. **Two thresholds**: an S1's best remaining candidate is accepted if `prob ≥ t1`, and any other
   candidate if `prob ≥ t2`, up to `MAX_MATCHES_PER_S1` = 10.
   - A separate `t1` lets the top candidate be accepted at a lower confidence, which gains recall
     without adding extra links. In `run_100k_v1`, t1 = 0.36 and t2 = 0.86.
4. The thresholds are chosen in `val` by a full grid sweep (0.10 to 0.96, step 0.02, both axes) that
   maximizes Macro F0.5. They are written to `metrics/metrics.json`, and `test` reads them from there.
   `--min-precision P` instead picks the highest-recall thresholds with precision ≥ P.

### 3.8 Test inference (`run_test`)

1. Load the 1.73M test S1 ids.
2. Resumable blocking into `checkpoints/test/blocking/<country>/chunks/chunk_XXXXX.parquet`
   (25k S1 per chunk).
3. Build the context rival tables for the test files (cached), then for each chunk: pair features +
   context features → stage 1 → stage 2. Each chunk writes:
   - its `candidate_pairs` lines (`.cand.tsv`);
   - its pairs with `prob ≥ min(t1, t2) − 0.15` (`.parquet`). Everything lower can never be accepted.

   Finished chunks are skipped on resume.
4. Assemble `candidate_pairs.tsv`, where every S1 gets a row, even an empty one. Apply the decision layer
   and write `matching_results.tsv`, which is always a subset of the candidates.
5. Run `utils/validate_submission.py` and print per-country match rates.

---

## 4. Context features (rival businesses) in detail

### 4.1 Why they exist

Validation scored 0.969 but the leaderboard 0.948. The analysis of the test predictions showed the main
cause is **France** (about 15% of test, never seen in training):
- French business names come from a small vocabulary and often contain the city ("Lille Club SARL",
  "Pornic Club SARL", "Saint-Herblain Institut SARL", "Deleves Maison / Deleves Sportive / Deleves Amis…").
- Source 1 businesses sharing an address are 2–3× more common in France.
- Source 1 businesses sharing an address **and** the first name word are 7–50× more common.
- 1.4% of French targets were claimed by two or more S1 businesses with prob ≥ 0.86, versus 0.06% in the
  US and 0.2% in India.

The pairwise model only sees one pair at a time. It gave 0.99 to
*"Defense Sante SAS, 99 **Rue Auber**"* for the target *"defensesante.com, 99 **Rue du Chêne Houpline**"*,
because name and house number agree. It never saw that another S1, *"Defense Sante SARL, 99 Rue du Chêne
Houpline"*, matches the target better.

### 4.2 Family 1: distinctive-token evidence (12 columns)

Name words (`name_core`) and address words (`addr_t`, with French `N°` and `bis`/`ter` cleaned) are
turned into binary word vectors. Each word is weighted by its **IDF over the Source 1 file of that
country**: `idf = ln((1+N)/(1+df)) + 1`. Common words ("club", "rue", a city name) weigh little and rare
words (a street name, a distinctive business word) weigh a lot. This is frequency-based agreement
weighting, as in the classic Fellegi–Sunter record linkage.

| Column | Meaning |
| --- | --- |
| `nm_idf_cos` / `ad_idf_cos` | IDF-weighted cosine of name / address words |
| `nm_idf_jacc` / `ad_idf_jacc` | IDF-weighted Jaccard |
| `nm_s1_miss` / `ad_s1_miss` | share of the S1's IDF mass missing from the target |
| `nm_t_extra` / `ad_t_extra` | share of the target's IDF mass missing from the S1 |
| `nm_miss_abs`, `nm_extra_abs`, `ad_miss_abs`, `ad_extra_abs` | the same as absolute IDF mass |

These values are NaN when either side is empty; XGBoost handles missing values natively.

### 4.3 Family 2: rival Source 1 businesses (9 columns)

**Rival table.** Built once per (target file, country) and cached in `dataset/norm/ctx/`:
1. Every target is queried **against the entire Source 1 file** of its country (reverse retrieval) with
   two views, taking the top `RIVAL_K` = 10 from each with cosine ≥ 0.25:
   - char 3-gram TF-IDF on name + address (robust to typos);
   - IDF-weighted words on name + address. Words in more than 1% of S1 records are left out of this
     retrieval index only, for speed. This view is robust to rare-fragment domination, e.g. the
     department *"Loire-Atlantique"* pulling in every *"Avenue de l'Atlantique"*.
2. The union is re-scored with the evidence score
   `score = 0.5·nm_idf_cos + 0.4·ad_idf_cos + 0.1·house_equal`.
3. The best `RIVAL_KEEP` = 5 S1 per target are stored (row, score, name / address evidence, house flag).

**Pair features.** For a pair (s1, target), `own` = the same evidence score for this S1. The "best rival"
is the highest-scoring S1 in the target's rival list **other than s1 itself**.

| Column | Meaning |
| --- | --- |
| `ctx_own_score` | evidence score of this pair |
| `rv_has_rival` | the target has at least one other plausible S1 |
| `rv_best_score` | evidence score of the best rival S1 |
| `rv_margin` | `own − best rival`: positive means this S1 is the best owner |
| `rv_best_nm`, `rv_best_ad`, `rv_best_house` | the rival's name / address / house evidence |
| `rv_n_better` | how many rivals score higher than this S1 |
| `rv_n_close` | how many rivals score within 0.05 of this S1 |

Because Source 1 is deduplicated, a rival that matches the target better means the target most likely
belongs to that rival. The features are computed the same way in train, validation and test, always
against the whole Source 1 file of that dataset.

Example (test, France). Target *"PORNIC PRIMAIRE SAS, Rue André Louërat, Pornic"*:

| S1 | stage-2 prob (v1, no context) | `rv_margin` |
| --- | --- | --- |
| Pornic Primaire SAS, 45 Rue André Louërat (correct) | 1.000 | **+0.14** |
| Pornic Agricole SARL, 89 Rue André Louërat | 0.969 | −0.23 |
| Pornic Union SASU, 69 Rue André Louërat | 0.803 | −0.17 |

**Cost:** about 80 s per 700k targets on the GPU, so about 20 min once for train and 20 min once for test.
Pair features take about 3–8 s per million pairs. Turn them off with `--no-context` or
`USE_CONTEXT = False`.

---

## 5. Configuration reference (`config.py`)

All settings live in `PipelineConfig` in `config.py`. Command-line flags override some of them per run.
The effective config of every run is saved to `runs/<run>/config.json`.

### 5.1 Paths
| Key | Default | Meaning |
| --- | --- | --- |
| `DATA_DIR` | `dataset/` | data folder (`--data-dir`) |
| `RUNS_DIR` | `runs/` | output root (`--runs-dir`) |
| `TRAIN_S1_PATH` … `TEST_S3_PATH` | `dataset/<split>_source<k>_final.parquet` | preprocessed sources |
| `TRAIN_GT_PATH` | `dataset/train_ground_truth.tsv` | ground truth |
| `TEST_DIR` | `dataset/test` | raw test TSVs for the validator |
| `VALIDATOR_PATH` | `utils/validate_submission.py` | submission checker |

### 5.2 Sampling and split
| Key | Default | Flag | Meaning |
| --- | --- | --- | --- |
| `SAMPLE_SIZE_S1` | `None` (all) | `--sample-size` | S1 entities used for train + validation (the first N by `entity_id`, a random-like subset). In test mode it limits test S1, for quick checks only. |
| `TRAIN_SPLIT_RATIO` | 0.70 | `--train-split` | share of the sample for training; the rest is validation. We use **0.85**. |
| `VAL_SPLIT_RATIO` | 0.30 | (set automatically) | 1 − train split |
| `RANDOM_SEED` | 42 | | seed for the split, the ranker sample and the models |
| `VAL_SAMPLE_SIZE_S1` | 50,000 | `--val-sample-size` | maximum validation S1 scored in `val` |

### 5.3 Blocking
| Key | Default | Flag | Meaning |
| --- | --- | --- | --- |
| `MAX_CANDIDATES_PER_S1` | 80 | `--max-candidates` | candidates kept per S1 (S2 + S3 combined). More = higher recall ceiling, slower. |
| `BLOCKING_PARAMS.topk` | name 15, skel 0, addr 20, combo 15 | | TF-IDF neighbours retrieved per field (0 = field not retrieved) |
| `BLOCKING_PARAMS.cos_floor` | 0.2 | | minimum retrieval cosine |
| `BLOCKING_PARAMS.key_freq_cap` | 200 | | skip exact keys shared by more targets than this |
| `BLOCKING_PARAMS.max_df_frac` | name/addr/combo 0.01, skel 0.05 | | drop n-grams present in more than this share of targets |
| `BLOCKING_PARAMS.n_features` | 2^20 | | hashing space of the TF-IDF |
| `BLOCKING_PARAMS.query_chunk` | 20,000 | | S1 queries per retrieval batch |
| `BLOCKING_PARAMS.n_jobs` | `None` | `--cpu-threads` | CPU workers (None = the thread cap) |
| `BLOCKING_PARAMS.use_gpu` | `True` | `--cpu-blocking` / `--device cpu` | GPU retrieval |
| `USE_BLOCK_RANKER` | `True` | | learned cap ranker instead of the heuristic score |
| `RANKER_TRAIN_S1` | 5,000 | | train S1 used to fit the ranker |
| `TEST_CHUNK_S1` | 25,000 | | S1 per checkpointed chunk (test, and train/val blocking) |
| `TEST_KEEP_MARGIN` | 0.15 | | test pairs with `prob < min(t1,t2) − margin` are not kept after scoring |

Changing `topk`, `cos_floor`, `key_freq_cap`, `max_df_frac`, `n_features` or `MAX_CANDIDATES_PER_S1`
changes the blocking output. Checkpoints detect this and re-block. Changing `max_df_frac` or
`n_features` also rebuilds the TF-IDF caches.

### 5.4 Features, context and stacking
| Key | Default | Flag | Meaning |
| --- | --- | --- | --- |
| `BATCH_SIZE` | 100,000 | | rows per feature / prediction batch |
| `USE_CONTEXT` | `True` | `--no-context` | context features ([section 4](#4-context-features-rival-businesses-in-detail)) |
| `USE_STACKING` | `True` | `--no-stacking` | stage-2 model |
| `STACK_OOF_FOLDS` | 5 | `--oof-folds` | folds for out-of-fold stage-1 predictions (3 is about 40% faster) |
| `MAX_TRAIN_NEGATIVES_PER_S1` | `None` | `--max-train-negatives` | negative subsampling (forced off when stacking is on, because stage 2 needs complete groups) |
| `RANDOM_TRAIN_NEGATIVES_PER_S1` | 4 | | extra random negatives when subsampling |

Constants inside `src/context.py`: `RIVAL_K` = 10, `RIVAL_KEEP` = 5, `RIVAL_FLOOR` = 0.25,
`WORD_RETRIEVAL_MAX_DF` = 0.01, `RIVAL_QUERY_CHUNK` = 50,000.

### 5.5 Model
| Key | Default | Flag |
| --- | --- | --- |
| `MODEL_TYPE` | `xgboost` | `--model-type xgboost\|lightgbm\|ensemble` |
| `DEVICE` | `cuda` | `--device cuda\|cpu` |

`XGBOOST_PARAMS` (tuned on 3.4M pairs of `run_50k_v2`):

| Param | Value | | Param | Value |
| --- | --- | --- | --- | --- |
| objective | binary:logistic | | max_depth | 8 |
| tree_method | hist | | min_child_weight | 2 |
| n_estimators | 2400 | | subsample | 0.8 |
| learning_rate | 0.03 | | colsample_bytree | 0.6 |
| scale_pos_weight | 2.0 | | random_state | 42 |

`LIGHTGBM_PARAMS`: 800 trees, learning rate 0.05, 127 leaves, subsample 0.8, colsample 0.8,
min_child_samples 20, scale_pos_weight 2.0, threads = `--cpu-threads`.

### 5.6 Decision / thresholds
| Key | Default | Flag | Meaning |
| --- | --- | --- | --- |
| `EXCLUSIVE_TARGETS` | `True` | | each target goes only to its best S1 |
| `MAX_MATCHES_PER_S1` | 10 | | cap on links per S1 |
| `THRESHOLD_SWEEP_MIN/MAX/STEP` | 0.10 / 0.96 / 0.02 | | grid for the (t1, t2) sweep |
| `MIN_PRECISION` | `None` | `--min-precision` | None = maximize F0.5; a number = max recall at precision ≥ it |
| `DEFAULT_THRESHOLD` | 0.50 | `--threshold` | used only if no `metrics.json` exists; `--threshold` forces t1 = t2 in test |

### 5.7 Optuna tuning
| Key | Default | Flag |
| --- | --- | --- |
| `TUNE_TRAIN_FRAC` | 0.4 | `--tune-train-frac` (share of train S1 per trial) |
| `TUNE_TIMEOUT_MIN` | 75 | `--tune-timeout` |
| trials | 20 | `--n-trials` |

Search space (TPE sampler; the current defaults are tried first; early stopping of 150 rounds on
validation logloss; the objective is validation Macro F0.5):
- `learning_rate` 0.02–0.12 (log)
- `max_depth` 6–12
- `min_child_weight` 1–20 (log)
- `subsample` 0.6–1
- `colsample_bytree` 0.4–1
- `colsample_bynode` 0.5–1
- `gamma` 1e-3–5 (log)
- `reg_alpha` 1e-3–10 (log)
- `reg_lambda` 0.1–20 (log)
- `scale_pos_weight` 1–4
- `max_bin` 256 or 512

The best trial's tree count is scaled up for the full training set. Results are saved to
`runs/<run>/optuna/`.

### 5.8 Legacy keys (unused)
`COUNTRY_PARTITION`, `MIN_NAME_TOKEN_LEN`, `NAME_PREFIX_LEN`, `TFIDF_NGRAM_RANGE`, `TFIDF_ANALYZER`,
`TFIDF_MAX_FEATURES` are left over from the first version and are not read by the current code.

---

## 6. Phase reference: train / validation / test

Each phase shows a step progress bar and saves per-step timings to `runs/<run>/metrics/timings_<phase>.json`.

### TRAIN (`--mode train`)
| Step | What happens | Checkpoint |
| --- | --- | --- |
| 1 | Load ground truth (S1 → set of targets) | |
| 2 | Split the S1 sample into train / validation (seeded shuffle) | `train_s1_ids.npy`, `val_s1_ids.npy` |
| 3a | Train the BlockRanker on the uncapped candidates of 5,000 train S1; prints recall@80 heuristic vs learned | `models/block_ranker.joblib` |
| 3 | Block all train S1 (resumable per country / source / chunk) | `checkpoints/candidates_train.parquet` |
| 4 | Label pairs from ground truth; print the train recall ceiling | |
| 5 | Pairwise features (87) | `features/train_features.parquet` |
| 5b | Context features (21); builds the train rival tables the first time | `features/train_context.parquet` |
| 6 | 5 OOF folds → stage-2 model → stage-1 model; importances | `checkpoints/train/oof/fold*.npy`, `models/model_stage2.joblib`, `models/model.joblib`, `models/train_state.json` |

### VALIDATION (`--mode val`)
| Step | What happens | Checkpoint |
| --- | --- | --- |
| 1 | Load `val_s1_ids.npy` (capped at `VAL_SAMPLE_SIZE_S1`) | |
| 2 | Block the validation S1 (resumable) | `checkpoints/val_pairs.parquet` |
| 3 | Pairwise features | `features/val_features.parquet` |
| 3b | Context features | `features/val_context.parquet` |
| 4 | Stage 1 + stage 2 → sweep (t1, t2) → metrics and diagnostics | `metrics/metrics.json`, `threshold_sweep*.csv` |

`metrics.json` fields:
- **Headline:** `best_macro_f05`, `precision`, `recall`, `singleton_accuracy`.
- **Thresholds:** `threshold_top1` (t1) and `threshold_rest` (t2). Test mode reads these.
- **Blocking:** `candidate_recall` (the blocking ceiling) and `avg_candidates_per_s1`.
- **Ablations:** `macro_f05_single_threshold`, `macro_f05_no_exclusivity`, `macro_f05_stage1_only`.
- **Diagnostics:** `macro_f05_US` / `macro_f05_IN` (per country), `macro_f05_lookalike` (S1 whose name is
  shared with another S1 business) and `lookalike_s1_share`.

### TEST (`--mode test`)
| Step | What happens | Checkpoint |
| --- | --- | --- |
| 1 | Load all test S1 ids | |
| 2 | Resumable blocking of 1.73M S1 | `checkpoints/test/blocking/<country>/chunks/*` |
| 3 | Build the test rival tables (cached), then per chunk: features + context + stage 1 + stage 2 | `checkpoints/test/scoring/<country>__chunk_*.parquet / .cand.tsv` |
| 4 | Decision layer; write `matching_results.tsv` + `candidate_pairs.tsv`; per-country summary | `checkpoints/test_scored.parquet`, `output/` |
| 5 | Run `utils/validate_submission.py` | |

The test phase needs `models/` and `metrics/metrics.json` from the train and val phases of the same run.

---

## 7. Checkpoints, resuming, and the run folder

```
runs/<run>/
├── config.json                      # effective settings of the last command
├── train_s1_ids.npy, val_s1_ids.npy
├── models/
│   ├── block_ranker.joblib          # blocking cap ranker
│   ├── model.joblib                 # stage 1 (lightgbm_model.joblib = same file, legacy name)
│   ├── model_stage2.joblib          # stage 2
│   ├── feature_extractor.joblib
│   └── train_state.json             # training signature + which models are finished
├── features/                        # train/val pairwise + context features (parquet)
├── checkpoints/
│   ├── candidates_train.parquet, val_pairs.parquet
│   ├── train/blocking, val/blocking # in-progress blocking chunks (removed once merged)
│   ├── train/oof/                   # finished OOF folds (removed once training completes)
│   ├── test/blocking/<country>/chunks/chunk_XXXXX.parquet (+ _DONE)
│   ├── test/scoring/<country>__chunk_XXXXX.parquet / .cand.tsv
│   └── test_scored.parquet
├── metrics/                         # metrics.json, sweeps, feature importances, timings_*.json
├── optuna/                          # tuning trials + best params (if tuned)
└── output/                          # matching_results.tsv, candidate_pairs.tsv
```

- **Resume:** rerun the same command. Blocking resumes at the last finished country / source / chunk,
  training at the last finished fold or model, test scoring at the last finished chunk.
- **Fingerprints:** blocking checkpoints store the S1 ids, blocking settings, normalization version and
  ranker file. Scoring checkpoints store the model files and thresholds. Context caches store their
  version. Training stores a hash of features, rows and params. If anything changes, the stale part is
  rebuilt automatically.
- `--force-recompute` discards the checkpoints of the phase you run.
- **Shared caches** in `dataset/norm/` are reused by every run:
  - `*_norm_v2.parquet`: normalized text
  - `tidx/`: prepared target frames and key tables
  - `tfidf/`: TF-IDF matrices
  - `ctx/`: word matrices and rival tables

  Deleting them is safe; they are rebuilt, slowly.

---

## 8. Command-line flags

| Flag | Default | Meaning |
| --- | --- | --- |
| `--mode train\|val\|test\|all` | all | phase(s) to run |
| `--run-name NAME` | auto `run_N` | folder under `runs/` |
| `--sample-size N` | all | S1 sample for train / val (see 5.2) |
| `--train-split R` | 0.70 | train share of the sample (we use 0.85) |
| `--val-sample-size N` | 50,000 | max validation S1 scored |
| `--init-from RUN` | | reuse the split, blocking, features and test blocking of another run |
| `--model-type` | xgboost | `xgboost`, `lightgbm` or `ensemble` |
| `--device cuda\|cpu` | cuda | GPU or CPU for models and retrieval |
| `--cpu-threads N` | 4 | cap on every CPU thread pool / process pool (set before libraries load) |
| `--cpu-blocking` | off | retrieval on CPU even with `--device cuda` |
| `--max-candidates N` | 80 | candidates per S1 |
| `--oof-folds N` | 5 | stacking folds |
| `--no-stacking` / `--no-context` | off | disable stage 2 / context features |
| `--max-train-negatives N` | all | negative subsampling (only without stacking) |
| `--tune-hyperparams` | off | Optuna tuning (with `--n-trials`, `--tune-timeout`, `--tune-train-frac`) |
| `--min-precision P` | | recall-first threshold choice |
| `--threshold T` | | force a single threshold in test |
| `--force-recompute` | off | ignore and delete checkpoints of the phase |
| `--data-dir DIR` / `--runs-dir DIR` | | alternative data / output folders |

---

## 9. Resources, timings, and troubleshooting

**CPU / GPU / RAM.**
- `--cpu-threads` caps NumPy/BLAS, DuckDB, RapidFuzz, joblib workers, LightGBM and XGBoost together.
  Use 8–12 on a 20-thread laptop, and fewer if you want to keep using the machine.
- The heavy maths (retrieval, XGBoost training and prediction) runs on the GPU.
- Memory is freed between stages; GPU memory pools are released after each chunk.
- **Run one pipeline at a time.** Two in parallel exhausted 16 GB RAM and page file in testing.

**Approximate timings** (RTX 4050 laptop, `--cpu-threads 8–12`):

| Phase | First time | Re-run from checkpoints |
| --- | --- | --- |
| Caches in `dataset/norm/` (first ever run) | 30–60 min | reused |
| Train, 100k sample | ~1.5 h (blocking ~30 min, context ~20 min, OOF + models ~20 min) | minutes |
| Validation (15k S1) | ~5 min | ~1 min |
| Test (1.73M S1) | ~2–2.5 h (blocking ~85 min, context ~20 min, scoring ~30 min) | minutes |

| Symptom | Fix |
| --- | --- |
| `ModuleNotFoundError: cupy` / CUDA DLL errors | `--device cpu`, or reinstall `cupy-cuda12x[ctk]==14.2.0` for a CUDA 12 driver |
| GPU out of memory | Handled automatically (batch split → CPU fallback). Close other GPU programs. |
| `WinError 1450` / `PicklingError` in features | Lower `--cpu-threads` (e.g. 4); the pipeline also falls back to one process automatically |
| Machine freezes / RAM full | One phase at a time, close browsers, lower `--cpu-threads` |
| `Optuna needs cached validation features` | Run `--mode val` once in that run folder first |
| `Model not found … Run training first` | `val` / `test` need a finished `train` in the same `--run-name` |
| Crash or Ctrl+C | Rerun the same command; it resumes |
| Blocking suddenly slow again | A normalization / blocking setting changed, so caches are being rebuilt (one-time) |
| Odd characters in the console (`�`) | Only the Windows console code page; the files are UTF-8 and correct |

---

## 10. Working as a team (git)

- **Never commit data or runs.** `.gitignore` excludes `dataset/`, `runs/`, `output/`, `optuna/`, `venv/`
  and any `*.parquet / *.npz / *.npy / *.joblib / *.tsv / *.csv`. Before committing, `git status` must
  show only code and docs.
- **Sharing a trained run with a teammate:** zip `runs/<run>/models/` and `runs/<run>/metrics/metrics.json`
  and send it outside git (Drive etc.). They put it in `runs/<same-run-name>/` and can run `--mode test`
  directly.
- Use your own `--run-name` per experiment (e.g. `run_100k_v3_alice`) and record the validation F0.5 and
  leaderboard score in the results table at the top of this file.
- The rules forbid external data or APIs (geocoding, business registries…). Everything here uses only the
  provided files.

---

## 11. Repository layout

```
├── run_pipeline.py            # CLI entry point (sets the CPU thread cap before heavy imports)
├── config.py                  # every setting (section 5)
├── requirements.txt           # pinned dependencies (Python 3.11)
├── data-preprocessing.ipynb   # raw TSV -> *_final.parquet
├── src/
│   ├── pipeline.py            # train / val / test orchestration, checkpoints
│   ├── text_norm.py           # normalization + cached normalized parquet
│   ├── blocking.py            # key rules, TF-IDF retrieval, BlockRanker, resumable blocking
│   ├── gpu_topk.py            # GPU sparse top-K (CuPy / cuSPARSE) with OOM handling
│   ├── features.py            # 87 pairwise features
│   ├── context.py             # distinctive-token + rival-S1 context features
│   ├── stacking.py            # stage-2 sibling features
│   ├── model.py               # XGBoost / LightGBM / ensemble wrapper
│   ├── metrics.py             # Macro F0.5, decision layer, threshold sweeps
│   ├── tuner.py               # Optuna tuning
│   ├── data_loader.py         # ground truth, S1 split
│   ├── run_manager.py         # run folders, --init-from
│   ├── progress.py            # progress bars + step timings
│   └── resources.py           # CPU thread cap
├── utils/validate_submission.py   # official submission checker
├── docs/PROBLEM_STATEMENT.md
└── Documentation_template.md  # methodology write-up for the final submission
```
