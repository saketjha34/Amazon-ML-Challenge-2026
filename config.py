"""
================================================================================
Configuration Settings - Business Entity Resolution (ML Challenge 2026)
================================================================================
All pipeline configurations can be customized in this file.
You can adjust:
  - Dataset paths
  - Train/Validation split ratio (default: 70% train / 30% val by S1 entity)
  - Sample sizes (e.g. 5000 for fast experimentation, None for full dataset)
  - Blocking parameters
  - Feature engineering options (TF-IDF, RapidFuzz, Jaccard)
  - LightGBM model hyperparameters
  - Threshold tuning range for Macro F0.5 optimization
  - Run output directory (automatic run_1, run_2, run_3 versioning)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any, List


@dataclass
class PipelineConfig:
    # --------------------------------------------------------------------------
    # 1. PATHS
    # --------------------------------------------------------------------------
    BASE_DIR: Path = Path(__file__).resolve().parent
    DATA_DIR: Path = BASE_DIR / "dataset"
    RUNS_DIR: Path = BASE_DIR / "runs"

    # Parquet source file paths
    TRAIN_S1_PATH: Path = DATA_DIR / "train_source1_final.parquet"
    TRAIN_S2_PATH: Path = DATA_DIR / "train_source2_final.parquet"
    TRAIN_S3_PATH: Path = DATA_DIR / "train_source3_final.parquet"
    TRAIN_GT_PATH: Path = DATA_DIR / "train_ground_truth.tsv"

    TEST_S1_PATH: Path = DATA_DIR / "test_source1_final.parquet"
    TEST_S2_PATH: Path = DATA_DIR / "test_source2_final.parquet"
    TEST_S3_PATH: Path = DATA_DIR / "test_source3_final.parquet"

    # --------------------------------------------------------------------------
    # 2. TRAIN / VALIDATION SPLIT (70-30 Split by S1 Entity)
    # --------------------------------------------------------------------------
    TRAIN_SPLIT_RATIO: float = 0.70  # 70% of Source 1 entities for training
    VAL_SPLIT_RATIO: float = 0.30    # 30% of Source 1 entities for validation
    RANDOM_SEED: int = 42

    # --------------------------------------------------------------------------
    # 3. SAMPLE SIZE CONTROL
    # --------------------------------------------------------------------------
    # Set to an integer (e.g., 5000, 10000, 50000) for rapid testing/experimentation.
    # Set to None to run on the ENTIRE dataset.
    SAMPLE_SIZE_S1: Optional[int] = None
    # None = the first SAMPLE_SIZE_S1 S1 by entity_id (how run_100k_v1/v2 were sampled).
    # An integer = a random sample drawn with this seed from the whole S1 file (--sample-seed).
    SAMPLE_SEED: Optional[int] = None
    # Runs whose train + validation S1 are excluded from this run's sample (--exclude-runs), which makes
    # samples disjoint so the runs can be ensembled without leaking each other's validation data.
    SAMPLE_EXCLUDE_RUNS: List[str] = field(default_factory=list)
    # Other runs whose models are averaged with this run's models in val / test (--ensemble-with).
    ENSEMBLE_RUNS: List[str] = field(default_factory=list)

    # Candidates kept per S1 (across S2+S3), ranked by blocking similarity score
    MAX_CANDIDATES_PER_S1: int = 80
    MAX_MATCHES_PER_S1: int = 10

    # --------------------------------------------------------------------------
    # 4. BLOCKING PARAMETERS
    # --------------------------------------------------------------------------
    # Hard partition by country (US with US, India with India, France with France)
    COUNTRY_PARTITION: bool = True
    MIN_NAME_TOKEN_LEN: int = 3
    NAME_PREFIX_LEN: int = 4
    # Blocking v2: exact-key rules + char 3-gram TF-IDF top-K retrieval (see src/blocking.py)
    BLOCKING_PARAMS: Dict[str, Any] = field(default_factory=lambda: {
        # skel = 0: phonetic-skeleton top-K retrieval disabled (found only 0.09% of true links on its
        # own but cost ~55% of GPU retrieval time); skel_cos is still computed as a feature.
        "topk": {"name": 15, "skel": 0, "addr": 20, "combo": 15},
        "cos_floor": 0.2,        # minimum cosine for a retrieved neighbour
        "key_freq_cap": 200,     # skip exact keys shared by more targets than this
        # drop n-grams present in more than this share of targets (per field)
        "max_df_frac": {"name": 0.01, "skel": 0.05, "addr": 0.01, "combo": 0.01},
        "n_features": 2 ** 20,
        "query_chunk": 20_000,
        # Reverse retrieval: each target's best reverse_k S1 businesses (context rival table) are added
        # as candidates. Recovers ~31% of the true links plain S1->target retrieval never finds.
        "reverse_k": 5,
        "n_jobs": None,          # None = use --cpu-threads (src/resources.py)
        "use_gpu": True,         # top-K retrieval on the GPU (CuPy/cuSPARSE); --cpu-blocking to disable
    })
    # S1 entities per test inference chunk (block -> features -> predict), bounds memory
    TEST_CHUNK_S1: int = 25_000
    # Test pairs with prob below min(t1, t2) - margin are dropped after scoring (they can never be
    # accepted); the margin lets thresholds be lowered later without re-scoring.
    TEST_KEEP_MARGIN: float = 0.15

    # --------------------------------------------------------------------------
    # 5. FEATURE ENGINEERING (TF-IDF, RapidFuzz, Jaccard)
    # --------------------------------------------------------------------------
    TFIDF_NGRAM_RANGE: tuple = (3, 4)
    TFIDF_ANALYZER: str = "char_wb"
    TFIDF_MAX_FEATURES: int = 50_000
    BATCH_SIZE: int = 100_000

    # --------------------------------------------------------------------------
    # 6. MODEL SELECTION & GPU HYPERPARAMETERS
    # --------------------------------------------------------------------------
    MODEL_TYPE: str = "xgboost"  # "xgboost" (recommended, fast GPU) or "lightgbm"
    DEVICE: str = "cuda"         # "cuda" for NVIDIA GPU, "cpu" for CPU

    XGBOOST_PARAMS: Dict[str, Any] = field(default_factory=lambda: {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "device": "cuda",
        # Tuned on run_50k_v2 features (3.4M pairs): stage-1 F0.5 0.9644 -> 0.9656 vs 800 x d8 / lr .05
        "n_estimators": 2400,
        "learning_rate": 0.03,
        "max_depth": 8,
        "min_child_weight": 2,
        "subsample": 0.8,
        "colsample_bytree": 0.6,
        "scale_pos_weight": 2.0,
        "random_state": 42,
    })

    LIGHTGBM_PARAMS: Dict[str, Any] = field(default_factory=lambda: {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "n_estimators": 800,
        "learning_rate": 0.05,
        "num_leaves": 127,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "min_child_samples": 20,
        "scale_pos_weight": 2.0,
        "random_state": 42,
        "n_jobs": None,        # None = use --cpu-threads (src/resources.py)
        "verbose": -1,
    })

    # --------------------------------------------------------------------------
    # 7. TRAINING & VALIDATION EFFICIENCY (Speed Optimizations)
    # --------------------------------------------------------------------------
    # Retains 100% of true positive matches and caps negative candidate pairs per S1
    # entity to avoid wasting time on millions of trivial negatives.
    # Set to an integer (e.g. 6 or 8) or None to train on all pairs.
    # Hardest negatives (highest blocking score) kept per S1, plus random extra negatives
    # None = train on every candidate (matches the inference distribution; required for stacking)
    MAX_TRAIN_NEGATIVES_PER_S1: Optional[int] = None
    RANDOM_TRAIN_NEGATIVES_PER_S1: int = 4

    # Sample size of S1 validation entities for rapid threshold tuning (e.g. 50,000).
    # Set to None to evaluate all validation entities.
    VAL_SAMPLE_SIZE_S1: Optional[int] = 50_000

    # --------------------------------------------------------------------------
    # 8. THRESHOLD TUNING (F_0.5 Optimization)
    # --------------------------------------------------------------------------
    DEFAULT_THRESHOLD: float = 0.50
    THRESHOLD_SWEEP_MIN: float = 0.10
    THRESHOLD_SWEEP_MAX: float = 0.96
    THRESHOLD_SWEEP_STEP: float = 0.02
    # Each S2/S3 record is assigned to at most one S1 (its highest-probability S1)
    EXCLUSIVE_TARGETS: bool = True
    # None = choose thresholds that maximize Macro F0.5 (the leaderboard metric).
    # A value such as 0.975 = choose the thresholds with the highest recall whose precision >= it.
    MIN_PRECISION: Optional[float] = None
    # Test-calibrated thresholds: the (t1, t2) sweep maximizes Macro F0.5 with every false match counted
    # this many times (leaderboard probes on run_100k_v2: stricter-than-validation thresholds scored
    # 0.957 / 0.9597 vs 0.953). 1.0 = plain validation optimum (also reported in metrics.json).
    THRESHOLD_FP_WEIGHT: float = 2.0
    # Training weight of negatives whose target belongs to no S1 business (distractors): test has ~40%
    # such targets vs 26% in training -> odds ratio ~1.9. 1.0 = off.
    ORPHAN_NEG_WEIGHT: float = 1.9

    # --------------------------------------------------------------------------
    # 10. STAGE-2 STACKING (re-score candidates using sibling candidates of the same S1)
    # --------------------------------------------------------------------------
    USE_STACKING: bool = True
    STACK_OOF_FOLDS: int = 5

    # --------------------------------------------------------------------------
    # 11. LEARNED BLOCKING RANKER (decides which candidates survive MAX_CANDIDATES_PER_S1)
    # --------------------------------------------------------------------------
    USE_BLOCK_RANKER: bool = True

    # --------------------------------------------------------------------------
    # 12. CONTEXT FEATURES (src/context.py): distinctive-token evidence + rival S1 businesses
    #     from reverse retrieval over the whole S1 file. Targets look-alike businesses (France).
    # --------------------------------------------------------------------------
    USE_CONTEXT: bool = True

    # Optuna (--tune-hyperparams): trials train on this share of train S1, stop after TUNE_TIMEOUT_MIN
    TUNE_TRAIN_FRAC: float = 0.4
    TUNE_TIMEOUT_MIN: int = 75
    RANKER_TRAIN_S1: int = 5_000     # train S1 entities whose raw (uncapped) candidates train the ranker

    # --------------------------------------------------------------------------
    # 9. VALIDATION SCRIPT
    # --------------------------------------------------------------------------
    VALIDATOR_PATH: Path = BASE_DIR / "utils" / "validate_submission.py"
    TEST_DIR: Path = DATA_DIR / "test"


# Global default instance
DEFAULT_CONFIG = PipelineConfig()

