#!/usr/bin/env python3
"""
================================================================================
Unified Pipeline Runner - Business Entity Resolution (ML Challenge 2026)
================================================================================
Usage:
  # 1. Run Complete End-to-End Pipeline (Train -> Val -> Test) on a fast sample (5,000 S1 entities):
  python run_pipeline.py --mode all --sample-size 5000

  # 2. Run Full End-to-End Pipeline on the complete dataset:
  python run_pipeline.py --mode all

  # 3. Run individual phases:
  python run_pipeline.py --mode train --sample-size 10000
  python run_pipeline.py --mode val   --run-name run_1
  python run_pipeline.py --mode test  --run-name run_1

  # 4. Custom output folder name:
  python run_pipeline.py --mode all --run-name run_1
"""

import argparse
import sys
from pathlib import Path


def _preparse_cpu_threads(default: int = 4) -> int:
    # Read --cpu-threads before NumPy & co. are imported so their thread pools obey the cap
    for i, a in enumerate(sys.argv):
        if a == "--cpu-threads" and i + 1 < len(sys.argv):
            return int(sys.argv[i + 1])
        if a.startswith("--cpu-threads="):
            return int(a.split("=", 1)[1])
    return default


sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.resources import apply_cpu_limit  # noqa: E402  (stdlib-only module)
apply_cpu_limit(_preparse_cpu_threads())

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import DEFAULT_CONFIG, PipelineConfig
from src.run_manager import RunManager
from src.pipeline import EntityResolutionPipeline


def main():
    parser = argparse.ArgumentParser(
        description="Business Entity Resolution ML Pipeline (ML Challenge 2026)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "val", "test", "all", "decide"],
        default="all",
        help="Pipeline mode to run: 'train', 'val', 'test', or 'all' (default: all)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Optional S1 entity sample size limit for quick testing (e.g. 5000). None runs on full dataset.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Name of run folder inside runs/ (e.g. run_1). If omitted, automatically increments to run_1, run_2, etc.",
    )
    parser.add_argument(
        "--train-split",
        type=float,
        default=None,
        help="Ratio of S1 entities allocated to training (default: 0.70)",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=None,
        help="Ratio of S1 entities allocated to validation (default: 0.30)",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["xgboost", "lightgbm", "ensemble"],
        default="xgboost",
        help="Classifier model architecture: 'xgboost' (GPU), 'lightgbm' (CPU), or 'ensemble' (blend)",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu"],
        default="cuda",
        help="Compute device for XGBoost/features ('cuda' for NVIDIA GPU, 'cpu' for CPU)",
    )
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="Force recomputation of candidates and feature matrices, ignoring saved checkpoints",
    )
    parser.add_argument(
        "--tune-hyperparams",
        action="store_true",
        help="Run Optuna Bayesian hyperparameter optimization to maximize Macro F_0.5",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=20,
        help="Number of Optuna tuning trials (default: 20)",
    )
    parser.add_argument(
        "--max-train-negatives",
        type=int,
        default=None,
        help="Max negative candidate pairs per S1 in training (e.g. 6). None uses all candidates.",
    )
    parser.add_argument(
        "--val-sample-size",
        type=int,
        default=None,
        help="S1 entity sample size for validation threshold sweep (default: 50000). None uses all.",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=None,
        help="Candidates kept per S1 after blocking (default: 80). Higher = more recall, slower.",
    )
    parser.add_argument(
        "--oof-folds",
        type=int,
        default=None,
        help="Out-of-fold folds for stage-2 stacking (default: 5). 3 trains ~40%% faster.",
    )
    parser.add_argument(
        "--tune-timeout",
        type=int,
        default=None,
        help="Optuna time budget in minutes (default: 75)",
    )
    parser.add_argument(
        "--tune-train-frac",
        type=float,
        default=None,
        help="Share of train S1 used per Optuna trial (default: 0.4)",
    )
    parser.add_argument(
        "--min-precision",
        type=float,
        default=None,
        help="Recall-first thresholds: max recall with precision >= this (e.g. 0.975). "
             "Default: maximize Macro F0.5 (leaderboard metric).",
    )
    parser.add_argument(
        "--cpu-blocking",
        action="store_true",
        help="Run blocking top-K retrieval on the CPU instead of the GPU.",
    )
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=4,
        help="Max CPU threads/processes for every CPU stage (default: 4). GPU does the heavy math.",
    )
    parser.add_argument(
        "--no-stacking",
        action="store_true",
        help="Disable the stage-2 stacking model (stage-1 scores only)",
    )
    parser.add_argument(
        "--init-from",
        type=str,
        default=None,
        help="Reuse blocking / features / test-blocking checkpoints of another run folder (e.g. run_100k_v1)",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=None,
        help="Draw --sample-size S1 at random with this seed (default: the first N by entity_id)",
    )
    parser.add_argument(
        "--exclude-runs",
        type=str,
        default=None,
        help="Comma-separated runs whose train+val S1 are excluded from this sample (disjoint samples)",
    )
    parser.add_argument(
        "--ensemble-with",
        type=str,
        default=None,
        help="Comma-separated runs whose models are averaged with this run's models (val and test)",
    )
    parser.add_argument(
        "--no-context",
        action="store_true",
        help="Disable rival-S1 / distinctive-token context features",
    )
    parser.add_argument(
        "--fp-weight",
        type=float,
        default=None,
        help="Threshold choice counts each false match this many times (default 2.0; 1.0 = validation optimum)",
    )
    parser.add_argument(
        "--orphan-weight",
        type=float,
        default=None,
        help="Training weight of negatives on targets owned by no S1 (default 1.9; 1.0 = off)",
    )
    parser.add_argument(
        "--t1",
        type=float,
        default=None,
        help="--mode decide: threshold for each S1's top candidate",
    )
    parser.add_argument(
        "--t2",
        type=float,
        default=None,
        help="--mode decide: threshold for the other candidates",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Manual threshold override for test matching",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Custom dataset directory path (default: dataset/)",
    )
    parser.add_argument(
        "--runs-dir",
        type=str,
        default=None,
        help="Custom runs root directory path (default: runs/)",
    )

    args = parser.parse_args()

    # Load configuration
    cfg = DEFAULT_CONFIG

    # Apply model & device overrides
    if args.model_type:
        cfg.MODEL_TYPE = args.model_type
    if args.device:
        cfg.DEVICE = args.device
        # Blocking top-K retrieval runs on the GPU unless --cpu-blocking or --device cpu
        cfg.BLOCKING_PARAMS["use_gpu"] = (args.device == "cuda") and not args.cpu_blocking
    if args.max_train_negatives is not None:
        cfg.MAX_TRAIN_NEGATIVES_PER_S1 = args.max_train_negatives
    if args.val_sample_size is not None:
        cfg.VAL_SAMPLE_SIZE_S1 = args.val_sample_size
    if args.max_candidates is not None:
        cfg.MAX_CANDIDATES_PER_S1 = args.max_candidates
    if args.no_stacking:
        cfg.USE_STACKING = False
    if args.no_context:
        cfg.USE_CONTEXT = False
    if args.fp_weight is not None:
        cfg.THRESHOLD_FP_WEIGHT = args.fp_weight
    if args.orphan_weight is not None:
        cfg.ORPHAN_NEG_WEIGHT = args.orphan_weight
    if args.sample_seed is not None:
        cfg.SAMPLE_SEED = args.sample_seed
    if args.exclude_runs:
        cfg.SAMPLE_EXCLUDE_RUNS = [r.strip() for r in args.exclude_runs.split(",") if r.strip()]
    if args.ensemble_with:
        cfg.ENSEMBLE_RUNS = [r.strip() for r in args.ensemble_with.split(",") if r.strip()]
    if args.oof_folds is not None:
        cfg.STACK_OOF_FOLDS = args.oof_folds
    if args.min_precision is not None:
        cfg.MIN_PRECISION = args.min_precision
    if args.tune_timeout is not None:
        cfg.TUNE_TIMEOUT_MIN = args.tune_timeout
    if args.tune_train_frac is not None:
        cfg.TUNE_TRAIN_FRAC = args.tune_train_frac

    # Apply directory & dataset overrides
    if args.data_dir:
        cfg.DATA_DIR = Path(args.data_dir)
        cfg.TRAIN_S1_PATH = cfg.DATA_DIR / "train_source1_final.parquet"
        cfg.TRAIN_S2_PATH = cfg.DATA_DIR / "train_source2_final.parquet"
        cfg.TRAIN_S3_PATH = cfg.DATA_DIR / "train_source3_final.parquet"
        cfg.TRAIN_GT_PATH = cfg.DATA_DIR / "train_ground_truth.tsv"
        cfg.TEST_S1_PATH = cfg.DATA_DIR / "test_source1_final.parquet"
        cfg.TEST_S2_PATH = cfg.DATA_DIR / "test_source2_final.parquet"
        cfg.TEST_S3_PATH = cfg.DATA_DIR / "test_source3_final.parquet"
        cfg.TEST_DIR = cfg.DATA_DIR / "test"

    if args.runs_dir:
        cfg.RUNS_DIR = Path(args.runs_dir)

    if args.sample_size is not None:
        cfg.SAMPLE_SIZE_S1 = args.sample_size

    if args.train_split is not None:
        cfg.TRAIN_SPLIT_RATIO = args.train_split
        cfg.VAL_SPLIT_RATIO = 1.0 - args.train_split

    if args.threshold is not None:
        cfg.DEFAULT_THRESHOLD = args.threshold

    # Initialize Run Manager (handles run_1, run_2, run_3 automatic directory creation)
    rm = RunManager(runs_dir=cfg.RUNS_DIR, custom_run_name=args.run_name)
    rm.save_config(cfg.__dict__)
    if args.init_from:
        rm.seed_from(args.init_from)

    print(f"Target Run Directory: {rm.run_dir}")
    print(f"Model Architecture  : {cfg.MODEL_TYPE.upper()} (Device: {cfg.DEVICE.upper()})")
    print(f"Sample Size Limit   : {cfg.SAMPLE_SIZE_S1}")
    print(f"Train/Val Split     : {cfg.TRAIN_SPLIT_RATIO:.0%} / {cfg.VAL_SPLIT_RATIO:.0%}")
    print(f"Force Recompute     : {args.force_recompute}")
    print(f"Optuna Hyper-tuning : {args.tune_hyperparams} (Trials: {args.n_trials})")

    # Build Pipeline
    pipeline = EntityResolutionPipeline(config=cfg, run_manager=rm)

    # Execute requested mode
    if args.mode == "all":
        pipeline.run_all(
            force_recompute=args.force_recompute,
            tune_hyperparams=args.tune_hyperparams,
            n_trials=args.n_trials,
        )
    elif args.mode == "train":
        pipeline.run_train(
            force_recompute=args.force_recompute,
            tune_hyperparams=args.tune_hyperparams,
            n_trials=args.n_trials,
        )
    elif args.mode == "val":
        pipeline.run_val(force_recompute=args.force_recompute)
    elif args.mode == "test":
        pipeline.run_test(threshold=args.threshold, force_recompute=args.force_recompute)
    elif args.mode == "decide":
        if args.t1 is None or args.t2 is None:
            parser.error("--mode decide needs --t1 and --t2")
        pipeline.run_decide(args.t1, args.t2)


if __name__ == "__main__":
    main()
