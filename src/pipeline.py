"""
Pipeline Orchestrator - Coordinates Training, Validation, and Test Inference.
Implements the complete roadmap (Sections 11 to 19, 23, 24).
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import pandas as pd

from config import PipelineConfig
from src.data_loader import (
    load_ground_truth,
    split_s1_entities,
    get_all_s1_ids,
)
from src.blocking import (
    generate_candidate_pairs,
    iter_candidate_pairs,
    collect_raw_pairs,
    BlockRanker,
    ranker_matrix,
    block_score,
)
from src.features import FeatureExtractor, add_group_features
from src.model import EntityMatcherModel
from src.metrics import (
    sweep_thresholds,
    sweep_thresholds_2d,
    candidate_recall,
    decide_matches,
)
from src.run_manager import RunManager
from src.stacking import stack_features
from src.progress import PhaseProgress, pbar
from src.context import ContextBuilder, CONTEXT_COLUMNS, CTX_VERSION


# Non-feature columns stored alongside features in the feature checkpoints
META_COLS = ["s1_id", "candidate_id", "rule", "target_core", "target_addr", "label"]
STACK_META = ["s1_id", "target_core", "target_addr"]


def _disabled_features() -> List[str]:
    """Rule flags of retrieval fields switched off in BLOCKING_PARAMS (always 0 -> not model inputs)."""
    from config import DEFAULT_CONFIG
    return [f"rule_knn_{f}" for f, k in DEFAULT_CONFIG.BLOCKING_PARAMS["topk"].items() if not k]


def _model_input(X: pd.DataFrame) -> pd.DataFrame:
    return X.drop(columns=[c for c in _disabled_features() if c in X.columns])


def _feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    return _model_input(df.drop(columns=[c for c in META_COLS if c in df.columns]))


def _stage2_input(X: pd.DataFrame, meta: pd.DataFrame, p1: np.ndarray) -> pd.DataFrame:
    S = stack_features(meta.reset_index(drop=True), p1)
    return pd.concat([X.reset_index(drop=True), S], axis=1)


def _with_ctx(X: pd.DataFrame, ctx: Optional[pd.DataFrame]) -> pd.DataFrame:
    if ctx is None:
        return X
    X = X.drop(columns=[c for c in CONTEXT_COLUMNS if c in X.columns])
    return pd.concat([X.reset_index(drop=True), ctx.reset_index(drop=True)], axis=1)


def _sig_hash(obj) -> str:
    import hashlib
    return hashlib.md5(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:10]


def _atomic_save(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def _ids_match(cached_ids, expected_ids, min_cover: float = 0.9) -> bool:
    """A cached pair/feature file belongs to this split: its S1 ids are a subset of the expected ids and
    cover most of them (S1 without any candidate are legitimately absent)."""
    got = set(pd.unique(np.asarray(cached_ids, dtype=object)))
    exp = set(expected_ids)
    return bool(got) and got <= exp and len(got) >= min_cover * len(exp)


def _free_gpu() -> None:
    from src.blocking import _free_memory
    _free_memory()


def _label_pairs(pairs_df: pd.DataFrame, gt_map: Dict[str, set]) -> np.ndarray:
    return np.fromiter(
        (c in gt_map.get(s, ()) for s, c in zip(pairs_df["s1_id"].values, pairs_df["candidate_id"].values)),
        dtype=np.int32, count=len(pairs_df),
    )


class EntityResolutionPipeline:
    def __init__(self, config: PipelineConfig, run_manager: RunManager):
        self.config = config
        self.rm = run_manager

    def _blocking_params(self) -> Dict[str, Any]:
        """Blocking params plus the learned cap ranker of this run (if trained)."""
        p = dict(self.config.BLOCKING_PARAMS)
        path = self.rm.models_dir / "block_ranker.joblib"
        if self.config.USE_BLOCK_RANKER and path.is_file():
            import joblib
            ranker = joblib.load(path)
            # CPU on purpose: XGBoost keeps its GPU memory for the whole process, which starved the
            # cuSPARSE top-K search (GPU out-of-memory on India test blocking). The ranker is small.
            ranker.device = "cpu"
            p["ranker"] = ranker
        return p

    def _block(self, s1_ids: List[str], s1_path, s2_path, s3_path, ckpt_dir: Optional[Path] = None) -> pd.DataFrame:
        """
        Candidate pairs for s1_ids. With ckpt_dir the blocking is resumable (per country / source /
        chunk checkpoints, same machinery as test mode); the chunk files are merged and removed.
        """
        if ckpt_dir is None:
            return generate_candidate_pairs(
                s1_ids=s1_ids,
                s1_parquet_path=s1_path,
                s2_parquet_path=s2_path,
                s3_parquet_path=s3_path,
                max_candidates_per_s1=self.config.MAX_CANDIDATES_PER_S1,
                verbose=True,
                params=self._blocking_params(),
            )
        import shutil
        from src.blocking import block_to_checkpoints, blocking_signature, _free_memory, OUTPUT_COLUMNS
        bparams = self._blocking_params()
        sig = blocking_signature(s1_ids, self.config.MAX_CANDIDATES_PER_S1, bparams,
                                 self.rm.models_dir / "block_ranker.joblib")
        chunks = block_to_checkpoints(
            s1_ids=s1_ids, s1_parquet_path=s1_path, s2_parquet_path=s2_path, s3_parquet_path=s3_path,
            ckpt_dir=ckpt_dir, max_candidates_per_s1=self.config.MAX_CANDIDATES_PER_S1, params=bparams,
            chunk_s1=self.config.TEST_CHUNK_S1, signature=sig, verbose=True)
        del bparams
        _free_memory()
        out = (pd.concat([pd.read_parquet(c) for c in chunks], ignore_index=True) if chunks
               else pd.DataFrame(columns=OUTPUT_COLUMNS))
        return out

    def _context(self, meta: pd.DataFrame, test: bool, cache: Optional[Path], force: bool = False
                 ) -> Optional[pd.DataFrame]:
        """
        Context features (src/context.py) aligned with meta rows (needs s1_id, candidate_id).
        Cached at `cache` together with the ids, so a stale cache is detected and rebuilt.
        """
        if not self.config.USE_CONTEXT:
            return None
        ids = meta[["s1_id", "candidate_id"]].reset_index(drop=True)
        if cache is not None and cache.is_file() and not force:
            c = pd.read_parquet(cache)
            if (len(c) == len(ids) and set(CONTEXT_COLUMNS) <= set(c.columns)
                    and int(c["ctx_version"].iloc[0] if "ctx_version" in c.columns and len(c) else -1) == CTX_VERSION
                    and np.array_equal(c["s1_id"].values, ids["s1_id"].values)
                    and np.array_equal(c["candidate_id"].values, ids["candidate_id"].values)):
                print(f"  [Checkpoint] Loaded context features from {cache.name}")
                return c[CONTEXT_COLUMNS]
            print(f"  [Checkpoint] {cache.name} is stale -> rebuilding context features")
        paths = ((self.config.TEST_S1_PATH, self.config.TEST_S2_PATH, self.config.TEST_S3_PATH) if test else
                 (self.config.TRAIN_S1_PATH, self.config.TRAIN_S2_PATH, self.config.TRAIN_S3_PATH))
        t0 = time.time()
        print(f"  [Context] Rival-S1 + distinctive-token features for {len(ids):,} pairs "
              f"(rival tables are built once per dataset and cached in dataset/norm/ctx/)")
        cb = ContextBuilder(*paths, params=self.config.BLOCKING_PARAMS)
        feats = cb.features(ids["s1_id"].values, ids["candidate_id"].values)
        cb.close()
        if cache is not None:
            save = pd.concat([ids, feats], axis=1)
            save["ctx_version"] = np.int16(CTX_VERSION)
            _atomic_save(save, cache)
            print(f"  [Checkpoint] Saved context features to {cache.name} ({time.time() - t0:.0f}s)")
        return feats

    def _train_block_ranker(self, train_s1_ids: List[str], gt_map: Dict[str, set], force: bool,
                            prog: Optional[PhaseProgress] = None) -> None:
        """
        Learn which blocking candidates to keep under the per-S1 cap.
        Trains on raw (uncapped) candidates of a subset of TRAIN S1 entities only, and reports
        recall@cap for the heuristic vs the learned ranker on a held-out part of that subset.
        """
        import joblib
        path = self.rm.models_dir / "block_ranker.joblib"
        cur_topk = dict(self.config.BLOCKING_PARAMS["topk"])
        if path.is_file() and not force:
            old = joblib.load(path)
            from src.blocking import RANKER_FEATURES
            if getattr(old, "features", None) != list(RANKER_FEATURES):
                print("  [Ranker] Blocking signals changed (new rules / reverse retrieval): retraining the ranker")
                force = True
            elif getattr(old, "topk", None) != cur_topk:
                print(f"  [Ranker] Retrieval config changed ({getattr(old, 'topk', None)} -> {cur_topk}): retraining. "
                      f"NOTE: blocking checkpoints fingerprinted with the old ranker (e.g. test blocking copied by "
                      f"--init-from) will be rebuilt.")
                force = True
        if not self.config.USE_BLOCK_RANKER or (path.is_file() and not force):
            if prog:
                prog.skip("3a", "blocking ranker (cached/disabled)")
            return
        rng = np.random.default_rng(self.config.RANDOM_SEED)
        n = min(self.config.RANKER_TRAIN_S1, len(train_s1_ids))
        ids = list(rng.choice(train_s1_ids, size=n, replace=False))
        if prog:
            prog.step("3a", f"Training blocking cap ranker on raw candidates of {n:,} train S1")
        raw = collect_raw_pairs(ids, self.config.TRAIN_S1_PATH, self.config.TRAIN_S2_PATH,
                                self.config.TRAIN_S3_PATH, verbose=True, params=self.config.BLOCKING_PARAMS)
        if len(raw) == 0:
            print("  [Ranker] No raw pairs; keeping heuristic cap score.")
            return
        y = _label_pairs(raw, gt_map)
        X = ranker_matrix(raw)
        total_true = sum(len(gt_map.get(s, ())) for s in ids)
        print(f"  Raw pairs: {len(raw):,} ({len(raw) / n:.0f}/S1) | raw recall (no cap): {y.sum() / max(total_true, 1):.4f}")

        # Held-out comparison of recall@cap (80/20 split by S1)
        s1_arr = raw["s1_id"].values
        uniq = np.unique(s1_arr)
        hold = set(rng.choice(uniq, size=max(1, len(uniq) // 5), replace=False))
        is_hold = np.fromiter((s in hold for s in s1_arr), dtype=bool, count=len(s1_arr))
        dev = "cuda" if self.config.BLOCKING_PARAMS.get("use_gpu", True) else "cpu"
        r_eval = BlockRanker(device=dev, topk=cur_topk).fit(X[~is_hold], y[~is_hold])
        cap = self.config.MAX_CANDIDATES_PER_S1
        hold_true = sum(len(gt_map.get(s, ())) for s in hold)
        for name, score in [("heuristic", block_score(raw[is_hold])), ("learned", r_eval.predict(X[is_hold]))]:
            h = pd.DataFrame({"s": s1_arr[is_hold], "y": y[is_hold], "v": score})
            kept = h.sort_values(["s", "v"], ascending=[True, False]).groupby("s", sort=False).head(cap)
            print(f"  recall@{cap} on held-out S1 ({name:9s}): {kept['y'].sum() / max(hold_true, 1):.4f}")

        ranker = BlockRanker(device=dev, topk=cur_topk).fit(X, y)
        joblib.dump(ranker, path)
        print(f"  Saved blocking ranker to {path}")

    def _other_run_ids(self, runs: List[str], which=("train", "val")) -> set:
        out = set()
        for r in runs:
            for w in which:
                f = self.rm.runs_dir / r / f"{w}_s1_ids.npy"
                if not f.is_file():
                    raise FileNotFoundError(f"{f} not found (needed by --exclude-runs / --ensemble-with)")
                out.update(np.load(f, allow_pickle=True).tolist())
        return out

    def _split(self) -> Tuple[List[str], List[str]]:
        excl = self._other_run_ids(self.config.SAMPLE_EXCLUDE_RUNS) if self.config.SAMPLE_EXCLUDE_RUNS else None
        if excl:
            print(f"  Excluding {len(excl):,} S1 used by {self.config.SAMPLE_EXCLUDE_RUNS} (disjoint sample)")
        return split_s1_entities(
            s1_parquet_path=self.config.TRAIN_S1_PATH,
            train_ratio=self.config.TRAIN_SPLIT_RATIO,
            val_ratio=self.config.VAL_SPLIT_RATIO,
            random_seed=self.config.RANDOM_SEED,
            sample_size=self.config.SAMPLE_SIZE_S1,
            sample_seed=self.config.SAMPLE_SEED,
            exclude_ids=excl,
        )

    def _scorers(self, model, model2) -> List[Tuple[str, Any, Any]]:
        """This run's (stage-1, stage-2) models plus those of every --ensemble-with run."""
        out = [(self.rm.run_name, model, model2)]
        for r in self.config.ENSEMBLE_RUNS:
            mdir = self.rm.runs_dir / r / "models"
            m1 = EntityMatcherModel.load(mdir / "model.joblib")
            m2 = (EntityMatcherModel.load(mdir / "model_stage2.joblib")
                  if self.config.USE_STACKING and (mdir / "model_stage2.joblib").is_file() else None)
            out.append((r, m1, m2))
        if len(out) > 1:
            print(f"  [Ensemble] averaging {len(out)} runs: {[t for t, _, _ in out]}")
        return out

    def _ensemble_files(self) -> List[list]:
        files = []
        for r in self.config.ENSEMBLE_RUNS:
            for n in ("model.joblib", "model_stage2.joblib"):
                p = self.rm.runs_dir / r / "models" / n
                if p.is_file():
                    files.append([r, n, p.stat().st_size, int(p.stat().st_mtime)])
        return files

    def _orphan_weights(self, y: pd.Series, pairs: pd.DataFrame, gt_map: Dict[str, set]) -> Optional[np.ndarray]:
        """Same distractor weights as final training (negatives on targets owned by no S1)."""
        w_orphan = float(self.config.ORPHAN_NEG_WEIGHT or 1.0)
        if w_orphan == 1.0:
            return None
        owned = {t for ts in gt_map.values() for t in ts}
        orphan_neg = (np.asarray(y) == 0) & ~np.fromiter((c in owned for c in pairs["candidate_id"].values),
                                                         dtype=bool, count=len(pairs))
        return np.where(orphan_neg, w_orphan, 1.0).astype(np.float32)

    def _score(self, scorers, X: pd.DataFrame, meta: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Average stage-1 and final probabilities over all scorers (each run's own stage 2 uses its own p1)."""
        p1_sum = prob_sum = None
        for _, m1, m2 in scorers:
            p1 = np.asarray(m1.predict_proba(X, batch_size=self.config.BATCH_SIZE), dtype=np.float32)
            prob = (np.asarray(m2.predict_proba(_stage2_input(X, meta[STACK_META], p1),
                                                batch_size=self.config.BATCH_SIZE), dtype=np.float32)
                    if m2 is not None else p1)
            p1_sum = p1 if p1_sum is None else p1_sum + p1
            prob_sum = prob if prob_sum is None else prob_sum + prob
        k = np.float32(len(scorers))
        return p1_sum / k, prob_sum / k

    def _load_model_and_fe(self, model, fe):
        if model is None:
            model_path = self.rm.models_dir / "model.joblib"
            if not model_path.is_file():
                model_path = self.rm.models_dir / "lightgbm_model.joblib"
            if not model_path.is_file():
                raise FileNotFoundError(f"Model not found at {self.rm.models_dir}. Run training first.")
            model = EntityMatcherModel.load(model_path)
        if fe is None:
            fe_path = self.rm.models_dir / "feature_extractor.joblib"
            if not fe_path.is_file():
                raise FileNotFoundError(f"Feature extractor not found at {fe_path}. Run training first.")
            fe = FeatureExtractor.load(str(fe_path))
        return model, fe

    def _load_stage2(self, model2=None) -> Optional[EntityMatcherModel]:
        if model2 is not None or not self.config.USE_STACKING:
            return model2
        path = self.rm.models_dir / "model_stage2.joblib"
        return EntityMatcherModel.load(path) if path.is_file() else None

    def run_train(
        self,
        force_recompute: bool = False,
        tune_hyperparams: bool = False,
        n_trials: int = 20,
    ) -> Dict[str, Any]:
        """
        Execute Training Phase:
        1. Load Ground Truth.
        2. Perform 70-30 split by Source 1 entity.
        3. Block Train S1 against S2 and S3 (with candidate checkpointing).
        4. Label pairs using ground truth & sample hard + random negatives.
        5. Extract pairwise features (with feature checkpointing).
        6. Train model (GPU-accelerated XGBoost, LightGBM, or Ensemble).
        7. Save model, feature extractor, feature datasets, and feature importances.
        """
        print("\n" + "=" * 70)
        print(f"[{self.rm.run_name}] STARTING TRAINING PHASE (70% S1 Entity Split)")
        print("=" * 70)
        t_start = time.time()

        ckpt_train_pairs = self.rm.checkpoints_dir / "candidates_train.parquet"
        ckpt_train_features = self.rm.features_dir / "train_features.parquet"
        fe_path = self.rm.models_dir / "feature_extractor.joblib"

        train_ckpt = self.rm.checkpoints_dir / "train"
        if force_recompute and train_ckpt.exists():
            import shutil
            print(f"  [Checkpoint] --force-recompute: removing {train_ckpt}")
            shutil.rmtree(train_ckpt)
        train_ckpt.mkdir(parents=True, exist_ok=True)

        prog = PhaseProgress("TRAIN", ["1", "2", "3a", "3", "4", "5", "5b", "6"], 7, self.rm.metrics_dir)
        prog.step("1", "Loading Ground Truth labels")
        gt_map = load_ground_truth(self.config.TRAIN_GT_PATH)
        print(f"  Loaded GT mapping for {len(gt_map):,} Source 1 entities.")

        prog.step("2", f"Splitting Source 1 entities ({int(self.config.TRAIN_SPLIT_RATIO * 100)}-{int(self.config.VAL_SPLIT_RATIO * 100)} split)")
        train_s1_ids, val_s1_ids = self._split()
        print(f"  Train S1 entities: {len(train_s1_ids):,}")
        print(f"  Validation S1 entities: {len(val_s1_ids):,}")
        np.save(self.rm.run_dir / "train_s1_ids.npy", np.array(train_s1_ids))
        np.save(self.rm.run_dir / "val_s1_ids.npy", np.array(val_s1_ids))

        fe = FeatureExtractor()
        if ckpt_train_features.is_file() and not _ids_match(
                pd.read_parquet(ckpt_train_features, columns=["s1_id"])["s1_id"].values, train_s1_ids):
            print(f"  [Checkpoint] {ckpt_train_features.name} was built for a different S1 sample -> recomputing")
            ckpt_train_features.unlink()
            ckpt_train_pairs.unlink(missing_ok=True)
            (self.rm.features_dir / "train_context.parquet").unlink(missing_ok=True)
        if ckpt_train_pairs.is_file() and not _ids_match(
                pd.read_parquet(ckpt_train_pairs, columns=["s1_id"])["s1_id"].values, train_s1_ids):
            print(f"  [Checkpoint] {ckpt_train_pairs.name} was built for a different S1 sample -> re-blocking")
            ckpt_train_pairs.unlink()
        if not force_recompute and ckpt_train_features.is_file() and fe_path.is_file():
            self._train_block_ranker(train_s1_ids, gt_map, force=False, prog=prog)
            prog.step("3", f"Loading cached training features (steps 3-5) from {ckpt_train_features.name}")
            cached_df = pd.read_parquet(ckpt_train_features)
            train_pairs_df = cached_df[[c for c in META_COLS if c in cached_df.columns]]
            y_train = cached_df["label"]
            X_train = _feature_matrix(cached_df)
            fe = FeatureExtractor.load(str(fe_path))
            pos_count = int(y_train.sum())
            neg_count = len(y_train) - pos_count
            print(f"  Loaded {len(cached_df):,} cached training feature rows ({pos_count:,} pos, {neg_count:,} neg).")
            prog.skip("4")
            prog.skip("5")
        else:
            self._train_block_ranker(train_s1_ids, gt_map, force=force_recompute, prog=prog)
            if not force_recompute and ckpt_train_pairs.is_file():
                prog.step("3", f"Loading cached candidate pairs from {ckpt_train_pairs.name}")
                train_pairs_df = pd.read_parquet(ckpt_train_pairs)
            else:
                prog.step("3", f"Blocking: candidate pairs for {len(train_s1_ids):,} train S1 (resumable)")
                block_dir = train_ckpt / "blocking"
                train_pairs_df = self._block(train_s1_ids, self.config.TRAIN_S1_PATH,
                                             self.config.TRAIN_S2_PATH, self.config.TRAIN_S3_PATH,
                                             ckpt_dir=block_dir)
                _atomic_save(train_pairs_df, ckpt_train_pairs)
                print(f"  [Checkpoint] Saved candidate pairs to {ckpt_train_pairs}")
                import shutil
                shutil.rmtree(block_dir, ignore_errors=True)   # merged into candidates_train.parquet

            if len(train_pairs_df) == 0:
                raise RuntimeError("No candidate pairs generated during training blocking. Check dataset paths.")

            cr = candidate_recall(train_pairs_df, gt_map, train_s1_ids)
            print(f"  Train blocking recall ceiling: {cr['candidate_recall']:.4f} "
                  f"({cr['avg_candidates_per_s1']:.1f} candidates / S1)")

            prog.step("4", "Assigning labels from Ground Truth")
            train_pairs_df = train_pairs_df.reset_index(drop=True)
            train_pairs_df["label"] = _label_pairs(train_pairs_df, gt_map)
            pos_count = int(train_pairs_df["label"].sum())
            neg_count = len(train_pairs_df) - pos_count
            print(f"  Training pairs (raw): {len(train_pairs_df):,} | Positives: {pos_count:,} "
                  f"({pos_count / max(len(train_pairs_df), 1):.2%}) | Negatives: {neg_count:,}")

            # Within-S1 context features need the complete candidate group: compute before subsampling
            group_feats = add_group_features(train_pairs_df)

            n_hard = self.config.MAX_TRAIN_NEGATIVES_PER_S1
            if self.config.USE_STACKING and n_hard:
                print("  [Stacking] Negative subsampling disabled: stage 2 needs complete candidate groups.")
                n_hard = None
            if n_hard is not None and n_hard > 0:
                neg = train_pairs_df[train_pairs_df["label"] == 0]
                neg_sorted = neg.sort_values(["s1_id", "block_score"], ascending=[True, False])
                hard_idx = neg_sorted.groupby("s1_id", sort=False).head(n_hard).index
                rest = neg.drop(index=hard_idx)
                n_rand = self.config.RANDOM_TRAIN_NEGATIVES_PER_S1
                rand_idx = (rest.sample(frac=1.0, random_state=self.config.RANDOM_SEED)
                                .groupby("s1_id", sort=False).head(n_rand).index) if n_rand > 0 else []
                pos_idx = train_pairs_df.index[train_pairs_df["label"] == 1]
                keep = np.sort(np.concatenate([pos_idx.values, np.asarray(hard_idx), np.asarray(rand_idx)])).astype(np.int64)
                train_pairs_df = train_pairs_df.loc[keep].reset_index(drop=True)
                group_feats = group_feats.loc[keep].reset_index(drop=True)
                neg_count = len(train_pairs_df) - pos_count
                print(f"  Sampled training pairs: {len(train_pairs_df):,} ({pos_count:,} positives, "
                      f"{neg_count:,} negatives: top-{n_hard} hardest + {n_rand} random per S1)")

            prog.step("5", f"Extracting pairwise features for {len(train_pairs_df):,} pairs")
            X_train = _model_input(fe.extract_features(train_pairs_df, batch_size=self.config.BATCH_SIZE,
                                                       group_df=group_feats))
            y_train = train_pairs_df["label"]

            fe.save(str(fe_path))
            print(f"  Saved Feature Extractor to {fe_path}")

            cached_save_df = pd.concat([
                train_pairs_df[["s1_id", "candidate_id", "rule", "target_core", "target_addr"]].reset_index(drop=True),
                X_train.reset_index(drop=True),
                y_train.rename("label").reset_index(drop=True),
            ], axis=1)
            train_pairs_df = cached_save_df[META_COLS]
            _atomic_save(cached_save_df, ckpt_train_features)
            print(f"  [Checkpoint] Saved model training features dataset to {ckpt_train_features}")
            del cached_save_df

        prog.step("5b", "Context features: rival S1 businesses + distinctive tokens")
        ctx = self._context(train_pairs_df, test=False, cache=self.rm.features_dir / "train_context.parquet")
        X_train = _with_ctx(X_train, ctx)
        del ctx

        if tune_hyperparams:
            from src.tuner import EntityResolutionTuner
            print("\n[Optuna] Tuning hyperparameters...")
            tuner = EntityResolutionTuner(
                model_type=self.config.MODEL_TYPE,
                device=self.config.DEVICE,
                n_trials=n_trials,
                timeout=self.config.TUNE_TIMEOUT_MIN * 60,
                random_seed=self.config.RANDOM_SEED,
                train_frac=self.config.TUNE_TRAIN_FRAC,
            )
            val_ckpt_feat = self.rm.features_dir / "val_features.parquet"
            if not val_ckpt_feat.is_file():
                raise FileNotFoundError(
                    f"Optuna needs cached validation features at {val_ckpt_feat}: run --mode val once first.")
            val_cached = pd.read_parquet(val_ckpt_feat)
            X_v = _with_ctx(_feature_matrix(val_cached),
                            self._context(val_cached, test=False, cache=self.rm.features_dir / "val_context.parquet"))
            val_p_df = val_cached[["s1_id", "candidate_id", "rule"]]
            val_ids_path = self.rm.run_dir / "val_s1_ids.npy"
            val_ids = (list(np.load(val_ids_path, allow_pickle=True)) if val_ids_path.is_file()
                       else list(val_p_df["s1_id"].unique()))
            if self.config.VAL_SAMPLE_SIZE_S1 and len(val_ids) > self.config.VAL_SAMPLE_SIZE_S1:
                val_ids = list(val_p_df["s1_id"].unique())

            best_p, best_score, _ = tuner.tune(
                X_train=X_train,
                y_train=y_train,
                X_val=X_v,
                val_pairs_df=val_p_df,
                ground_truth_map=gt_map,
                val_s1_ids=val_ids,
                output_dir=self.rm.run_dir / "optuna",
                train_groups=train_pairs_df["s1_id"].values,
                sample_weight=self._orphan_weights(y_train, train_pairs_df, gt_map),
                fp_weight=float(self.config.THRESHOLD_FP_WEIGHT or 1.0),
            )
            model_params = best_p
        else:
            xgb_params = {**self.config.XGBOOST_PARAMS, "device": self.config.DEVICE}
            if self.config.MODEL_TYPE == "xgboost":
                model_params = xgb_params
            elif self.config.MODEL_TYPE == "lightgbm":
                model_params = self.config.LIGHTGBM_PARAMS
            else:  # ensemble: XGBoost on GPU + LightGBM on CPU, trained concurrently
                model_params = {"xgb_params": xgb_params, "lgb_params": self.config.LIGHTGBM_PARAMS}

        def new_model():
            return EntityMatcherModel(model_type=self.config.MODEL_TYPE, device=self.config.DEVICE,
                                      params=model_params)

        prog.step("6", f"Training {self.config.MODEL_TYPE.upper()} (device: {self.config.DEVICE})"
                       + (" + stage-2 stacking" if self.config.USE_STACKING else ""))
        # Checkpointed training: every OOF fold, the stage-2 model and the stage-1 model are saved as
        # soon as they finish, keyed by a signature of (features, rows, params). Rerunning the same
        # command after a crash resumes at the first unfinished piece; a finished run is not retrained.
        X_train = X_train.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)
        n_folds = self.config.STACK_OOF_FOLDS
        # Distractor-aware weights. The test set has ~40% targets that belong to no S1 business
        # (unmatched in our test predictions; 23% more targets per S1 than train) vs 26% in the training
        # ground truth, and leaderboard probes confirmed false matches cost more on test than on
        # validation. Negatives whose target has NO owner in the ground truth are up-weighted by the
        # odds ratio so the model's probabilities reflect the test-time distractor density.
        owned = {t for ts in gt_map.values() for t in ts}
        w_orphan = float(self.config.ORPHAN_NEG_WEIGHT or 1.0)
        orphan_neg = ((y_train.values == 0)
                      & ~np.fromiter((c in owned for c in train_pairs_df["candidate_id"].values), dtype=bool,
                                     count=len(train_pairs_df)))
        weights = np.where(orphan_neg, w_orphan, 1.0).astype(np.float32) if w_orphan != 1.0 else None
        print(f"  [Weights] negatives on unowned (distractor) targets: {orphan_neg.sum():,} "
              f"({orphan_neg.mean():.1%} of pairs) x{w_orphan}")
        del owned
        tsig = _sig_hash({"features": list(X_train.columns), "n": len(X_train), "pos": int(y_train.sum()),
                          "model_type": self.config.MODEL_TYPE, "device": self.config.DEVICE,
                          "params": model_params, "stacking": self.config.USE_STACKING, "folds": n_folds,
                          "orphan_weight": w_orphan})
        state_path = self.rm.models_dir / "train_state.json"
        state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else {}
        if state.get("sig") != tsig or force_recompute:
            state = {"sig": tsig}

        def mark(key):
            state[key] = True
            state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")

        model2 = None
        if self.config.USE_STACKING:
            from sklearn.model_selection import GroupKFold
            stage2_path = self.rm.models_dir / "model_stage2.joblib"
            if state.get("stage2") and stage2_path.is_file():
                print("  [Checkpoint] Stage-2 model up to date -> loaded (OOF folds skipped)")
                model2 = EntityMatcherModel.load(stage2_path)
            else:
                oof_dir = train_ckpt / "oof"
                oof_dir.mkdir(parents=True, exist_ok=True)
                print(f"  [Stacking] Stage 1: {n_folds}-fold out-of-fold predictions (grouped by S1)...")
                groups = train_pairs_df["s1_id"].values
                oof = np.zeros(len(X_train), dtype=np.float32)
                folds = list(GroupKFold(n_splits=n_folds).split(X_train, y_train, groups))
                for k, (tr_i, va_i) in enumerate(pbar(folds, total=n_folds, desc="OOF folds", unit="fold", leave=True)):
                    fpath = oof_dir / f"fold{k}_{tsig}.npy"
                    if fpath.is_file():
                        oof[va_i] = np.load(fpath)
                        print(f"  [Checkpoint] OOF fold {k + 1}/{n_folds} loaded")
                        continue
                    m = new_model()
                    m.train(X_train.iloc[tr_i], y_train.iloc[tr_i],
                            sample_weight=None if weights is None else weights[tr_i])
                    pred = m.predict_proba(X_train.iloc[va_i], batch_size=self.config.BATCH_SIZE).astype(np.float32)
                    np.save(fpath.with_suffix(".tmp.npy"), pred)
                    os.replace(fpath.with_suffix(".tmp.npy"), fpath)
                    oof[va_i] = pred
                    del m
                    _free_gpu()
                print("  [Stacking] Stage 2: training on stage-1 features + sibling/stacking features...")
                X2 = _stage2_input(X_train, train_pairs_df[STACK_META], oof)
                model2 = new_model()
                model2.train(X2, y_train, sample_weight=weights)
                model2.save(stage2_path)
                mark("stage2")
                del X2
                _free_gpu()
            imp2 = model2.get_feature_importance()
            imp2.to_csv(self.rm.metrics_dir / "feature_importance_stage2.csv", index=False)
            print(f"  Stage-2 Top 10 Features:\n{imp2.head(10).to_string(index=False)}")

        stage1_path = self.rm.models_dir / "model.joblib"
        if state.get("stage1") and stage1_path.is_file():
            print("  [Checkpoint] Stage-1 model up to date -> loaded")
            model = EntityMatcherModel.load(stage1_path)
        else:
            print("  Stage 1: training on all training pairs...")
            model = new_model()
            model.train(X_train, y_train, sample_weight=weights)
            model.save(stage1_path)
            model.save(self.rm.models_dir / "lightgbm_model.joblib")
            mark("stage1")
        # finished: fold checkpoints are no longer needed
        import shutil
        shutil.rmtree(train_ckpt / "oof", ignore_errors=True)

        df_imp = model.get_feature_importance()
        imp_path = self.rm.metrics_dir / "feature_importance.csv"
        df_imp.to_csv(imp_path, index=False)
        print(f"  Stage-1 Top 10 Features:\n{df_imp.head(10).to_string(index=False)}")

        prog.done()
        elapsed = time.time() - t_start
        print(f"\nTraining Phase Complete! Total time: {elapsed:.2f}s")
        return {
            "train_pairs_count": len(train_pairs_df),
            "positives": pos_count,
            "negatives": neg_count,
            "train_s1_count": len(train_s1_ids),
            "elapsed_seconds": elapsed,
            "model": model,
            "model2": model2,
            "fe": fe,
        }

    def run_val(
        self,
        model: Optional[EntityMatcherModel] = None,
        fe: Optional[FeatureExtractor] = None,
        force_recompute: bool = False,
        model2: Optional[EntityMatcherModel] = None,
    ) -> Dict[str, Any]:
        """
        Execute Validation Phase:
        1. Load validation S1 entities (30% split).
        2. Block validation S1 against S2 and S3; report blocking recall ceiling.
        3. Extract features and predict match probabilities.
        4. Sweep (t1, t2) thresholds with target exclusivity to optimize Macro F_0.5.
        5. Save metrics and reports.
        """
        print("\n" + "=" * 70)
        print(f"[{self.rm.run_name}] STARTING VALIDATION & THRESHOLD TUNING (30% S1 Entity Split)")
        print("=" * 70)
        t_start = time.time()
        model, fe = self._load_model_and_fe(model, fe)
        model2 = self._load_stage2(model2)
        gt_map = load_ground_truth(self.config.TRAIN_GT_PATH)

        val_s1_ids_path = self.rm.run_dir / "val_s1_ids.npy"
        if val_s1_ids_path.is_file():
            val_s1_ids = list(np.load(val_s1_ids_path, allow_pickle=True))
        else:
            _, val_s1_ids = self._split()
        val_ckpt = self.rm.checkpoints_dir / "val"
        if force_recompute and val_ckpt.exists():
            import shutil
            shutil.rmtree(val_ckpt)
        prog = PhaseProgress("VAL", ["1", "2", "3", "3b", "4"], 5, self.rm.metrics_dir)
        prog.step("1", f"Validation set: {len(val_s1_ids):,} Source 1 entities")

        if self.config.VAL_SAMPLE_SIZE_S1 is not None and len(val_s1_ids) > self.config.VAL_SAMPLE_SIZE_S1:
            rng = np.random.default_rng(self.config.RANDOM_SEED)
            eval_val_s1_ids = list(rng.choice(val_s1_ids, size=self.config.VAL_SAMPLE_SIZE_S1, replace=False))
            print(f"  [Validation] Evaluating on sample of {len(eval_val_s1_ids):,} / {len(val_s1_ids):,} S1 entities.")
        else:
            eval_val_s1_ids = val_s1_ids

        ckpt_val_pairs = self.rm.checkpoints_dir / "val_pairs.parquet"
        ckpt_val_features = self.rm.features_dir / "val_features.parquet"

        for ck in (ckpt_val_features, ckpt_val_pairs):
            if ck.is_file() and not _ids_match(pd.read_parquet(ck, columns=["s1_id"])["s1_id"].values,
                                               eval_val_s1_ids):
                print(f"  [Checkpoint] {ck.name} was built for a different validation sample -> recomputing")
                ck.unlink()
                (self.rm.features_dir / "val_context.parquet").unlink(missing_ok=True)
        if not force_recompute and ckpt_val_features.is_file():
            prog.step("2", f"Loading cached validation features from {ckpt_val_features.name}")
            cached_val_df = pd.read_parquet(ckpt_val_features)
            X_val = _feature_matrix(cached_val_df)
            val_pairs_df = cached_val_df[[c for c in META_COLS if c in cached_val_df.columns]]
            print(f"  Loaded {len(cached_val_df):,} cached validation feature rows.")
            prog.skip("3")
        else:
            if not force_recompute and ckpt_val_pairs.is_file():
                prog.step("2", f"Loading cached validation pairs from {ckpt_val_pairs.name}")
                val_pairs_df = pd.read_parquet(ckpt_val_pairs)
            else:
                prog.step("2", f"Blocking: candidate pairs for {len(eval_val_s1_ids):,} validation S1")
                block_dir = val_ckpt / "blocking"
                val_pairs_df = self._block(eval_val_s1_ids, self.config.TRAIN_S1_PATH,
                                           self.config.TRAIN_S2_PATH, self.config.TRAIN_S3_PATH,
                                           ckpt_dir=block_dir)
                _atomic_save(val_pairs_df, ckpt_val_pairs)
                print(f"  [Checkpoint] Saved validation candidate pairs to {ckpt_val_pairs}")
                import shutil
                shutil.rmtree(block_dir, ignore_errors=True)

            if len(val_pairs_df) == 0:
                print("Warning: No validation candidates generated.")
                return {"optimal_threshold": self.config.DEFAULT_THRESHOLD, "best_macro_f05": 0.0}

            prog.step("3", f"Extracting features + scoring {len(val_pairs_df):,} validation pairs")
            val_pairs_df = val_pairs_df.reset_index(drop=True)
            X_val = _model_input(fe.extract_features(val_pairs_df, batch_size=self.config.BATCH_SIZE))
            val_cached_save = pd.concat([
                val_pairs_df[["s1_id", "candidate_id", "rule", "target_core", "target_addr"]].reset_index(drop=True),
                X_val.reset_index(drop=True),
            ], axis=1)
            _atomic_save(val_cached_save, ckpt_val_features)
            print(f"  [Checkpoint] Saved validation features dataset to {ckpt_val_features}")
            val_pairs_df = val_cached_save[["s1_id", "candidate_id", "rule", "target_core", "target_addr"]]

        prog.step("3b", "Context features: rival S1 businesses + distinctive tokens")
        X_val = _with_ctx(X_val, self._context(val_pairs_df, test=False,
                                               cache=self.rm.features_dir / "val_context.parquet",
                                               force=force_recompute))
        if self.config.ENSEMBLE_RUNS:
            leak = set(eval_val_s1_ids) & self._other_run_ids(self.config.ENSEMBLE_RUNS, which=("train",))
            if leak:
                print(f"  [Ensemble WARNING] {len(leak):,} validation S1 were TRAINING data of an ensembled run: "
                      f"validation scores and thresholds will be optimistic. Train the new run with "
                      f"--exclude-runs {','.join(self.config.ENSEMBLE_RUNS)}.")
        print(f"  Predicting match probabilities with {model.model_type.upper()} (stage 1 + stage 2)...")
        p1_val, val_probs = self._score(self._scorers(model, model2), X_val, val_pairs_df)

        cr = candidate_recall(val_pairs_df, gt_map, eval_val_s1_ids)
        print(f"  Blocking recall ceiling: {cr['candidate_recall']:.4f} ({cr['avg_candidates_per_s1']:.1f} candidates / S1)")

        prog.step("4", "Sweeping (t1, t2) thresholds to maximize Macro F0.5")
        sweep_kw = dict(
            val_pairs_df=val_pairs_df, val_probs=val_probs, ground_truth_map=gt_map,
            val_s1_ids=eval_val_s1_ids, min_thresh=self.config.THRESHOLD_SWEEP_MIN,
            max_thresh=self.config.THRESHOLD_SWEEP_MAX, step=self.config.THRESHOLD_SWEEP_STEP,
            max_matches_per_s1=self.config.MAX_MATCHES_PER_S1,
        )
        opt_1d, sweep_df = sweep_thresholds(exclusive_targets=self.config.EXCLUSIVE_TARGETS, **sweep_kw)
        fpw = float(self.config.THRESHOLD_FP_WEIGHT or 1.0)
        (t1, t2), sweep2_df = sweep_thresholds_2d(exclusive_targets=self.config.EXCLUSIVE_TARGETS,
                                                  fp_weight=fpw, **sweep_kw)
        val_opt = sweep2_df.loc[sweep2_df["macro_f05"].idxmax()]
        _, sweep_noex = sweep_thresholds(exclusive_targets=False, **sweep_kw)
        f05_stage1 = None
        if model2 is not None:
            _, sw1 = sweep_thresholds_2d(exclusive_targets=self.config.EXCLUSIVE_TARGETS,
                                         **{**sweep_kw, "val_probs": p1_val})
            f05_stage1 = float(sw1.iloc[0]["macro_f05"])

        sweep_df.to_csv(self.rm.metrics_dir / "threshold_sweep.csv", index=False)
        sweep2_df.to_csv(self.rm.metrics_dir / "threshold_sweep_2d.csv", index=False)
        print(f"  Saved threshold sweeps to {self.rm.metrics_dir}")

        best = sweep2_df.iloc[0]
        if self.config.MIN_PRECISION:
            # Recall-first: most recall whose precision stays >= MIN_PRECISION (ties -> higher F0.5)
            ok = sweep2_df[sweep2_df["precision"] >= self.config.MIN_PRECISION]
            if len(ok):
                best = ok.sort_values(["recall", "macro_f05"], ascending=False).iloc[0]
                t1, t2 = float(best["t1"]), float(best["t2"])
                print(f"  [Recall-first] thresholds chosen for precision >= {self.config.MIN_PRECISION} "
                      f"(F0.5-optimal was {sweep2_df.iloc[0]['macro_f05']:.4f})")
            else:
                print(f"  [Recall-first] no threshold reaches precision {self.config.MIN_PRECISION}; using F0.5-optimal")
        best_1d = sweep_df[sweep_df["threshold"] == opt_1d].iloc[0]
        best_noex = sweep_noex.loc[sweep_noex["macro_f05"].idxmax()]
        print("\n" + "-" * 60)
        print(f"  Blocking recall ceiling : {cr['candidate_recall']:.4f}")
        print(f"  Single threshold, no exclusivity : F0.5 {best_noex['macro_f05']:.4f} @ {best_noex['threshold']:.2f}")
        print(f"  Single threshold + exclusivity   : F0.5 {best_1d['macro_f05']:.4f} @ {opt_1d:.2f}")
        if f05_stage1 is not None:
            print(f"  Stage 1 only (t1/t2 + excl.)     : F0.5 {f05_stage1:.4f}")
        print(f"  Validation-optimal thresholds  : t1={val_opt['t1']:.2f} t2={val_opt['t2']:.2f} "
              f"(F0.5 {val_opt['macro_f05']:.4f})")
        print(f"CHOSEN for test (false matches weighted x{fpw}): t1={t1:.2f}  t2={t2:.2f}")
        print(f"  Macro F_0.5 Score     : {best['macro_f05']:.4f}")
        print(f"  Precision             : {best['precision']:.4f}")
        print(f"  Recall                : {best['recall']:.4f}")
        print(f"  Singleton Accuracy    : {best['singleton_accuracy']:.2%}")
        print(f"  Total Matches Predict : {int(best['total_matches_predicted']):,}")
        print("-" * 60)

        # Diagnostics: per country, and on "look-alike" S1 (core name shared with another S1 business,
        # the situation that is ~20x more frequent in the unseen French test data).
        diag = {}
        try:
            from src.metrics import entity_f05_vector
            from src.context import _countries_of
            from src.text_norm import ensure_norm
            fvec = entity_f05_vector(val_pairs_df, val_probs, gt_map, eval_val_s1_ids, t1, t2,
                                     self.config.MAX_MATCHES_PER_S1, self.config.EXCLUSIVE_TARGETS)
            ids_arr = np.asarray(eval_val_s1_ids, dtype=object)
            ctry = _countries_of(ids_arr, ensure_norm(self.config.TRAIN_S1_PATH))
            for c in sorted(set(ctry) - {""}):
                diag[f"macro_f05_{c}"] = float(fvec[ctry == c].mean())
            if "s1freq_own_name_log" in X_val.columns:
                shared = (pd.Series(X_val["s1freq_own_name_log"].values, index=val_pairs_df["s1_id"].values)
                          .groupby(level=0).max() >= np.log1p(2) - 1e-6)
                look = pd.Series(ids_arr).map(shared).fillna(False).values.astype(bool)
                diag["lookalike_s1_share"] = float(look.mean())
                diag["macro_f05_lookalike"] = float(fvec[look].mean()) if look.any() else None
            print("  Diagnostics: " + ", ".join(f"{k}={v:.4f}" for k, v in diag.items() if v is not None))
        except Exception as e:  # diagnostics must never break validation
            print(f"  [Diagnostics skipped: {e}]")

        metrics_summary = {
            "optimal_threshold": t2,
            "threshold_fp_weight": fpw,
            "val_optimal_t1": float(val_opt["t1"]),
            "val_optimal_t2": float(val_opt["t2"]),
            "val_optimal_macro_f05": float(val_opt["macro_f05"]),
            "threshold_top1": t1,
            "threshold_rest": t2,
            "best_macro_f05": float(best["macro_f05"]),
            "precision": float(best["precision"]),
            "recall": float(best["recall"]),
            "singleton_accuracy": float(best["singleton_accuracy"]),
            "candidate_recall": float(cr["candidate_recall"]),
            "avg_candidates_per_s1": float(cr["avg_candidates_per_s1"]),
            "macro_f05_single_threshold": float(best_1d["macro_f05"]),
            "macro_f05_no_exclusivity": float(best_noex["macro_f05"]),
            "macro_f05_stage1_only": f05_stage1,
            **diag,
            "context_features": bool(self.config.USE_CONTEXT),
            "ensemble_runs": [self.rm.run_name] + list(self.config.ENSEMBLE_RUNS),
            "stacking": model2 is not None,
            "exclusive_targets": bool(self.config.EXCLUSIVE_TARGETS),
            "val_s1_entities": len(eval_val_s1_ids),
            "val_pairs_count": len(val_pairs_df),
            "elapsed_seconds": time.time() - t_start,
        }
        self.rm.save_metrics(metrics_summary)
        prog.done()
        return metrics_summary

    def _resolve_thresholds(self, threshold: Optional[float]) -> Tuple[float, float]:
        if threshold is not None:
            return float(threshold), float(threshold)
        metrics_path = self.rm.metrics_dir / "metrics.json"
        if metrics_path.is_file():
            with open(metrics_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            t2 = data.get("threshold_rest", data.get("optimal_threshold", self.config.DEFAULT_THRESHOLD))
            t1 = data.get("threshold_top1", t2)
            return float(t1), float(t2)
        return self.config.DEFAULT_THRESHOLD, self.config.DEFAULT_THRESHOLD

    def run_test(
        self,
        model: Optional[EntityMatcherModel] = None,
        fe: Optional[FeatureExtractor] = None,
        threshold: Optional[float] = None,
        threshold_top1: Optional[float] = None,
        force_recompute: bool = False,
        model2: Optional[EntityMatcherModel] = None,
    ) -> Dict[str, Any]:
        """
        Execute Test Inference Phase:
        1. Load all test S1 IDs.
        2. For each chunk of S1: block against S2/S3, extract features, predict P(same business).
        3. Apply target exclusivity and the (t1, t2) thresholds.
        4. Write matching_results.tsv / candidate_pairs.tsv (every test S1 gets a row).
        5. Run utils/validate_submission.py.
        """
        print("\n" + "=" * 70)
        print(f"[{self.rm.run_name}] STARTING TEST INFERENCE & SUBMISSION GENERATION")
        print("=" * 70)
        t_start = time.time()
        model, fe = self._load_model_and_fe(model, fe)
        model2 = self._load_stage2(model2)
        print(f"  Scoring: stage 1{' + stage-2 stacking' if model2 is not None else ' only'}")

        t1, t2 = self._resolve_thresholds(threshold)
        # Thresholds are only valid for the model set they were tuned on: follow metrics.json
        mpath = self.rm.metrics_dir / "metrics.json"
        if mpath.is_file():
            tuned = json.loads(mpath.read_text(encoding="utf-8")).get("ensemble_runs", [self.rm.run_name])[1:]
            if list(tuned) != list(self.config.ENSEMBLE_RUNS):
                print(f"  [Ensemble] thresholds in metrics.json were tuned for ensemble {tuned or 'none'} "
                      f"-> using that (not {self.config.ENSEMBLE_RUNS or 'none'})")
                self.config.ENSEMBLE_RUNS = list(tuned)
        if threshold_top1 is not None:
            t1 = float(threshold_top1)
        print(f"  Decision thresholds: top-1 t1={t1:.4f}, others t2={t2:.4f} "
              f"(target exclusivity: {self.config.EXCLUSIVE_TARGETS})")

        prog = PhaseProgress("TEST", ["1", "2", "3", "4", "5"], 5, self.rm.metrics_dir)
        prog.step("1", "Loading Test Source 1 entity IDs")
        all_test_s1_ids = get_all_s1_ids(self.config.TEST_S1_PATH, limit=self.config.SAMPLE_SIZE_S1)
        print(f"  Loaded {len(all_test_s1_ids):,} Test Source 1 entities.")

        # ------------------------------------------------------------------
        # Resumable test pipeline. Everything is checkpointed under checkpoints/test/:
        #   blocking/<country>/chunks/chunk_*.parquet   candidate pairs (per 25k S1)
        #   scoring/<country>__chunk_*.parquet          kept scored pairs of that chunk
        #   scoring/<country>__chunk_*.cand.tsv         candidate_pairs.tsv lines of that chunk
        # A crash or Ctrl+C loses at most the chunk in progress; rerun the same command to resume.
        # --force-recompute discards all test checkpoints.
        # ------------------------------------------------------------------
        import shutil
        from src.blocking import block_to_checkpoints, blocking_signature, _free_memory
        test_ckpt = self.rm.checkpoints_dir / "test"
        if force_recompute and test_ckpt.exists():
            print(f"  [Checkpoint] --force-recompute: removing {test_ckpt}")
            shutil.rmtree(test_ckpt)
        block_dir, score_dir = test_ckpt / "blocking", test_ckpt / "scoring"
        score_dir.mkdir(parents=True, exist_ok=True)
        ckpt_test_scored = self.rm.checkpoints_dir / "test_scored.parquet"

        # ---- Stage 2: blocking (checkpointed per country / source / chunk) ----
        prog.step("2", f"Blocking {len(all_test_s1_ids):,} test S1 (resumable)")
        bparams = self._blocking_params()
        bsig = blocking_signature(all_test_s1_ids, self.config.MAX_CANDIDATES_PER_S1, bparams,
                                  self.rm.models_dir / "block_ranker.joblib")
        chunk_paths = block_to_checkpoints(
            s1_ids=all_test_s1_ids,
            s1_parquet_path=self.config.TEST_S1_PATH,
            s2_parquet_path=self.config.TEST_S2_PATH,
            s3_parquet_path=self.config.TEST_S3_PATH,
            ckpt_dir=block_dir,
            max_candidates_per_s1=self.config.MAX_CANDIDATES_PER_S1,
            params=bparams,
            chunk_s1=self.config.TEST_CHUNK_S1,
            signature=bsig,
            verbose=True,
        )
        del bparams
        _free_memory()

        # ---- Stage 3: features + scoring (checkpointed per chunk) ----
        keep_floor = max(0.0, min(t1, t2) - self.config.TEST_KEEP_MARGIN)
        ssig = {"blocking": bsig, "keep_floor": round(keep_floor, 4),
                "models": [[p.name, p.stat().st_size, int(p.stat().st_mtime)]
                           for p in [self.rm.models_dir / "model.joblib", self.rm.models_dir / "model_stage2.joblib"]
                           if p.is_file()] + [bool(model2 is not None)],
                "context": CTX_VERSION if self.config.USE_CONTEXT else None,
                "ensemble": self._ensemble_files()}
        ssig_path = score_dir / "signature.json"
        if ssig_path.is_file() and json.loads(ssig_path.read_text(encoding="utf-8")) != ssig:
            print("  [Checkpoint] model / thresholds / blocking changed -> discarding scoring checkpoints")
            shutil.rmtree(score_dir)
            score_dir.mkdir(parents=True, exist_ok=True)
        ssig_path.write_text(json.dumps(ssig, indent=2), encoding="utf-8")

        def part_paths(cp_: Path):
            stem = f"{cp_.parent.parent.name}__{cp_.stem}"
            return score_dir / f"{stem}.parquet", score_dir / f"{stem}.cand.tsv"

        todo = [c for c in chunk_paths if not part_paths(c)[0].is_file()]
        scorers = self._scorers(model, model2)
        prog.step("3", f"Features + scoring {len(chunk_paths)} chunks ({len(chunk_paths) - len(todo)} already done)")
        ctx_builder = None
        if self.config.USE_CONTEXT and todo:
            # Rival tables (reverse retrieval of every test target against the whole test S1 file) are
            # built once per country and cached in dataset/norm/ctx/ (~80s per 700k targets on GPU).
            ctx_builder = ContextBuilder(self.config.TEST_S1_PATH, self.config.TEST_S2_PATH,
                                         self.config.TEST_S3_PATH, params=self.config.BLOCKING_PARAMS)
            ctx_builder.prepare(sorted({c.parent.parent.name for c in todo}))
        chunk_bar = pbar(total=len(chunk_paths), initial=len(chunk_paths) - len(todo),
                         desc="test chunks scored", unit="chunk", leave=True)
        for cpath in todo:
            tc = time.time()
            pq_out, tsv_out = part_paths(cpath)
            chunk_df = pd.read_parquet(cpath)
            X = _model_input(fe.extract_features(chunk_df, batch_size=self.config.BATCH_SIZE))
            if ctx_builder is not None:
                country = cpath.parent.parent.name
                X = _with_ctx(X, ctx_builder.features(chunk_df["s1_id"].values, chunk_df["candidate_id"].values,
                                                      np.full(len(chunk_df), country, dtype=object)))
            p1, probs = self._score(scorers, X, chunk_df)
            del X
            probs = np.asarray(probs, dtype=np.float32).copy()
            probs[chunk_df["rule"].values == "exact_name_addr"] = 1.0
            # candidate_pairs.tsv lines for this chunk (each S1 lives in exactly one chunk)
            cdf = pd.DataFrame({"s1_id": chunk_df["s1_id"].values, "c": chunk_df["candidate_id"].values,
                                "p": probs}).sort_values(["s1_id", "p"], ascending=[True, False], kind="stable")
            lines = [f"{s1_id}\t{','.join(cands)}\n" for s1_id, cands in cdf.groupby("s1_id", sort=False)["c"]]
            tmp = tsv_out.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8", newline="") as fh:
                fh.writelines(lines)
            os.replace(tmp, tsv_out)
            keep = probs >= keep_floor
            part = pd.DataFrame({
                "s1_id": chunk_df["s1_id"].values[keep],
                "candidate_id": chunk_df["candidate_id"].values[keep],
                "s1_country": chunk_df["s1_country"].values[keep],
                "p1": np.asarray(p1, dtype=np.float32)[keep],
                "prob": probs[keep],
            })
            tmp = pq_out.with_suffix(".tmp")
            part.to_parquet(tmp, index=False)
            os.replace(tmp, pq_out)          # parquet written last = chunk is complete
            chunk_bar.update(1)
            chunk_bar.set_postfix_str(f"{len(chunk_df):,} pairs in {time.time() - tc:.0f}s, kept {len(part):,}")
            del chunk_df, cdf, lines, part, probs, p1
            _free_memory()
        chunk_bar.close()
        if ctx_builder is not None:
            ctx_builder.close()
            del ctx_builder
            _free_memory()

        # ---- Assemble: candidate_pairs.tsv + kept scored pairs ----
        written_s1 = set()
        cand_tmp = self.rm.output_dir / "candidate_pairs.partial.tsv"
        with open(cand_tmp, "w", encoding="utf-8", newline="") as out_f:
            out_f.write("source1_entity_id\tcandidate_entity_ids\n")
            for cpath in chunk_paths:
                with open(part_paths(cpath)[1], "r", encoding="utf-8") as in_f:
                    for line in in_f:
                        out_f.write(line)
                        written_s1.add(line.split("\t", 1)[0])
            for s1_id in all_test_s1_ids:          # S1 without any candidate still need a row
                if s1_id not in written_s1:
                    out_f.write(f"{s1_id}\t\n")
        os.replace(cand_tmp, self.rm.output_dir / "candidate_pairs.tsv")
        del written_s1
        scored = pd.concat([pd.read_parquet(part_paths(c)[0]) for c in chunk_paths], ignore_index=True) \
            if chunk_paths else pd.DataFrame(columns=["s1_id", "candidate_id", "s1_country", "p1", "prob"])
        scored.to_parquet(ckpt_test_scored, index=False)
        print(f"  Kept {len(scored):,} scored pairs with prob >= {keep_floor:.2f} for the decision layer "
              f"[{ckpt_test_scored.name}]")

        prog.step("4", f"Decision layer + writing submission files ({len(scored):,} scored pairs)")
        # Every kept pair was a candidate, so matches are always a subset of candidate_pairs.tsv
        s1_match_map = decide_matches(scored[["s1_id", "candidate_id", "prob"]], t1, t2,
                                      max_matches_per_s1=self.config.MAX_MATCHES_PER_S1,
                                      exclusive_targets=self.config.EXCLUSIVE_TARGETS)

        out_matched_ids = []
        singletons_count = matched_s1_count = total_links = 0
        for s1_id in all_test_s1_ids:
            m_valid = s1_match_map.get(s1_id, [])
            out_matched_ids.append(",".join(m_valid))
            total_links += len(m_valid)
            if m_valid:
                matched_s1_count += 1
            else:
                singletons_count += 1

        df_matching = pd.DataFrame({"source1_entity_id": all_test_s1_ids, "matched_entity_ids": out_matched_ids})

        matching_tsv = self.rm.output_dir / "matching_results.tsv"
        candidate_tsv = self.rm.output_dir / "candidate_pairs.tsv"
        df_matching.to_csv(matching_tsv, sep="\t", index=False, encoding="utf-8")
        df_matching.to_csv(self.rm.output_dir / "matching_results.csv", sep=",", index=False, encoding="utf-8")
        print(f"  Saved TSV: {matching_tsv}")
        print(f"  Saved TSV: {candidate_tsv}")

        n_total = len(all_test_s1_ids)
        print("\nSubmission Summary:")
        print(f"  Total S1 entities        : {n_total:,}")
        print(f"  Entities with >= 1 match : {matched_s1_count:,} ({matched_s1_count / max(n_total, 1):.2%})")
        print(f"  Singletons (0 matches)   : {singletons_count:,} ({singletons_count / max(n_total, 1):.2%})")
        print(f"  Total predicted links    : {total_links:,}")
        if len(scored):
            country_of = scored.drop_duplicates("s1_id").set_index("s1_id")["s1_country"]
            by_c = pd.DataFrame({"s1_id": all_test_s1_ids,
                                 "n": [len(m.split(",")) if m else 0 for m in out_matched_ids]})
            by_c["country"] = by_c["s1_id"].map(country_of).fillna("(no likely match)")
            summ = by_c.groupby("country").agg(s1=("s1_id", "size"), matched=("n", lambda x: (x > 0).mean()),
                                               links_per_s1=("n", "mean"))
            print(f"  Per-country:\n{summ.to_string()}")

        prog.step("5", "Running utils/validate_submission.py")
        if self.config.VALIDATOR_PATH.is_file():
            test_dir = self.rm.run_dir / "val_test_dir" if self.config.SAMPLE_SIZE_S1 else self.config.TEST_DIR
            test_dir.mkdir(parents=True, exist_ok=True)
            val_s1_file = test_dir / "test_source1.tsv"
            if not val_s1_file.is_file() or self.config.SAMPLE_SIZE_S1:
                import duckdb
                con_val = duckdb.connect()
                con_val.register("s1_out", pd.DataFrame({"entity_id": all_test_s1_ids}))
                test_s1_str = str(self.config.TEST_S1_PATH).replace("\\", "/")
                val_s1_str = str(val_s1_file).replace("\\", "/")
                con_val.execute(f"""
                    COPY (
                        SELECT s1.entity_id, s1.business_name, s1.business_address, s1.country
                        FROM '{test_s1_str}' s1
                        JOIN s1_out ON s1.entity_id = s1_out.entity_id
                    ) TO '{val_s1_str}' (HEADER, DELIMITER '\t');
                """)
            cmd = [sys.executable, str(self.config.VALIDATOR_PATH),
                   "--matching", str(matching_tsv), "--candidate", str(candidate_tsv),
                   "--test-dir", str(test_dir)]
            res = subprocess.run(cmd, capture_output=True, text=True)
            print(res.stdout)
            if res.stderr:
                print(res.stderr)

        prog.done()
        elapsed = time.time() - t_start
        print(f"\nTest Inference Phase Complete in {elapsed:.2f}s!")
        return {
            "total_s1": n_total,
            "matched_s1": matched_s1_count,
            "singletons": singletons_count,
            "total_links": total_links,
            "elapsed_seconds": elapsed,
        }

    def run_decide(self, t1: float, t2: float) -> Dict[str, Any]:
        """
        Re-apply the decision layer to the saved test scores (checkpoints/test_scored.parquet) with new
        thresholds - seconds instead of re-scoring. Writes output/decide_t1_<t1>_t2_<t2>/ with
        matching_results.tsv + candidate_pairs.tsv (copied) and runs the validator on them.
        The saved scores only contain pairs with prob >= the kept floor, so t1, t2 must be >= that floor.
        """
        import shutil
        scored_path = self.rm.checkpoints_dir / "test_scored.parquet"
        cand_path = self.rm.output_dir / "candidate_pairs.tsv"
        if not scored_path.is_file() or not cand_path.is_file():
            raise FileNotFoundError(f"{scored_path} / {cand_path} missing: run --mode test once first.")
        scored = pd.read_parquet(scored_path, columns=["s1_id", "candidate_id", "s1_country", "prob"])
        floor = float(scored["prob"].min()) if len(scored) else 0.0
        if min(t1, t2) < floor - 1e-4:
            raise ValueError(f"Saved test scores only keep prob >= {floor:.3f}; choose thresholds >= that "
                             f"(or rerun --mode test with lower thresholds in metrics.json).")
        all_ids = get_all_s1_ids(self.config.TEST_S1_PATH, limit=self.config.SAMPLE_SIZE_S1)
        match = decide_matches(scored[["s1_id", "candidate_id", "prob"]], t1, t2,
                               max_matches_per_s1=self.config.MAX_MATCHES_PER_S1,
                               exclusive_targets=self.config.EXCLUSIVE_TARGETS)
        out_dir = self.rm.output_dir / f"decide_t1_{t1:.2f}_t2_{t2:.2f}"
        out_dir.mkdir(parents=True, exist_ok=True)
        rows = [",".join(match.get(i, [])) for i in all_ids]
        pd.DataFrame({"source1_entity_id": all_ids, "matched_entity_ids": rows}).to_csv(
            out_dir / "matching_results.tsv", sep="\t", index=False, encoding="utf-8")
        shutil.copy2(cand_path, out_dir / "candidate_pairs.tsv")
        n_links = sum(len(r.split(",")) for r in rows if r)
        n_matched = sum(1 for r in rows if r)
        country = scored.drop_duplicates("s1_id").set_index("s1_id")["s1_country"]
        k = pd.Series([len(r.split(",")) if r else 0 for r in rows], index=all_ids)
        by_c = pd.DataFrame({"k": k.values, "country": k.index.map(country).fillna("(none)")})
        print(f"\n[DECIDE] t1={t1:.2f} t2={t2:.2f}: {n_matched:,}/{len(all_ids):,} S1 matched, {n_links:,} links")
        print(by_c.groupby("country")["k"].agg(links_per_s1="mean", matched=lambda x: (x > 0).mean()).to_string())
        if self.config.VALIDATOR_PATH.is_file() and not self.config.SAMPLE_SIZE_S1:
            res = subprocess.run([sys.executable, str(self.config.VALIDATOR_PATH), "--matching",
                                  str(out_dir / "matching_results.tsv"), "--candidate",
                                  str(out_dir / "candidate_pairs.tsv"), "--test-dir", str(self.config.TEST_DIR)],
                                 capture_output=True, text=True)
            print(res.stdout[-600:])
        print(f"  Submission file: {out_dir / 'matching_results.tsv'}")
        return {"t1": t1, "t2": t2, "links": n_links, "matched_s1": n_matched, "dir": str(out_dir)}

    def run_all(
        self,
        force_recompute: bool = False,
        tune_hyperparams: bool = False,
        n_trials: int = 20,
    ) -> Dict[str, Any]:
        """Run Train -> Validation (Tuning) -> Test Inference end-to-end in one go."""
        t_all_start = time.time()
        print("\n" + "#" * 70)
        print(f"RUNNING COMPLETE END-TO-END PIPELINE: {self.rm.run_name}")
        print("#" * 70)

        train_res = self.run_train(force_recompute=force_recompute, tune_hyperparams=tune_hyperparams,
                                   n_trials=n_trials)
        val_res = self.run_val(model=train_res.get("model"), fe=train_res.get("fe"),
                               force_recompute=force_recompute, model2=train_res.get("model2"))
        test_res = self.run_test(model=train_res.get("model"), fe=train_res.get("fe"),
                                 force_recompute=force_recompute, model2=train_res.get("model2"))

        total_elapsed = time.time() - t_all_start
        n_pairs = max(train_res["train_pairs_count"], 1)
        n_test = max(test_res["total_s1"], 1)
        summary_text = f"""
================================================================================
RUN SUMMARY: {self.rm.run_name}
================================================================================
Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}
Total Pipeline Time: {total_elapsed:.2f} seconds

1. TRAINING RESULTS:
   - S1 Training Entities: {train_res['train_s1_count']:,}
   - Training Pairs (sampled): {train_res['train_pairs_count']:,}
   - Positive Matches: {train_res['positives']:,} ({train_res['positives'] / n_pairs:.2%})
   - Negatives: {train_res['negatives']:,}

2. VALIDATION & TUNING RESULTS (70-30 Entity Split):
   - Blocking Recall Ceiling: {val_res.get('candidate_recall', 0.0):.4f} ({val_res.get('avg_candidates_per_s1', 0.0):.1f} cands / S1)
   - Thresholds: top-1 t1={val_res.get('threshold_top1', 0.0):.2f}, others t2={val_res.get('threshold_rest', 0.0):.2f}
   - Best Macro F_0.5 Score: {val_res['best_macro_f05']:.4f}
   - Overall Precision: {val_res['precision']:.4f}
   - Overall Recall: {val_res['recall']:.4f}
   - Singleton Accuracy: {val_res['singleton_accuracy']:.2%}

3. TEST INFERENCE RESULTS:
   - Total Test S1 Entities: {test_res['total_s1']:,}
   - S1 Entities with Match: {test_res['matched_s1']:,} ({test_res['matched_s1'] / n_test:.2%})
   - Predicted Singletons: {test_res['singletons']:,} ({test_res['singletons'] / n_test:.2%})
   - Total Matched Links: {test_res['total_links']:,}

OUTPUT FILES SAVED IN: {self.rm.output_dir}
   - matching_results.tsv
   - candidate_pairs.tsv
================================================================================
"""
        print(summary_text)
        self.rm.save_summary_text(summary_text)
        return {"train": train_res, "val": val_res, "test": test_res, "total_elapsed": total_elapsed}
