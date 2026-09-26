"""
Optuna Hyperparameter Tuning Engine for Entity Resolution.

Tunes the stage-1 XGBoost model on CACHED feature checkpoints (no re-blocking / re-extraction):
  * each trial trains on a random subset of TRAIN S1 groups (TUNE_TRAIN_FRAC) on the GPU,
    with early stopping on validation logloss (so n_estimators is found, not guessed),
  * the objective is validation Macro F0.5 after the same (t1, t2) threshold sweep + target
    exclusivity used by the real validation phase.
The final model is then retrained by the pipeline on ALL training pairs (OOF + stage 2 + stage 1)
with n_estimators scaled up for the larger training set.
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import optuna
import pandas as pd

from src.metrics import sweep_thresholds_2d
from src.resources import cpu_threads


class EntityResolutionTuner:
    """Optuna TPE search over XGBoost parameters maximizing validation Macro F0.5."""

    def __init__(
        self,
        model_type: str = "xgboost",
        device: str = "cuda",
        n_trials: int = 25,
        timeout: Optional[int] = 3600,
        random_seed: int = 42,
        train_frac: float = 0.4,
    ):
        self.model_type = model_type.lower()
        self.device = device.lower()
        self.n_trials = n_trials
        self.timeout = timeout
        self.random_seed = random_seed
        self.train_frac = train_frac
        self.study: Optional[optuna.Study] = None
        self.best_params: Dict[str, Any] = {}
        self.best_f05: float = 0.0

    def tune(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        val_pairs_df: pd.DataFrame,
        ground_truth_map: Dict[str, set],
        val_s1_ids: List[str],
        output_dir: Optional[Path] = None,
        train_groups: Optional[np.ndarray] = None,
    ) -> Tuple[Dict[str, Any], float, float]:
        import xgboost as xgb

        if self.model_type not in ("xgboost", "ensemble"):
            raise ValueError("Optuna tuning is implemented for XGBoost (use --model-type xgboost)")

        rng = np.random.default_rng(self.random_seed)
        if train_groups is not None and self.train_frac < 1.0:
            uniq = np.unique(train_groups)
            keep_groups = set(rng.choice(uniq, size=max(1, int(len(uniq) * self.train_frac)), replace=False))
            sel = np.fromiter((g in keep_groups for g in train_groups), dtype=bool, count=len(train_groups))
        else:
            sel = np.ones(len(X_train), dtype=bool)
        Xt = np.ascontiguousarray(X_train.values[sel], dtype=np.float32)
        yt = np.ascontiguousarray(y_train.values[sel], dtype=np.int32)
        Xv = np.ascontiguousarray(X_val[X_train.columns].values, dtype=np.float32)
        yv = np.fromiter((c in ground_truth_map.get(s, ()) for s, c in
                          zip(val_pairs_df["s1_id"].values, val_pairs_df["candidate_id"].values)),
                         dtype=np.int32, count=len(val_pairs_df))
        print(f"\n[Optuna] {self.n_trials} trials (timeout {self.timeout}s) | trial train rows {len(Xt):,} "
              f"({self.train_frac:.0%} of train S1) | val rows {len(Xv):,} | objective: val Macro F0.5")
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial: optuna.Trial) -> float:
            t0 = time.time()
            params = {
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "tree_method": "hist",
                "device": self.device,
                "n_jobs": cpu_threads(),
                "random_state": self.random_seed,
                "n_estimators": 4000,
                "early_stopping_rounds": 150,
                "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.12, log=True),
                "max_depth": trial.suggest_int("max_depth", 6, 12),
                "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 20.0, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
                "colsample_bynode": trial.suggest_float("colsample_bynode", 0.5, 1.0),
                "gamma": trial.suggest_float("gamma", 1e-3, 5.0, log=True),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 20.0, log=True),
                "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 4.0),
                "max_bin": trial.suggest_categorical("max_bin", [256, 512]),
            }
            clf = xgb.XGBClassifier(**params)
            clf.fit(Xt, yt, eval_set=[(Xv, yv)], verbose=False)
            best_it = int(clf.best_iteration) + 1
            booster = clf.get_booster()
            try:
                import cupy as cp
                booster.set_param({"device": self.device})
                probs = cp.asnumpy(booster.inplace_predict(cp.asarray(Xv), iteration_range=(0, best_it)))
            except Exception:
                probs = booster.inplace_predict(Xv, iteration_range=(0, best_it))
            (t1, t2), sw = sweep_thresholds_2d(val_pairs_df, probs, ground_truth_map, val_s1_ids,
                                               min_thresh=0.1, max_thresh=0.96, step=0.04)
            b = sw.iloc[0]
            trial.set_user_attr("best_iteration", best_it)
            trial.set_user_attr("t1", t1)
            trial.set_user_attr("t2", t2)
            trial.set_user_attr("precision", float(b["precision"]))
            trial.set_user_attr("recall", float(b["recall"]))
            print(f"  [Trial {trial.number:02d}] F0.5 {b['macro_f05']:.4f} | P {b['precision']:.4f} R {b['recall']:.4f} "
                  f"| trees {best_it} depth {params['max_depth']} lr {params['learning_rate']:.3f} "
                  f"| {time.time() - t0:.0f}s", flush=True)
            return float(b["macro_f05"])

        sampler = optuna.samplers.TPESampler(seed=self.random_seed, n_startup_trials=min(8, self.n_trials))
        self.study = optuna.create_study(direction="maximize", sampler=sampler)
        # Seed the search with the current default configuration
        self.study.enqueue_trial({"learning_rate": 0.03, "max_depth": 8, "min_child_weight": 2.0, "subsample": 0.8,
                                  "colsample_bytree": 0.6, "colsample_bynode": 1.0, "gamma": 1e-3,
                                  "reg_alpha": 1e-3, "reg_lambda": 1.0, "scale_pos_weight": 2.0, "max_bin": 256})
        self.study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout)

        best = self.study.best_trial
        # More training rows in the final fit -> allow proportionally more boosting rounds
        n_final = int(best.user_attrs["best_iteration"] * (1.0 + 0.5 * (1.0 / max(self.train_frac, 1e-3) - 1.0) ** 0.5))
        self.best_params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "tree_method": "hist",
            "device": self.device,
            "random_state": self.random_seed,
            "n_estimators": n_final,
            **best.params,
        }
        self.best_f05 = float(best.value)

        print("\n" + "=" * 60)
        print("OPTUNA TUNING COMPLETE")
        print(f"  Best trial #{best.number}: Macro F0.5 {self.best_f05:.4f} "
              f"(P {best.user_attrs['precision']:.4f}, R {best.user_attrs['recall']:.4f}, "
              f"{best.user_attrs['best_iteration']} trees on {self.train_frac:.0%} data -> {n_final} for full data)")
        print(f"  Params: {json.dumps(self.best_params, indent=2)}")
        print("=" * 60)

        if output_dir:
            out_p = Path(output_dir)
            out_p.mkdir(parents=True, exist_ok=True)
            self.study.trials_dataframe().to_csv(out_p / "optuna_trials.csv", index=False)
            with open(out_p / "best_params.json", "w", encoding="utf-8") as f:
                json.dump({"best_macro_f05_stage1": self.best_f05, "best_params": self.best_params,
                           "trial_attrs": best.user_attrs}, f, indent=2)
            print(f"  Saved Optuna results to {out_p}")
        return self.best_params, self.best_f05, float(best.user_attrs.get("t2", 0.5))
