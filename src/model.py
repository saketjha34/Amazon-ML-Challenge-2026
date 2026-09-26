"""
Machine Learning Model - GPU-Accelerated Classifier for Entity Resolution.
Supports XGBoost (GPU histogram via CUDA) and LightGBM.
Implements Section 13 of the Business Entity Resolution Architecture.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import joblib
import numpy as np
import pandas as pd

# Check CUDA availability
try:
    import xgboost as _test_xgb
    _test_clf = _test_xgb.XGBClassifier(tree_method="hist", device="cuda", n_estimators=1)
    _test_clf.fit(np.zeros((2, 2), dtype=np.float32), np.array([0, 1]))
    CUDA_AVAILABLE = True
except Exception:
    CUDA_AVAILABLE = False


class EntityMatcherModel:
    """
    Binary Classifier predicting P(same business) for candidate pairs.
    Supports GPU-accelerated XGBoost and CPU/GPU LightGBM.
    """
    def __init__(
        self,
        model_type: str = "xgboost",
        device: str = "cuda" if CUDA_AVAILABLE else "cpu",
        params: Optional[Dict[str, Any]] = None,
    ):
        self.model_type = model_type.lower()
        self.device = device.lower()
        self.params = params or self._default_params()
        self.feature_names: List[str] = []
        self.is_trained = False
        self.model = self._init_model()

    def _default_params(self) -> Dict[str, Any]:
        if self.model_type == "xgboost":
            return {
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "tree_method": "hist",
                "device": self.device,
                "n_estimators": 400,
                "learning_rate": 0.05,
                "max_depth": 7,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "scale_pos_weight": 2.0,
                "random_state": 42,
            }
        else:
            return {
                "objective": "binary",
                "metric": "binary_logloss",
                "boosting_type": "gbdt",
                "n_estimators": 350,
                "learning_rate": 0.05,
                "num_leaves": 63,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "min_child_samples": 20,
                "scale_pos_weight": 2.0,
                "random_state": 42,
                "n_jobs": -1,
                "verbose": -1,
            }

    @staticmethod
    def _with_threads(params: Dict[str, Any], key: str) -> Dict[str, Any]:
        from src.resources import cpu_threads
        p = dict(params)
        if not p.get(key) or p.get(key) == -1:
            p[key] = cpu_threads()
        return p

    def _init_model(self):
        if self.model_type == "xgboost":
            import xgboost as xgb
            return xgb.XGBClassifier(**self._with_threads(self.params, "n_jobs"))
        elif self.model_type == "lightgbm":
            from lightgbm import LGBMClassifier
            return LGBMClassifier(**self._with_threads(self.params, "n_jobs"))
        elif self.model_type == "ensemble":
            import xgboost as xgb
            from lightgbm import LGBMClassifier
            xgb_p = self.params.get("xgb_params", {
                "objective": "binary:logistic", "eval_metric": "logloss",
                "tree_method": "hist", "device": self.device,
                "n_estimators": 400, "learning_rate": 0.05, "max_depth": 7,
                "scale_pos_weight": 2.0, "random_state": 42,
            })
            lgb_p = self.params.get("lgb_params", {
                "objective": "binary", "metric": "binary_logloss",
                "boosting_type": "gbdt", "n_estimators": 350, "learning_rate": 0.05,
                "num_leaves": 63, "scale_pos_weight": 2.0, "random_state": 42,
                "n_jobs": -1, "verbose": -1,
            })
            return {
                "xgb": xgb.XGBClassifier(**self._with_threads(xgb_p, "n_jobs")),
                "lgb": LGBMClassifier(**self._with_threads(lgb_p, "n_jobs")),
            }
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

    def train(self, X_train: pd.DataFrame, y_train: pd.Series):
        """Train classifier on pairwise features."""
        self.feature_names = list(X_train.columns)
        X_mat = np.ascontiguousarray(X_train.values, dtype=np.float32)
        y_vec = np.ascontiguousarray(y_train.values, dtype=np.int32)

        if self.model_type == "ensemble":
            from concurrent.futures import ThreadPoolExecutor
            print(f"    [Model] Training ENSEMBLE concurrently (XGBoost on GPU + LightGBM on CPU) on "
                  f"{len(X_train):,} pairs with {len(X_train.columns)} features...")
            # Both libraries release the GIL while fitting, so GPU and CPU train at the same time
            with ThreadPoolExecutor(max_workers=2) as ex:
                f_xgb = ex.submit(self.model["xgb"].fit, X_mat, y_vec)
                f_lgb = ex.submit(self.model["lgb"].fit, X_train, y_train)
                f_xgb.result()
                f_lgb.result()
            print("    [Model] ENSEMBLE training complete.")
        else:
            backend_info = f"XGBoost on GPU ({self.device})" if self.model_type == "xgboost" and self.device == "cuda" else f"{self.model_type.upper()} on CPU"
            print(f"    [Model] Training {backend_info} on {len(X_train):,} pairs with {len(X_train.columns)} features...")
            self.model.fit(X_mat, y_vec)
            print(f"    [Model] {self.model_type.upper()} training complete.")

        self.is_trained = True

    def predict_proba(self, X: pd.DataFrame, batch_size: int = 200_000) -> np.ndarray:
        """Predict match probabilities P(same business) in fast batches."""
        if not self.is_trained:
            raise RuntimeError("Model is not trained. Call train() or load() first.")

        if len(X) == 0:
            return np.array([], dtype=np.float32)

        if list(X.columns) != self.feature_names:
            # Align to training columns; inputs dropped since training (e.g. a disabled rule) -> 0
            X = X.reindex(columns=self.feature_names, fill_value=0.0)
        if self.model_type == "xgboost":
            return self._predict_xgb(self.model, X, batch_size)
        elif self.model_type == "lightgbm":
            probs = []
            for start in range(0, len(X), batch_size):
                end = min(start + batch_size, len(X))
                batch = X.iloc[start:end]
                p = self.model.predict_proba(batch)[:, 1]
                probs.append(p)
            return np.concatenate(probs).astype(np.float32)
        elif self.model_type == "ensemble":
            # Blend 65% XGBoost (GPU) + 35% LightGBM (CPU); both halves run concurrently
            from concurrent.futures import ThreadPoolExecutor
            X = X[self.feature_names] if list(X.columns) != self.feature_names else X
            with ThreadPoolExecutor(max_workers=2) as ex:
                f_xgb = ex.submit(self._predict_xgb, self.model["xgb"], X, batch_size)
                f_lgb = ex.submit(lambda: np.concatenate([
                    self.model["lgb"].predict_proba(X.iloc[s:s + batch_size])[:, 1]
                    for s in range(0, len(X), batch_size)]).astype(np.float32))
                p_xgb, p_lgb = f_xgb.result(), f_lgb.result()
            return (0.65 * p_xgb + 0.35 * p_lgb).astype(np.float32)

    def _predict_xgb(self, clf, X: pd.DataFrame, batch_size: int) -> np.ndarray:
        """XGBoost prediction; in-place on the GPU (CuPy) when available, DMatrix on CPU otherwise."""
        import xgboost as xgb
        booster = clf.get_booster()
        cp = None
        if self.device == "cuda" and CUDA_AVAILABLE:
            try:
                import cupy as cp
                booster.set_param({"device": "cuda"})
            except Exception:
                cp = None
        out = []
        for s in range(0, len(X), batch_size):
            vals = np.ascontiguousarray(X.iloc[s:s + batch_size].values, dtype=np.float32)
            if cp is not None:
                out.append(cp.asnumpy(booster.inplace_predict(cp.asarray(vals))))
            else:
                out.append(booster.predict(xgb.DMatrix(vals, feature_names=self.feature_names)))
        return np.concatenate(out).astype(np.float32)

    def get_feature_importance(self) -> pd.DataFrame:
        """Return feature importance table sorted descending."""
        if not self.is_trained:
            return pd.DataFrame(columns=["feature", "importance"])

        if self.model_type == "ensemble":
            imp_xgb = self.model["xgb"].feature_importances_
            imp_xgb = imp_xgb / max(imp_xgb.sum(), 1e-6)
            imp_lgb = self.model["lgb"].feature_importances_
            imp_lgb = imp_lgb / max(imp_lgb.sum(), 1e-6)
            importances = 0.65 * imp_xgb + 0.35 * imp_lgb
        else:
            importances = self.model.feature_importances_

        df_imp = pd.DataFrame({
            "feature": self.feature_names,
            "importance": importances,
        }).sort_values(by="importance", ascending=False).reset_index(drop=True)
        return df_imp

    def save(self, filepath: Path):
        """Save model and metadata to disk."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "model": self.model,
            "model_type": self.model_type,
            "device": self.device,
            "feature_names": self.feature_names,
            "params": self.params,
        }, filepath)
        print(f"    [Model] Saved model to {filepath}")

    @classmethod
    def load(cls, filepath: Path) -> "EntityMatcherModel":
        """Load trained model from disk."""
        data = joblib.load(filepath)
        instance = cls(
            model_type=data.get("model_type", "xgboost"),
            device=data.get("device", "cuda" if CUDA_AVAILABLE else "cpu"),
            params=data.get("params"),
        )
        instance.model = data["model"]
        instance.feature_names = data["feature_names"]
        instance.is_trained = True
        return instance


