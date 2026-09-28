"""
Run Manager - Handles automatic versioning for run_1, run_2, run_3, etc.
"""

import json
import re
import shutil
from pathlib import Path
from typing import Optional, Dict, Any


class RunManager:
    """
    Manages experiment runs, creating dedicated directories:
      runs/run_1/
      runs/run_2/
      runs/run_3/
    """
    def __init__(self, runs_dir: Path, custom_run_name: Optional[str] = None):
        self.runs_dir = Path(runs_dir)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.custom_run_name = custom_run_name

        self.run_dir = self._create_run_directory()
        self.output_dir = self.run_dir / "output"
        self.models_dir = self.run_dir / "models"
        self.metrics_dir = self.run_dir / "metrics"
        self.features_dir = self.run_dir / "features"
        self.checkpoints_dir = self.run_dir / "checkpoints"

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.features_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    def _create_run_directory(self) -> Path:
        if self.custom_run_name:
            target = self.runs_dir / self.custom_run_name
            target.mkdir(parents=True, exist_ok=True)
            return target

        # Scan existing run_N folders
        existing_runs = []
        for item in self.runs_dir.iterdir():
            if item.is_dir():
                match = re.match(r"^run_(\d+)$", item.name)
                if match:
                    existing_runs.append(int(match.group(1)))

        next_idx = max(existing_runs, default=0) + 1
        new_dir = self.runs_dir / f"run_{next_idx}"
        new_dir.mkdir(parents=True, exist_ok=True)
        return new_dir

    @property
    def run_name(self) -> str:
        return self.run_dir.name

    def save_config(self, config_dict: Dict[str, Any]):
        """Save a snapshot of the run configuration."""
        clean_dict = {}
        for k, v in config_dict.items():
            if isinstance(v, Path):
                clean_dict[k] = str(v)
            else:
                try:
                    json.dumps(v)
                    clean_dict[k] = v
                except (TypeError, OverflowError):
                    clean_dict[k] = str(v)

        config_path = self.run_dir / "config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(clean_dict, f, indent=2)

    def save_metrics(self, metrics_dict: Dict[str, Any]):
        """Save evaluation and run metrics."""
        metrics_path = self.metrics_dir / "metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics_dict, f, indent=2)

    def save_summary_text(self, text: str):
        """Save human-readable run summary text."""
        summary_path = self.run_dir / "run_summary.txt"
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(text)

    # Artefacts that do not depend on the matching model: blocking output, pairwise / context
    # features and the test blocking checkpoints. Reusing them skips hours of blocking.
    SEED_ITEMS = [
        "train_s1_ids.npy", "val_s1_ids.npy",
        "models/block_ranker.joblib", "models/feature_extractor.joblib",
        "checkpoints/candidates_train.parquet", "checkpoints/val_pairs.parquet",
        "features/train_features.parquet", "features/val_features.parquet",
        "features/train_context.parquet", "features/val_context.parquet",
        "checkpoints/test/blocking",
    ]

    def seed_from(self, other_run: str) -> None:
        """Hard-link (or copy) reusable artefacts of runs/<other_run> into this run (missing ones only)."""
        import os
        src_root = self.runs_dir / other_run
        if not src_root.is_dir():
            raise FileNotFoundError(f"--init-from: {src_root} does not exist")
        if src_root.resolve() == self.run_dir.resolve():
            return

        def link(src: Path, dst: Path) -> bool:
            if dst.exists():
                return False
            dst.parent.mkdir(parents=True, exist_ok=True)
            # Only large write-once data files are hard-linked (instant, no extra disk). Everything that
            # can be rewritten in place (models, ids, signature.json, _DONE) is copied: rewriting a hard
            # link would silently change the source run too. copy2 keeps the mtime, which the blocking
            # checkpoint fingerprint uses for the ranker file.
            if src.suffix in (".parquet", ".npz"):
                try:
                    os.link(src, dst)
                    return True
                except OSError:
                    pass
            shutil.copy2(src, dst)
            return True

        n = 0
        for item in self.SEED_ITEMS:
            src = src_root / item
            if src.is_file():
                n += link(src, self.run_dir / item)
            elif src.is_dir():
                for f in src.rglob("*"):
                    if f.is_file():
                        n += link(f, self.run_dir / item / f.relative_to(src))
        print(f"[init-from] Reused {n} artefact files from {src_root.name} (blocking, features, test blocking)")
