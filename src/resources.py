"""
Resource limits - one place that decides how many CPU threads/processes the pipeline may use.

run_pipeline.py sets ER_CPU_THREADS (from --cpu-threads) before any heavy library is imported,
so NumPy/BLAS/OpenMP pools, DuckDB, RapidFuzz, sparse_dot_topn, joblib workers, LightGBM and
XGBoost all respect the same cap. Heavy numeric work runs on the GPU when available.
"""

import os

DEFAULT_CPU_THREADS = 4


def cpu_threads() -> int:
    try:
        n = int(os.environ.get("ER_CPU_THREADS", DEFAULT_CPU_THREADS))
    except ValueError:
        n = DEFAULT_CPU_THREADS
    return max(1, min(n, os.cpu_count() or n))


def apply_cpu_limit(n: int) -> None:
    """Must run before NumPy / SciPy / sklearn are imported to limit their native thread pools."""
    n = str(max(1, int(n)))
    os.environ["ER_CPU_THREADS"] = n
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "RAYON_NUM_THREADS", "LOKY_MAX_CPU_COUNT"):
        os.environ[var] = n
