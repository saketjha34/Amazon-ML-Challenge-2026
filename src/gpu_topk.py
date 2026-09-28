"""
GPU top-K sparse cosine retrieval (CuPy / cuSPARSE SpGEMM).

For each query row of Q (L2-normalised TF-IDF), return the top_n target rows of T by cosine
(dot product), keeping only values > threshold - the same contract as
sparse_dot_topn.sp_matmul_topn.

Memory safety (6 GB laptop GPUs):
  * Batches are planned up front: for every query row we compute an upper bound of the SpGEMM
    work (sum over its n-grams of how many targets contain that n-gram) and pack rows so each
    batch stays under a budget derived from the currently free GPU memory.
  * Any CUDA allocation failure (OutOfMemoryError, cudaErrorMemoryAllocation, driver OOM) splits
    the batch in half and retries; if a single row still does not fit, MemoryError is raised and
    the caller falls back to the CPU implementation for that field.
"""

import warnings

import numpy as np
import scipy.sparse as sp

try:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="CUDA path could not be detected")
        import cupy as cp
        import cupyx.scipy.sparse as cps
    GPU_AVAILABLE = cp.cuda.runtime.getDeviceCount() > 0
    _CUDA_ERRORS = (cp.cuda.memory.OutOfMemoryError, cp.cuda.runtime.CUDARuntimeError,
                    cp.cuda.driver.CUDADriverError, MemoryError)
except Exception:  # pragma: no cover - no CUDA / CuPy
    cp = None
    cps = None
    GPU_AVAILABLE = False
    _CUDA_ERRORS = (MemoryError,)

MAX_BATCH_ROWS = 1024
MAX_OOM_RETRIES = 4             # after this many GPU failures in one call -> MemoryError -> CPU fallback
_BUDGET_SCALE = [1.0]           # adaptive: halved after every GPU failure, recovers slowly on success
BYTES_PER_PRODUCT = 24          # conservative: SpGEMM intermediates + output (value, index, row)
MAX_WORK_PER_BATCH = 150_000_000


def _free_pool():
    try:
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass


def free_gpu() -> None:
    """Release every cached CuPy block (call after each chunk so the next one starts clean)."""
    if cp is not None:
        _free_pool()


class GpuGuard:
    """
    Per-index GPU health: after `max_failures` out-of-memory fallbacks, the index is moved off the GPU
    for the rest of the pass (its device memory released) instead of failing again on every chunk.
    """

    def __init__(self, max_failures: int = 2):
        self.max_failures = max_failures
        self.failures = {}

    def failed(self, key, gmats: dict) -> None:
        self.failures[key] = self.failures.get(key, 0) + 1
        if self.failures[key] >= self.max_failures and key in gmats:
            g = gmats.pop(key)
            try:
                g.free()
            except Exception:
                pass
            _free_pool()
            print(f"      [GPU] '{key}' moved to CPU for the rest of this pass (GPU memory is too tight)")


def _work_budget() -> int:
    """Max SpGEMM products per batch, from currently free GPU memory."""
    try:
        _free_pool()
        free, _ = cp.cuda.runtime.memGetInfo()
        return int(max(500_000, min(MAX_WORK_PER_BATCH,
                                    0.25 * free * _BUDGET_SCALE[0] / BYTES_PER_PRODUCT)))
    except Exception:
        return 10_000_000


class GpuTargetMatrix:
    """Target matrix transposed (n_features x n_targets) and resident on the GPU."""

    def __init__(self, T: sp.csr_matrix):
        self.n_targets = T.shape[0]
        TT = T.T.tocsr().astype(np.float32)
        TT.sort_indices()                      # cuSPARSE SpGEMM requires sorted column indices
        self.feat_df = np.diff(TT.indptr).astype(np.int64)   # targets containing each n-gram (CPU)
        self.TT = cps.csr_matrix(TT)

    def free(self):
        self.TT = None
        _free_pool()


def _plan_batches(Q: sp.csr_matrix, feat_df: np.ndarray, budget: int):
    """Pack consecutive query rows into batches whose estimated SpGEMM work <= budget."""
    per_nz = feat_df[Q.indices]
    cums = np.concatenate([[0], np.cumsum(per_nz)])
    work = cums[Q.indptr[1:]] - cums[Q.indptr[:-1]]          # per-row upper bound
    cum = np.cumsum(work)
    n = Q.shape[0]
    bounds, start = [], 0
    while start < n:
        base = cum[start - 1] if start else 0
        end = int(np.searchsorted(cum, base + budget, side="right"))
        end = min(max(end, start + 1), start + MAX_BATCH_ROWS, n)
        bounds.append((start, end))
        start = end
    return bounds


def _topn_batch(Qb: sp.csr_matrix, G: GpuTargetMatrix, top_n: int, threshold: float, r0: int):
    Qg = cps.csr_matrix(Qb)
    C = (Qg @ G.TT).tocoo()
    del Qg
    keep = C.data > threshold
    r, c, v = C.row[keep], C.col[keep], C.data[keep]
    del C, keep
    if r.size == 0:
        return None
    order = cp.lexsort(cp.stack([-v, r]))        # by row, then value desc
    r, c, v = r[order], c[order], v[order]
    first = cp.searchsorted(r, r, side="left")
    sel = (cp.arange(r.size, dtype=first.dtype) - first) < top_n
    out = (cp.asnumpy(r[sel]).astype(np.int64) + r0,
           cp.asnumpy(c[sel]).astype(np.int64),
           cp.asnumpy(v[sel]).astype(np.float32))
    del r, c, v, order, first, sel
    return out


def gpu_topn(Q: sp.csr_matrix, G: GpuTargetMatrix, top_n: int, threshold: float, **_):
    """
    Returns (rows, cols, vals) numpy arrays (like a COO result), rows relative to Q.
    Raises MemoryError if even a single query row cannot be processed on the GPU.
    """
    Q = Q.tocsr().astype(np.float32)
    Q.sort_indices()
    rows_out, cols_out, vals_out = [], [], []
    stack = list(reversed(_plan_batches(Q, G.feat_df, _work_budget())))
    fails = 0
    while stack:
        a, b = stack.pop()
        try:
            res = _topn_batch(Q[a:b], G, top_n, threshold, a)
        except Exception:  # any CUDA / cuSPARSE allocation failure: split and retry
            _free_pool()
            fails += 1
            _BUDGET_SCALE[0] = max(0.05, _BUDGET_SCALE[0] * 0.5)   # plan smaller batches from now on
            if b - a <= 1 or fails > MAX_OOM_RETRIES:
                raise MemoryError("GPU top-K: out of GPU memory, falling back to CPU for this chunk")
            mid = (a + b) // 2
            stack.extend([(mid, b), (a, mid)])   # retry both halves (a..mid first)
            continue
        if res is not None:
            rows_out.append(res[0])
            cols_out.append(res[1])
            vals_out.append(res[2])
    _free_pool()
    if not fails:
        _BUDGET_SCALE[0] = min(1.0, _BUDGET_SCALE[0] * 1.1)
    if not rows_out:
        e = np.zeros(0, dtype=np.int64)
        return e, e, np.zeros(0, dtype=np.float32)
    return np.concatenate(rows_out), np.concatenate(cols_out), np.concatenate(vals_out)
