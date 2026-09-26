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

MAX_BATCH_ROWS = 2048
BYTES_PER_PRODUCT = 24          # conservative: SpGEMM intermediates + output (value, index, row)
MAX_WORK_PER_BATCH = 150_000_000


def _free_pool():
    try:
        cp.get_default_memory_pool().free_all_blocks()
    except Exception:
        pass


def _work_budget() -> int:
    """Max SpGEMM products per batch, from currently free GPU memory."""
    try:
        _free_pool()
        free, _ = cp.cuda.runtime.memGetInfo()
        return int(max(1_000_000, min(MAX_WORK_PER_BATCH, 0.35 * free / BYTES_PER_PRODUCT)))
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
    while stack:
        a, b = stack.pop()
        try:
            res = _topn_batch(Q[a:b], G, top_n, threshold, a)
        except Exception:  # any CUDA / cuSPARSE allocation failure: split and retry
            _free_pool()
            if b - a <= 1:
                raise MemoryError("GPU top-K: a single query row does not fit in GPU memory")
            mid = (a + b) // 2
            stack.extend([(mid, b), (a, mid)])   # retry both halves (a..mid first)
            continue
        if res is not None:
            rows_out.append(res[0])
            cols_out.append(res[1])
            vals_out.append(res[2])
    _free_pool()
    if not rows_out:
        e = np.zeros(0, dtype=np.int64)
        return e, e, np.zeros(0, dtype=np.float32)
    return np.concatenate(rows_out), np.concatenate(cols_out), np.concatenate(vals_out)
