"""
Stage-2 Stacking Features - Re-score each candidate using the other candidates of the same S1.

A Source 1 entity typically has several true matches (~3.4 on average) that are variants of the
same business, so they look like each other. A candidate that is similar to other confident
candidates of the same S1 is more likely a true match than an isolated one; a candidate that
competes with a much stronger sibling but does not resemble it is likely a different business.

Inputs are the stage-1 probabilities (out-of-fold during training, so stage 2 never sees
leaked scores) plus the candidate strings. All features are computed within each S1 group,
so every S1 group must be complete in the frame.
"""

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist


def _obj(s):
    """Plain Python-object string array (pyarrow-backed element access is ~10x slower)."""
    return np.array(s.fillna("").astype(str).tolist(), dtype=object)

TOP_K = 3

STACK_COLUMNS = (
    ["p1", "p1_rank", "p1_max", "p1_gap", "p1_second", "p1_sum", "p1_n_gt50", "p1_n_gt90"]
    + [f"sib{k}_{c}" for k in range(1, TOP_K + 1) for c in ("p1", "core_sim", "addr_sim")]
    + ["sib_support", "sib_same_addr_max_p1", "sib_same_core_max_p1"]
)


def _other_max(keys: pd.Series, p1: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """For each row: max p1 among OTHER rows sharing the same key (0 if none)."""
    out = np.zeros(len(p1), np.float32)
    idx = np.flatnonzero(valid)
    if len(idx) == 0:
        return out
    codes = pd.factorize(keys.values[idx])[0]
    p = p1[idx]
    order = np.lexsort((-p, codes))
    c_o, p_o = codes[order], p[order]
    m = len(c_o)
    first = np.r_[True, c_o[1:] != c_o[:-1]]
    starts = np.flatnonzero(first)
    sizes = np.diff(np.r_[starts, m])
    g = np.repeat(np.arange(len(starts)), sizes)
    best = p_o[starts]
    second = np.where(sizes > 1, p_o[np.minimum(starts + 1, m - 1)], 0.0)
    other = np.where(first, second[g], best[g])   # best row sees the runner-up, others see the best
    out[idx[order]] = other
    return out


def stack_features(pairs: pd.DataFrame, p1: np.ndarray, workers: int = None) -> pd.DataFrame:
    """
    pairs: rows with s1_id, target_core, target_addr (row-aligned with p1).
    Returns STACK_COLUMNS, row-aligned with pairs.
    """
    n = len(pairs)
    if n == 0:
        return pd.DataFrame(columns=STACK_COLUMNS)
    if not workers:
        from src.resources import cpu_threads
        workers = cpu_threads()
    p1 = np.asarray(p1, dtype=np.float32)
    s1 = _obj(pairs["s1_id"])
    core = _obj(pairs["target_core"])
    addr = _obj(pairs["target_addr"])

    # Order rows by (s1, p1 desc); work in that order and scatter back at the end.
    order = np.lexsort((-p1, s1))
    s1_o, p_o = s1[order], p1[order]
    starts = np.flatnonzero(np.r_[True, s1_o[1:] != s1_o[:-1]])
    sizes = np.diff(np.r_[starts, n])
    gid = np.repeat(np.arange(len(starts)), sizes)
    rank = np.arange(n) - starts[gid]

    f = {}
    f["p1"] = p_o
    f["p1_rank"] = rank.astype(np.float32)
    gmax = p_o[starts]
    f["p1_max"] = gmax[gid]
    f["p1_gap"] = gmax[gid] - p_o
    second = np.where(sizes > 1, p_o[np.minimum(starts + 1, n - 1)], 0.0)
    f["p1_second"] = second[gid].astype(np.float32)
    f["p1_sum"] = np.add.reduceat(p_o, starts)[gid]
    f["p1_n_gt50"] = np.add.reduceat((p_o >= 0.5).astype(np.float32), starts)[gid]
    f["p1_n_gt90"] = np.add.reduceat((p_o >= 0.9).astype(np.float32), starts)[gid]

    # Top-(K+1) rows of each group (by p1), then per row the top-K excluding itself.
    top = np.full((len(starts), TOP_K + 1), -1, dtype=np.int64)
    for k in range(TOP_K + 1):
        has = sizes > k
        top[has, k] = starts[has] + k
    cand = top[gid]                                   # (n, K+1) positions in sorted order
    self_pos = np.arange(n)[:, None]
    keep = (cand != self_pos) & (cand >= 0)
    sel_order = np.argsort(~keep, axis=1, kind="stable")[:, :TOP_K]
    sel = np.take_along_axis(cand, sel_order, axis=1)
    sel_ok = np.take_along_axis(keep, sel_order, axis=1)

    core_o, addr_o = core[order], addr[order]
    addr_has = np.fromiter((len(a) > 0 for a in addr_o), dtype=bool, count=n)
    support = np.zeros(n, np.float32)
    for k in range(TOP_K):
        j = np.where(sel_ok[:, k], sel[:, k], 0)
        ok = sel_ok[:, k]
        pk = np.where(ok, p_o[j], 0.0).astype(np.float32)
        cs = cpdist(list(core_o), list(core_o[j]), scorer=fuzz.token_set_ratio, workers=workers,
                    dtype=np.float32) / 100.0
        a_both = addr_has & addr_has[j]
        asim = cpdist(list(addr_o), list(addr_o[j]), scorer=fuzz.token_set_ratio, workers=workers,
                      dtype=np.float32) / 100.0
        cs = np.where(ok, cs, 0.0).astype(np.float32)
        asim = np.where(ok & a_both, asim, 0.0).astype(np.float32)
        f[f"sib{k + 1}_p1"] = pk
        f[f"sib{k + 1}_core_sim"] = cs
        f[f"sib{k + 1}_addr_sim"] = asim
        # when an address is missing, name similarity alone carries the sibling evidence
        sim = np.where(a_both, 0.5 * cs + 0.5 * asim, cs)
        support += pk * sim

    f["sib_support"] = support
    s1_ser = pd.Series(s1_o)
    f["sib_same_addr_max_p1"] = _other_max(s1_ser + "\x1f" + pd.Series(addr_o), p_o, addr_o != "")
    f["sib_same_core_max_p1"] = _other_max(s1_ser + "\x1f" + pd.Series(core_o), p_o, core_o != "")

    out = {}
    for c in STACK_COLUMNS:
        col = np.empty(n, dtype=np.float32)
        col[order] = np.asarray(f[c], dtype=np.float32)   # scatter back to input row order
        out[c] = col
    return pd.DataFrame(out)
