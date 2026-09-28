"""
Candidate Blocking Engine v2 - Exact-key rules + TF-IDF nearest-neighbour retrieval.

For every country present in the S1 subset (open set) and every target source (S2, S3):
  1. Exact-key rules (hash joins) on normalized keys: full name, legal-stripped core name,
     sorted-token key, phonetic skeleton, full address, name-prefix+house, first-2-words+state.
     Keys shared by more than KEY_FREQ_CAP targets are skipped (they are handled by retrieval).
  2. Top-K cosine retrieval (char 3-gram hashed TF-IDF, sparse_dot_topn) on four fields:
     core name, phonetic skeleton, canonical address, and name+address combined.
  3. The union of all pairs keeps a multi-hot rule bitmask and exact cosines for all four fields.
Both sources are then merged per S1 and capped at MAX_CANDIDATES_PER_S1 by a similarity score
(not by rule order), so neither source is starved.

Target TF-IDF matrices are cached on disk (dataset/norm/tfidf/) and reused across runs.
"""

import gc
import hashlib
import re
import json
import os
import time
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import duckdb
import numpy as np
import pandas as pd
import scipy.sparse as sp
from joblib import Parallel, delayed
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize
from sparse_dot_topn import sp_matmul_topn

from src.text_norm import ensure_norm, NORM_VERSION
from src.gpu_topk import GPU_AVAILABLE, GpuGuard, GpuTargetMatrix, free_gpu, gpu_topn
from src.progress import pbar
from src.resources import cpu_threads

RULE_NAMES = [
    "exact_name", "core_name", "name_key", "skel_key", "exact_addr", "prefix4_house",
    "cw12_state", "num_state", "skel1_state", "knn_name", "knn_skel", "knn_addr", "knn_combo",
    # Reverse retrieval: the S1 is one of the target's best S1 businesses in the WHOLE S1 file (from the
    # context rival tables). S1->target top-K misses targets with short / partial addresses or heavy
    # transliteration because thousands of other targets outrank them; target->S1 finds them.
    "rev_knn",
]
REV_MISSING_RANK = 9                  # rev_rank of a pair that is not in the target's reverse list
RULE_BIT = {r: 1 << i for i, r in enumerate(RULE_NAMES)}
EXACT_NAME_BITS = RULE_BIT["exact_name"] | RULE_BIT["core_name"] | RULE_BIT["name_key"]

# (rule, key column, min key length)
KEY_RULES = [
    ("exact_name", "name_t", 3),
    ("core_name", "name_core", 3),
    ("name_key", "name_key", 3),
    ("skel_key", "name_skel", 4),
    ("exact_addr", "addr_t", 8),
    ("prefix4_house", "k_p4h", 6),
    ("cw12_state", "k_cw12s", 8),
    # Local keys: most remaining misses have short/partial target addresses (e.g. "no 12 bangalore ka")
    # and transliterated names, so they rank in the thousands country-wide. Sharing any address
    # number (house/plot/PIN) within the same state, or the first phonetic-name token within the
    # same state, catches them; the learned ranker then decides which ones survive the cap.
    ("num_state", "k_numst", 3),      # multi-valued: one key per address number
    ("skel1_state", "k_skel1s", 6),
]
MULTI_KEY_COLS = {"k_numst"}          # space-separated key lists (exploded before joining)
KEY_COLS_VERSION = 2                  # bump when _add_key_cols changes (rebuilds cached target frames)
FIELDS = ["name", "skel", "addr", "combo"]
SRC_CODE = {"S2": 0, "S3": 1}

DEFAULT_BLOCKING_PARAMS = {
    "topk": {"name": 15, "skel": 10, "addr": 20, "combo": 15},
    "cos_floor": 0.2,
    "key_freq_cap": 200,
    # Per-field document-frequency cutoff: n-grams in more than this share of targets are dropped.
    # The phonetic skeleton has a tiny alphabet, so it needs a looser cutoff to keep any signal.
    "max_df_frac": {"name": 0.01, "skel": 0.05, "addr": 0.01, "combo": 0.01},
    "n_features": 2 ** 20,
    "query_chunk": 20_000,
    "reverse_k": 5,          # add each target's top reverse_k S1 (rival table) as candidates; 0 = off
    "n_jobs": 12,
    "use_gpu": True,   # GPU top-K retrieval (CuPy/cuSPARSE); CPU fallback on OOM
}

OUTPUT_COLUMNS = [
    "s1_id", "s1_name", "s1_core", "s1_key", "s1_skel", "s1_addr", "s1_city", "s1_state",
    "s1_house", "s1_country", "s1_native",
    "candidate_id", "candidate_source", "target_name", "target_core", "target_key", "target_skel",
    "target_addr", "target_city", "target_state", "target_house", "target_country", "target_native",
    "target_core_freq", "target_core_s1freq", "s1_core_s1freq",
    "other_s1_same_name_house", "other_s1_same_name_addr",
    "name_cos", "skel_cos", "addr_cos", "combo_cos", "rule_mask", "rule", "block_score", "rev_rank",
]

_NORM_COLS = ["entity_id", "country_normalized", "state_code", "city", "name_t", "name_core",
              "name_key", "name_skel", "addr_t", "house_c", "is_native"]


# ------------------------------------------------------------------------------
# Hashed TF-IDF helpers
# ------------------------------------------------------------------------------
def _resolve_params(params: Optional[dict]) -> dict:
    p = dict(DEFAULT_BLOCKING_PARAMS)
    if params:
        p.update(params)
    if not p.get("n_jobs"):
        p["n_jobs"] = cpu_threads()
    return p


def _hv(n_features: int) -> HashingVectorizer:
    return HashingVectorizer(
        analyzer="char_wb", ngram_range=(3, 3), n_features=n_features,
        alternate_sign=False, norm=None, lowercase=False, dtype=np.float32,
    )


def _hash_counts(texts: List[str], n_features: int, n_jobs: int) -> sp.csr_matrix:
    hv = _hv(n_features)
    if len(texts) < 100_000 or n_jobs <= 1:
        return hv.transform(texts).tocsr()
    size = (len(texts) + n_jobs - 1) // n_jobs
    chunks = [texts[i:i + size] for i in range(0, len(texts), size)]
    mats = Parallel(n_jobs=n_jobs, backend="loky")(delayed(hv.transform)(c) for c in chunks)
    return sp.vstack(mats).tocsr()


def _apply_tfidf(X: sp.csr_matrix, idf: np.ndarray) -> sp.csr_matrix:
    X = X.astype(np.float32, copy=False)
    X.data = (1.0 + np.log(X.data)) * idf[X.indices]
    X.eliminate_zeros()
    return normalize(X, norm="l2", copy=False).astype(np.float32, copy=False)


def _field_text(df: pd.DataFrame, field: str) -> List[str]:
    if field == "name":
        return df["name_core"].tolist()
    if field == "skel":
        return df["name_skel"].tolist()
    if field == "addr":
        return df["addr_t"].tolist()
    return (df["name_core"] + " " + df["addr_t"]).tolist()


def _rowdot(Q: sp.csr_matrix, T: sp.csr_matrix, qi: np.ndarray, tj: np.ndarray, chunk: int = 500_000) -> np.ndarray:
    out = np.empty(len(qi), dtype=np.float32)
    for s in range(0, len(qi), chunk):
        e = min(s + chunk, len(qi))
        out[s:e] = np.asarray(Q[qi[s:e]].multiply(T[tj[s:e]]).sum(axis=1)).ravel()
    return out


def _add_key_cols(df: pd.DataFrame) -> pd.DataFrame:
    core = df["name_core"].fillna("")
    nospace = core.str.replace(" ", "", regex=False)
    house = df["house_c"].fillna("")
    df["k_p4h"] = np.where((nospace.str.len() >= 4) & (house != ""), nospace.str[:4] + "|" + house, "")
    w12 = core.str.split(" ").str[:2].str.join(" ")
    state = df["state_code"].fillna("")
    df["k_cw12s"] = np.where((w12.str.len() >= 6) & (state != ""), w12 + "|" + state, "")
    # every distinct address number (1-6 digits, not all zeros) + state, space-separated
    st_vals = state.values
    df["k_numst"] = [
        " ".join(sorted({f"{n}|{st}" for n in _NUM_RE.findall(a) if len(n) <= 6 and n.strip("0")}))
        if st else ""
        for a, st in zip(df["addr_t"].fillna("").values, st_vals)
    ]
    sk1 = df["name_skel"].fillna("").str.split(" ").str[0].fillna("")
    df["k_skel1s"] = np.where((sk1.str.len() >= 3) & (state != ""), sk1 + "|" + state, "")
    df["key_cols_v"] = KEY_COLS_VERSION
    return df


_NUM_RE = re.compile(r"\d+")


def _explode_keys(values: np.ndarray, rows: np.ndarray):
    """Split space-separated multi-keys: returns (keys, row index) arrays."""
    ks, rs = [], []
    for v, r in zip(values, rows):
        if v:
            parts = v.split(" ")
            ks.extend(parts)
            rs.extend([r] * len(parts))
    return np.asarray(ks, dtype=object), np.asarray(rs, dtype=np.int64)


def _load_norm(norm_path: Path, country: str, ids: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    con = duckdb.connect()
    con.execute(f"PRAGMA threads={cpu_threads()};")
    f = str(norm_path).replace("\\", "/")
    cols = ", ".join(_NORM_COLS)
    if ids is not None:
        con.register("sel_ids", ids)
        ncols = ", ".join(f"n.{c}" for c in _NORM_COLS)
        df = con.execute(
            f"SELECT {ncols} FROM '{f}' n JOIN sel_ids s ON n.entity_id = s.entity_id "
            f"WHERE n.country_normalized = ? ORDER BY n.entity_id", [country]
        ).df()
    else:
        df = con.execute(f"SELECT {cols} FROM '{f}' WHERE country_normalized = ?", [country]).df()
    con.close()
    for c in _NORM_COLS:
        if c != "is_native":
            df[c] = df[c].fillna("")
    return _add_key_cols(df)


# ------------------------------------------------------------------------------
# Target index (per source file & country)
# ------------------------------------------------------------------------------
class TargetIndex:
    def __init__(self, norm_path: Path, source_label: str, country: str, params: dict, verbose: bool = True):
        self.norm_path = Path(norm_path)
        self.source_label = source_label
        self.country = country
        self.params = params
        self.verbose = verbose
        self.cache_dir = self.norm_path.parent / "tfidf"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        # Prepared target rows (normalized columns + key columns) cached per (source file, country):
        # rebuilding them (DuckDB scan + pandas string ops) dominated every blocking pass.
        self.tidx_dir = self.norm_path.parent / "tidx"
        self.tidx_dir.mkdir(parents=True, exist_ok=True)
        frame_path = self.tidx_dir / f"{self.norm_path.stem}__{country}.parquet"
        if frame_path.is_file():
            self.df = pd.read_parquet(frame_path)
        else:
            self.df = _load_norm(self.norm_path, country)
            self.df.to_parquet(frame_path, index=False)
        cached_v = int(self.df["key_cols_v"].iloc[0]) if ("key_cols_v" in self.df.columns and len(self.df)) else 0
        if cached_v < KEY_COLS_VERSION:
            self.df = _add_key_cols(self.df)
            self.df.to_parquet(frame_path, index=False)
        if "core_freq" not in self.df.columns:
            # How many targets (this source & country) share this exact core name. Rare identical
            # names are near-certain matches even without an address; common ones are not.
            vc = self.df["name_core"].value_counts()
            self.df["core_freq"] = self.df["name_core"].map(vc).fillna(0).astype(np.int32).values
            self.df.to_parquet(frame_path, index=False)
        self.n = len(self.df)
        self._key_tables: Dict[str, pd.DataFrame] = {}
        self.addr_empty = np.asarray(self.df["addr_t"].fillna("").str.len().to_numpy() == 0, dtype=bool)

    def max_df(self, field: str) -> float:
        m = self.params["max_df_frac"]
        return float(m[field]) if isinstance(m, dict) else float(m)

    def _sig(self, field: str) -> str:
        sig = json.dumps({"max_df": self.max_df(field), "n_features": self.params["n_features"],
                          "norm": NORM_VERSION}, sort_keys=True)
        return hashlib.md5(sig.encode()).hexdigest()[:8]

    def key_table(self, col: str) -> pd.DataFrame:
        if col not in self._key_tables:
            path = self.tidx_dir / f"{self.norm_path.stem}__{self.country}__key_{col}__cap{self.params['key_freq_cap']}.parquet"
            if path.is_file():
                self._key_tables[col] = pd.read_parquet(path)
            else:
                if col in MULTI_KEY_COLS:
                    k, r = _explode_keys(self.df[col].fillna("").values, np.arange(self.n))
                    full = pd.DataFrame({"k": k, "tj": r}).drop_duplicates()
                else:
                    full = pd.DataFrame({"k": np.asarray(self.df[col].fillna("").values, dtype=object),
                                         "tj": np.arange(self.n, dtype=np.int64)})
                full = full[full["k"] != ""]
                vc = full["k"].value_counts()
                ok = vc[vc <= self.params["key_freq_cap"]].index
                kt = full[full["k"].isin(ok)].reset_index(drop=True)
                kt["k"] = kt["k"].astype(object)
                kt.to_parquet(path, index=False)
                self._key_tables[col] = kt
        return self._key_tables[col]

    def matrix(self, field: str):
        stem = f"{self.norm_path.stem}__{self.country}__{field}__{self._sig(field)}"
        m_path = self.cache_dir / f"{stem}.npz"
        i_path = self.cache_dir / f"{stem}.idf.npy"
        if m_path.is_file() and i_path.is_file():
            return sp.load_npz(m_path).tocsr(), np.load(i_path)
        t0 = time.time()
        X = _hash_counts(_field_text(self.df, field), self.params["n_features"], self.params["n_jobs"])
        df_counts = np.bincount(X.indices, minlength=X.shape[1])
        idf = (np.log((1.0 + self.n) / (1.0 + df_counts)) + 1.0).astype(np.float32)
        idf[df_counts > self.max_df(field) * self.n] = 0.0  # drop near-ubiquitous n-grams
        T = _apply_tfidf(X, idf)
        del X
        sp.save_npz(m_path, T, compressed=False)
        np.save(i_path, idf)
        if self.verbose:
            print(f"      [Index] Built {self.source_label}/{self.country}/{field}: {T.shape[0]:,} rows, "
                  f"{T.nnz:,} nnz in {time.time() - t0:.1f}s")
        return T, idf


def reverse_table(s1_norm: Path, target_norms: Dict[str, Path], label: str, country: str,
                  params: dict, verbose: bool = True):
    """(R, s1_entity_ids): each target's best S1 rows in the whole S1 file (context rival table, built
    and cached on first use). None when reverse retrieval is disabled."""
    if not params.get("reverse_k"):
        return None
    from src.context import CountryContext
    ctx = CountryContext(s1_norm, target_norms, country, {k: v for k, v in params.items() if k != "ranker"},
                         verbose)
    R = ctx.target(label)["rivals"]["R"]
    ids = ctx.s1.df["entity_id"].values
    del ctx
    gc.collect()
    return R, ids


def _reverse_pairs(s1: pd.DataFrame, rev, k: int):
    """Reverse candidates restricted to the S1 rows being blocked: (q, t, rank) sorted by q."""
    R, full_ids = rev
    k = min(int(k), R.shape[1])
    full_row = pd.Index(full_ids).get_indexer(s1["entity_id"].values)
    q_of_full = np.full(len(full_ids), -1, dtype=np.int32)
    ok = full_row >= 0
    q_of_full[full_row[ok]] = np.nonzero(ok)[0].astype(np.int32)
    s_all = R[:, :k].ravel()
    t_all = np.repeat(np.arange(R.shape[0], dtype=np.int32), k)
    r_all = np.tile(np.arange(k, dtype=np.int8), R.shape[0])
    q_all = np.where(s_all >= 0, q_of_full[np.maximum(s_all, 0)], -1)
    m = q_all >= 0
    q, t, r = q_all[m], t_all[m], r_all[m]
    order = np.argsort(q, kind="stable")
    return q[order].astype(np.int64), t[order].astype(np.int64), r[order]


def _block_source(s1: pd.DataFrame, idx: TargetIndex, params: dict,
                  per_s1_cap: Optional[int] = None, rev=None) -> pd.DataFrame:
    """
    All candidate pairs (compact numeric form) for S1 rows against one target index.
    per_s1_cap keeps only each S1's top pairs by block_score within this source; the final
    cross-source top-N is always a subset of the per-source top-N, so results are unchanged
    while peak memory drops.
    """
    nT = idx.n
    if nT == 0 or len(s1) == 0:
        return pd.DataFrame()
    mats = {f: idx.matrix(f) for f in FIELDS}
    active = [f for f in FIELDS if params["topk"].get(f, 0) > 0]   # topk 0 = cosine feature only
    gmats = {}
    if params.get("use_gpu", True) and GPU_AVAILABLE:
        try:
            gmats = {f: GpuTargetMatrix(mats[f][0]) for f in active}
        except Exception as e:  # not enough GPU memory / CUDA libs missing -> CPU
            print(f"      [GPU] Falling back to CPU top-K for {idx.source_label}/{idx.country}: {e}")
            for g in gmats.values():
                g.free()
            gmats = {}
    chunk = params["query_chunk"]
    out_parts = []
    guard = GpuGuard(max_failures=2)
    rq = rt = rr = None
    if rev is not None:
        rq, rt, rr = _reverse_pairs(s1, rev, params.get("reverse_k", 5))
    starts = range(0, len(s1), chunk)
    it = pbar(starts, total=len(starts), desc=f"{idx.source_label}/{idx.country} top-K"
              f"{' (GPU)' if gmats else ' (CPU)'}", unit="chunk") if len(starts) > 1 else starts

    def knn_chunk(c0: int):
        """Query vectors for every field + top-K pairs for the retrieved fields of one chunk."""
        s1c_ = s1.iloc[c0:c0 + chunk]
        Qs_, found = {}, []
        for f in FIELDS:
            T_, idf_ = mats[f]
            Q_ = _apply_tfidf(_hash_counts(_field_text(s1c_, f), params["n_features"], 1), idf_)
            Qs_[f] = Q_
            if f not in active:
                continue
            rows_ = cols_ = None
            g_ = gmats.get(f)
            if g_ is not None:
                try:
                    rows_, cols_, _ = gpu_topn(Q_, g_, params["topk"][f], params["cos_floor"])
                except Exception:  # GPU out of memory -> CPU for this field/chunk (and later chunks if repeated)
                    rows_ = None
                    guard.failed(f, gmats)
            if rows_ is None:
                C_ = sp_matmul_topn(Q_, T_, top_n=params["topk"][f], threshold=params["cos_floor"],
                                    n_threads=params["n_jobs"]).tocoo()
                rows_, cols_ = C_.row, C_.col
            found.append((f, rows_, cols_))
        return Qs_, found

    from concurrent.futures import ThreadPoolExecutor
    prefetch = ThreadPoolExecutor(max_workers=1)
    start_list = list(starts)
    pending = prefetch.submit(knn_chunk, start_list[0]) if start_list else None
    for ci, c0 in enumerate(it):
        Qs, found = pending.result()
        if ci + 1 < len(start_list):
            pending = prefetch.submit(knn_chunk, start_list[ci + 1])
        s1c = s1.iloc[c0:c0 + chunk]
        q_parts, t_parts, b_parts = [], [], []

        # 1. Exact-key rules
        for rule, col, minlen in KEY_RULES:
            keys = np.asarray(s1c[col].fillna("").values, dtype=object)
            if col in MULTI_KEY_COLS:
                k, r = _explode_keys(keys, np.arange(len(keys)))
                if not len(k):
                    continue
                left = pd.DataFrame({"k": k, "qi": r})
            else:
                valid = np.fromiter((len(k) >= minlen for k in keys), dtype=bool, count=len(keys))
                if not valid.any():
                    continue
                left = pd.DataFrame({"k": keys[valid], "qi": np.nonzero(valid)[0].astype(np.int64)})
            m = left.merge(idx.key_table(col), on="k", how="inner")
            if len(m):
                q_parts.append(m["qi"].values)
                t_parts.append(m["tj"].values)
                b_parts.append(np.full(len(m), RULE_BIT[rule], dtype=np.int32))

        # 2. TF-IDF top-K retrieval (computed by the prefetch thread)
        for f, rows, cols in found:
            if len(rows):
                q_parts.append(rows.astype(np.int64))
                t_parts.append(cols.astype(np.int64))
                b_parts.append(np.full(len(rows), RULE_BIT[f"knn_{f}"], dtype=np.int32))

        # 3. Reverse retrieval (target -> S1) from the rival table
        rk_key = rk_rank = None
        if rq is not None:
            lo, hi = np.searchsorted(rq, [c0, c0 + len(s1c)])
            if hi > lo:
                q_parts.append(rq[lo:hi] - c0)
                t_parts.append(rt[lo:hi])
                b_parts.append(np.full(hi - lo, RULE_BIT["rev_knn"], dtype=np.int32))
                rk_key = (rq[lo:hi] - c0) * nT + rt[lo:hi]
                o = np.argsort(rk_key)
                rk_key, rk_rank = rk_key[o], rr[lo:hi][o]

        if not q_parts:
            continue
        q = np.concatenate(q_parts)
        t = np.concatenate(t_parts)
        b = np.concatenate(b_parts)
        key = q * nT + t
        order = np.argsort(key, kind="stable")
        key, b = key[order], b[order]
        starts = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
        mask = np.bitwise_or.reduceat(b, starts)
        ukey = key[starts]
        q, t = ukey // nT, ukey % nT

        part = {"qi": (q + c0).astype(np.int64), "tj": t.astype(np.int64), "rule_mask": mask.astype(np.int32)}
        rev_rank = np.full(len(ukey), REV_MISSING_RANK, dtype=np.int8)
        if rk_key is not None and len(rk_key):
            pos = np.minimum(np.searchsorted(rk_key, ukey), len(rk_key) - 1)
            hit = rk_key[pos] == ukey
            rev_rank[hit] = rk_rank[pos[hit]]
        part["rev_rank"] = rev_rank
        for f in FIELDS:
            part[f"{f}_cos"] = _rowdot(Qs[f], mats[f][0], q, t)
        s1_addr_empty = (s1c["addr_t"].str.len().values == 0)
        part["addr_missing"] = (s1_addr_empty[q] | idx.addr_empty[t]).astype(np.int8)
        part = pd.DataFrame(part)
        if per_s1_cap:
            part["_cap"] = cap_score(part, params)
            part = (part.sort_values(["qi", "_cap"], ascending=[True, False], kind="stable")
                        .groupby("qi", sort=False).head(per_s1_cap)
                        .drop(columns="_cap").reset_index(drop=True))
        out_parts.append(part)
        # release this chunk's temporaries before the next one (RAM + cached GPU blocks)
        del Qs, found, q_parts, t_parts, b_parts, q, t, b, key, order, mask, ukey, part
        free_gpu()
        if ci % 5 == 4:
            gc.collect()

    prefetch.shutdown(wait=True)
    for g in gmats.values():
        g.free()
    del mats, gmats
    gc.collect()
    if not out_parts:
        return pd.DataFrame()
    return pd.concat(out_parts, ignore_index=True)


def block_score(df: pd.DataFrame) -> np.ndarray:
    """
    Cheap ranking score used to cap candidates per S1 (higher = more likely a match).
    A true match usually has at least one near-identical field, so the score blends the best
    field with the average. When either address is missing, address evidence is replaced by
    name evidence instead of counting as zero (such pairs were previously ranked out).
    """
    name_sim = np.maximum(df["name_cos"].values, df["skel_cos"].values)
    addr = df["addr_cos"].values
    combo = df["combo_cos"].values
    if "addr_missing" in df.columns:
        missing = df["addr_missing"].values.astype(bool)
        addr = np.where(missing, name_sim, addr)
        combo = np.where(missing, np.maximum(combo, name_sim), combo)
    mask = df["rule_mask"].values
    exact_name = ((mask & EXACT_NAME_BITS) != 0).astype(np.float32)
    exact_addr = ((mask & RULE_BIT["exact_addr"]) != 0).astype(np.float32)
    best = np.maximum(np.maximum(addr, name_sim), combo)
    avg = 0.45 * addr + 0.35 * name_sim + 0.20 * combo
    return (0.5 * best + 0.5 * avg + 0.15 * exact_name + 0.15 * exact_addr).astype(np.float32)


# ------------------------------------------------------------------------------
# Learned blocking ranker: decides which candidates survive the per-S1 cap
# ------------------------------------------------------------------------------
RANKER_FEATURES = (["name_cos", "skel_cos", "addr_cos", "combo_cos", "addr_missing", "heuristic", "rev_rank"]
                   + [f"bit_{r}" for r in RULE_NAMES])


def ranker_matrix(df: pd.DataFrame) -> np.ndarray:
    cols = [df[f"{f}_cos"].values.astype(np.float32) for f in FIELDS]
    miss = (df["addr_missing"].values if "addr_missing" in df.columns else np.zeros(len(df))).astype(np.float32)
    rrank = (df["rev_rank"].values if "rev_rank" in df.columns
             else np.full(len(df), REV_MISSING_RANK)).astype(np.float32)
    mask = df["rule_mask"].values
    bits = [((mask & RULE_BIT[r]) != 0).astype(np.float32) for r in RULE_NAMES]
    return np.column_stack(cols + [miss, block_score(df), rrank] + bits).astype(np.float32)


class BlockRanker:
    """
    Small GPU XGBoost model scoring P(true match) from cheap blocking signals (4 retrieval
    cosines, missing-address flag, heuristic score, 11 rule bits). Replaces the hand-weighted
    score when capping candidates per S1: with the heuristic, generic names (many exact-name
    targets at unrelated addresses) pushed true matches past the cap.
    """

    def __init__(self, device: str = "cuda", topk: Optional[dict] = None):
        self.device = device
        self.model = None
        self.topk = dict(topk or {})   # retrieval config it was trained under
        self.features = list(RANKER_FEATURES)  # feature layout it was trained on

    def fit(self, X: np.ndarray, y: np.ndarray):
        import xgboost as xgb
        self.model = xgb.XGBClassifier(
            objective="binary:logistic", tree_method="hist", device=self.device,
            n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.8,
            colsample_bytree=0.9, random_state=42, n_jobs=cpu_threads(),
        )
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray, batch: int = 2_000_000) -> np.ndarray:
        booster = self.model.get_booster()
        out = np.empty(len(X), dtype=np.float32)
        use_gpu = self.device == "cuda" and GPU_AVAILABLE
        if use_gpu:
            from src.gpu_topk import cp
            booster.set_param({"device": "cuda"})
        else:
            booster.set_param({"device": "cpu"})
        for s in range(0, len(X), batch):
            xb = X[s:s + batch]
            if use_gpu:
                try:
                    out[s:s + batch] = cp.asnumpy(booster.inplace_predict(cp.asarray(xb)))
                    continue
                except Exception:
                    use_gpu = False
                    booster.set_param({"device": "cpu"})
            out[s:s + batch] = booster.inplace_predict(xb)
        return out

    def score(self, df: pd.DataFrame, batch: int = 2_000_000) -> np.ndarray:
        out = np.empty(len(df), dtype=np.float32)
        for s in range(0, len(df), batch):
            out[s:s + batch] = self.predict(ranker_matrix(df.iloc[s:s + batch]))
        return out


def cap_score(df: pd.DataFrame, params: dict) -> np.ndarray:
    """Score used to cap candidates: learned ranker when available, heuristic otherwise."""
    ranker = params.get("ranker")
    return ranker.score(df) if ranker is not None else block_score(df)


def _primary_rule(mask: np.ndarray, exact_both: np.ndarray) -> np.ndarray:
    out = np.empty(len(mask), dtype=object)
    out[:] = ""
    for r in reversed(RULE_NAMES):  # lowest bit wins
        out[(mask & RULE_BIT[r]) != 0] = r
    out[exact_both] = "exact_name_addr"
    return out


def s1_name_counts(s1_norm: Path, country: str) -> pd.Series:
    """
    core name -> number of S1 records (whole S1 file, this country) with that core name.
    S1 is the deduplicated reference, so a target whose name belongs to exactly one S1 business
    most likely belongs to that business (P(match) ~ 1 / count), even without an address.
    """
    con = duckdb.connect()
    con.execute(f"PRAGMA threads={cpu_threads()};")
    f = str(s1_norm).replace("\\", "/")
    df = con.execute(f"SELECT name_core, count(*) AS n FROM '{f}' WHERE country_normalized = ? "
                     f"AND name_core != '' GROUP BY 1", [country]).df()
    con.close()
    return pd.Series(df["n"].values.astype(np.int32), index=df["name_core"].astype(object).values)


class S1Competitors:
    """
    Lookups over the WHOLE S1 file of a country (not just the S1 being blocked):
    does another S1 business have the target's exact core name + house number / + address?
    On validation data such pairs are true matches 0.1% / 0.0% of the time: the target belongs
    to that other business. S1 is deduplicated, so this is a clean, leak-free signal that works
    identically on the test set.
    """

    def __init__(self, s1_norm: Path, country: str):
        con = duckdb.connect()
        con.execute(f"PRAGMA threads={cpu_threads()};")
        f = str(s1_norm).replace("\\", "/")
        df = con.execute(f"SELECT entity_id, name_core, house_c, addr_t FROM '{f}' "
                         f"WHERE country_normalized = ? AND name_core != ''", [country]).df()
        con.close()
        self.house = self._index(df[df["house_c"].fillna("") != ""], ["name_core", "house_c"])
        self.addr = self._index(df[df["addr_t"].fillna("") != ""], ["name_core", "addr_t"])

    @staticmethod
    def _index(df: pd.DataFrame, keys: List[str]) -> pd.DataFrame:
        g = df.groupby(keys, sort=False)["entity_id"].agg(["size", "first"])
        g.columns = ["n", "first_id"]
        return g.reset_index()

    @staticmethod
    def _flag(tbl: pd.DataFrame, keys: List[str], left: pd.DataFrame) -> np.ndarray:
        m = left.merge(tbl, left_on=["target_core", keys[1] + "_t"], right_on=keys, how="left")
        n = m["n"].fillna(0).values
        other = (n >= 2) | ((n == 1) & (m["first_id"].values != m["s1_id"].values))
        return other.astype(np.int8)

    def flags(self, out: pd.DataFrame):
        left = pd.DataFrame({
            "s1_id": out["s1_id"].values,
            "target_core": out["target_core"].values,
            "house_c_t": out["target_house"].values,
            "addr_t_t": out["target_addr"].values,
        })
        f_house = self._flag(self.house, ["name_core", "house_c"], left)
        f_addr = self._flag(self.addr, ["name_core", "addr_t"], left)
        f_house[left["house_c_t"].values == ""] = 0
        f_addr[left["addr_t_t"].values == ""] = 0
        return f_house, f_addr


def _materialize(pairs: pd.DataFrame, s1: pd.DataFrame, targets: Dict[str, pd.DataFrame],
                 s1_freq: Optional[pd.Series] = None,
                 competitors: Optional["S1Competitors"] = None) -> pd.DataFrame:
    qi = pairs["qi"].values
    res = {
        "s1_id": s1["entity_id"].values[qi],
        "s1_name": s1["name_t"].values[qi],
        "s1_core": s1["name_core"].values[qi],
        "s1_key": s1["name_key"].values[qi],
        "s1_skel": s1["name_skel"].values[qi],
        "s1_addr": s1["addr_t"].values[qi],
        "s1_city": s1["city"].values[qi],
        "s1_state": s1["state_code"].values[qi],
        "s1_house": s1["house_c"].values[qi],
        "s1_country": s1["country_normalized"].values[qi],
        "s1_native": s1["is_native"].values[qi],
    }
    src = pairs["src"].values
    tj = pairs["tj"].values
    if src.dtype != object:                      # int8 codes -> labels
        src = np.where(src == SRC_CODE["S2"], "S2", "S3").astype(object)
    tcols = {"candidate_id": "entity_id", "target_name": "name_t", "target_core": "name_core",
             "target_key": "name_key", "target_skel": "name_skel", "target_addr": "addr_t",
             "target_city": "city", "target_state": "state_code", "target_house": "house_c",
             "target_country": "country_normalized", "target_native": "is_native",
             "target_core_freq": "core_freq"}
    for out_c, in_c in tcols.items():
        col = np.empty(len(pairs), dtype=object)
        for label, tdf in targets.items():
            sel = src == label
            if sel.any():
                col[sel] = tdf[in_c].values[tj[sel]]
        res[out_c] = col
    res["candidate_source"] = src
    for f in FIELDS:
        res[f"{f}_cos"] = pairs[f"{f}_cos"].values
    res["rule_mask"] = pairs["rule_mask"].values
    res["block_score"] = pairs["block_score"].values
    res["rev_rank"] = (pairs["rev_rank"].values if "rev_rank" in pairs.columns
                       else np.full(len(pairs), REV_MISSING_RANK, dtype=np.int8))
    out = pd.DataFrame(res)
    out["s1_native"] = out["s1_native"].astype(np.int8)
    out["target_native"] = out["target_native"].astype(np.int8)
    exact_both = ((out["s1_name"] == out["target_name"]) & (out["s1_addr"] == out["target_addr"])
                  & (out["s1_name"] != "") & (out["s1_addr"] != "")).values
    out["rule"] = _primary_rule(out["rule_mask"].values, exact_both)
    if s1_freq is not None and len(out):
        out["target_core_s1freq"] = out["target_core"].map(s1_freq).fillna(0).astype(np.int32).values
        out["s1_core_s1freq"] = out["s1_core"].map(s1_freq).fillna(0).astype(np.int32).values
    else:
        out["target_core_s1freq"] = 0
        out["s1_core_s1freq"] = 0
    if competitors is not None and len(out):
        out["other_s1_same_name_house"], out["other_s1_same_name_addr"] = competitors.flags(out)
    else:
        out["other_s1_same_name_house"] = 0
        out["other_s1_same_name_addr"] = 0
    return out[OUTPUT_COLUMNS]


def iter_candidate_pairs(
    s1_ids: List[str],
    s1_parquet_path: Path,
    s2_parquet_path: Path,
    s3_parquet_path: Path,
    max_candidates_per_s1: Optional[int] = 50,
    verbose: bool = True,
    params: Optional[dict] = None,
    yield_chunk_s1: int = 100_000,
) -> Iterator[pd.DataFrame]:
    """Yield candidate-pair DataFrames (OUTPUT_COLUMNS), grouped by country and S1 chunk."""
    p = _resolve_params(params)
    s1_norm = ensure_norm(s1_parquet_path)
    target_norms = [("S2", ensure_norm(s2_parquet_path)), ("S3", ensure_norm(s3_parquet_path))]

    ids_df = pd.DataFrame({"entity_id": list(s1_ids)})
    con = duckdb.connect()
    con.register("sel_ids", ids_df)
    f = str(s1_norm).replace("\\", "/")
    countries = [r[0] for r in con.execute(
        f"SELECT DISTINCT n.country_normalized FROM '{f}' n JOIN sel_ids s ON n.entity_id = s.entity_id "
        f"WHERE n.country_normalized != '' ORDER BY 1").fetchall()]
    con.close()
    if verbose:
        print(f"  [Blocking] {len(s1_ids):,} S1 entities across countries: {countries}")

    for country in countries:
        t0 = time.time()
        s1 = _load_norm(s1_norm, country, ids_df).reset_index(drop=True)
        s1_freq = s1_name_counts(s1_norm, country)
        competitors = S1Competitors(s1_norm, country)
        targets, pair_parts = {}, []
        for label, tpath in target_norms:
            ts = time.time()
            rev = reverse_table(s1_norm, dict(target_norms), label, country, p, verbose)
            idx = TargetIndex(tpath, label, country, p, verbose)
            pairs = _block_source(s1, idx, p, per_s1_cap=max_candidates_per_s1, rev=rev)
            del rev
            if len(pairs):
                # Keep only referenced target rows (bounds memory) and remap indices
                used = np.unique(pairs["tj"].values)
                targets[label] = idx.df.iloc[used].drop(
                    columns=[c for c in ("k_p4h", "k_cw12s", "k_numst", "k_skel1s", "key_cols_v")
                             if c in idx.df.columns]).reset_index(drop=True)
                pairs["tj"] = np.searchsorted(used, pairs["tj"].values)
                pairs["src"] = np.int8(SRC_CODE[label])
                pair_parts.append(pairs)
            if verbose:
                print(f"    S1 -> {label} ({country}): {len(pairs):,} raw pairs for {len(s1):,} S1 in {time.time() - ts:.1f}s")
            del idx
            gc.collect()
        if not pair_parts:
            continue
        pairs = pd.concat(pair_parts, ignore_index=True)
        del pair_parts
        pairs["block_score"] = block_score(pairs)      # heuristic score, kept as a model feature
        pairs["_cap"] = cap_score(pairs, p)            # learned ranker (if trained) decides the cap
        pairs = pairs.sort_values(["qi", "_cap"], ascending=[True, False], kind="stable")
        if max_candidates_per_s1:
            pairs = pairs.groupby("qi", sort=False).head(max_candidates_per_s1)
        pairs = pairs.drop(columns="_cap").reset_index(drop=True)
        if verbose:
            print(f"  [Blocking] {country}: {len(pairs):,} capped pairs for {pairs['qi'].nunique():,} / {len(s1):,} S1 "
                  f"({len(pairs) / max(len(s1), 1):.1f}/S1) in {time.time() - t0:.1f}s")

        qvals = pairs["qi"].values
        for q0 in range(0, len(s1), yield_chunk_s1):
            lo, hi = np.searchsorted(qvals, [q0, q0 + yield_chunk_s1])
            if hi > lo:
                yield _materialize(pairs.iloc[lo:hi], s1, targets, s1_freq, competitors)
        del pairs, targets, s1
        gc.collect()


# ------------------------------------------------------------------------------
# Resumable blocking (test mode): every stage is checkpointed to disk
# ------------------------------------------------------------------------------
def _atomic_parquet(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def _free_memory() -> None:
    gc.collect()
    try:
        from src.gpu_topk import cp
        if cp is not None:
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass


def blocking_signature(s1_ids: List[str], max_candidates_per_s1: Optional[int], params: dict,
                       ranker_path: Optional[Path] = None) -> dict:
    """Everything that changes blocking output; checkpoints are reused only if it matches."""
    ids = sorted(s1_ids)
    sig = {
        "n_s1": len(ids),
        "s1_ids_hash": hashlib.md5("|".join(ids).encode()).hexdigest(),
        "max_candidates_per_s1": max_candidates_per_s1,
        "norm_version": NORM_VERSION,
        "key_cols_version": KEY_COLS_VERSION,
        "rules": RULE_NAMES,
        "params": {k: params.get(k) for k in ("topk", "cos_floor", "key_freq_cap", "max_df_frac", "n_features",
                                              "reverse_k")},
    }
    if ranker_path is not None and Path(ranker_path).is_file():
        st = Path(ranker_path).stat()
        sig["ranker"] = [st.st_size, int(st.st_mtime)]
    return sig


def block_to_checkpoints(
    s1_ids: List[str],
    s1_parquet_path: Path,
    s2_parquet_path: Path,
    s3_parquet_path: Path,
    ckpt_dir: Path,
    max_candidates_per_s1: Optional[int] = 80,
    params: Optional[dict] = None,
    chunk_s1: int = 25_000,
    signature: Optional[dict] = None,
    verbose: bool = True,
) -> List[Path]:
    """
    Resumable blocking. Layout under ckpt_dir:
        signature.json                      settings fingerprint (mismatch -> checkpoints discarded)
        <country>/src_S2.pairs.parquet      per-source candidate pairs (before the cross-source cap)
        <country>/src_S2.targets.parquet    target rows referenced by those pairs
        <country>/chunks/chunk_00000.parquet  materialized candidates for up to chunk_s1 S1 entities
        <country>/chunks/_DONE              all chunks of this country written
    Any interruption resumes from the last finished source / chunk. Returns ordered chunk paths.
    """
    import shutil
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    sig_path = ckpt_dir / "signature.json"
    if signature is not None:
        if sig_path.is_file() and json.loads(sig_path.read_text(encoding="utf-8")) != signature:
            print("  [Blocking checkpoints] settings changed -> discarding old blocking checkpoints")
            shutil.rmtree(ckpt_dir)
            ckpt_dir.mkdir(parents=True, exist_ok=True)
        sig_path.write_text(json.dumps(signature, indent=2), encoding="utf-8")

    p = _resolve_params(params)
    s1_norm = ensure_norm(s1_parquet_path)
    target_norms = [("S2", ensure_norm(s2_parquet_path)), ("S3", ensure_norm(s3_parquet_path))]
    ids_df = pd.DataFrame({"entity_id": list(s1_ids)})
    con = duckdb.connect()
    con.register("sel_ids", ids_df)
    f = str(s1_norm).replace("\\", "/")
    countries = [r[0] for r in con.execute(
        f"SELECT DISTINCT n.country_normalized FROM '{f}' n JOIN sel_ids s ON n.entity_id = s.entity_id "
        f"WHERE n.country_normalized != '' ORDER BY 1").fetchall()]
    con.close()
    if verbose:
        print(f"  [Blocking] {len(s1_ids):,} S1 entities across countries: {countries} (checkpoints: {ckpt_dir})")

    all_chunks: List[Path] = []
    for country in countries:
        cdir = ckpt_dir / country
        chunk_dir = cdir / "chunks"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        done_flag = chunk_dir / "_DONE"
        if done_flag.is_file():
            chunks = sorted(chunk_dir.glob("chunk_*.parquet"))
            if verbose:
                print(f"  [Blocking] {country}: resumed from checkpoint ({len(chunks)} chunks)")
            all_chunks.extend(chunks)
            continue

        t0 = time.time()
        s1 = _load_norm(s1_norm, country, ids_df).reset_index(drop=True)
        targets, pair_parts = {}, []
        for label, tpath in target_norms:
            pf, tf = cdir / f"src_{label}.pairs.parquet", cdir / f"src_{label}.targets.parquet"
            if pf.is_file() and tf.is_file():
                pairs, tdf = pd.read_parquet(pf), pd.read_parquet(tf)
                if verbose:
                    print(f"    S1 -> {label} ({country}): resumed {len(pairs):,} pairs from checkpoint")
            else:
                ts = time.time()
                rev = reverse_table(s1_norm, dict(target_norms), label, country, p, verbose)
                idx = TargetIndex(tpath, label, country, p, verbose)
                pairs = _block_source(s1, idx, p, per_s1_cap=max_candidates_per_s1, rev=rev)
                del rev
                if len(pairs):
                    used = np.unique(pairs["tj"].values)
                    tdf = idx.df.iloc[used].drop(
                        columns=[c for c in ("k_p4h", "k_cw12s", "k_numst", "k_skel1s", "key_cols_v")
                                 if c in idx.df.columns]).reset_index(drop=True)
                    pairs["tj"] = np.searchsorted(used, pairs["tj"].values)
                    pairs["src"] = np.int8(SRC_CODE[label])
                else:
                    tdf = idx.df.iloc[:0].copy()
                _atomic_parquet(tdf, tf)
                _atomic_parquet(pairs, pf)
                if verbose:
                    print(f"    S1 -> {label} ({country}): {len(pairs):,} raw pairs for {len(s1):,} S1 in "
                          f"{time.time() - ts:.1f}s [checkpointed]")
                del idx
                _free_memory()
            if len(pairs):
                targets[label] = tdf
                pair_parts.append(pairs)

        if pair_parts:
            pairs = pd.concat(pair_parts, ignore_index=True)
            del pair_parts
            pairs["block_score"] = block_score(pairs)
            pairs["_cap"] = cap_score(pairs, p)
            pairs = pairs.sort_values(["qi", "_cap"], ascending=[True, False], kind="stable")
            if max_candidates_per_s1:
                pairs = pairs.groupby("qi", sort=False).head(max_candidates_per_s1)
            pairs = pairs.drop(columns="_cap").reset_index(drop=True)
            if verbose:
                print(f"  [Blocking] {country}: {len(pairs):,} capped pairs for {pairs['qi'].nunique():,} / "
                      f"{len(s1):,} S1 ({len(pairs) / max(len(s1), 1):.1f}/S1)")
            s1_freq = s1_name_counts(s1_norm, country)
            competitors = S1Competitors(s1_norm, country)
            qvals = pairs["qi"].values
            starts = list(range(0, len(s1), chunk_s1))
            for k, q0 in enumerate(pbar(starts, total=len(starts), desc=f"{country} save chunks", unit="chunk")):
                out = chunk_dir / f"chunk_{k:05d}.parquet"
                if out.is_file():
                    continue
                lo, hi = np.searchsorted(qvals, [q0, q0 + chunk_s1])
                if hi > lo:
                    _atomic_parquet(_materialize(pairs.iloc[lo:hi], s1, targets, s1_freq, competitors), out)
            del pairs, s1_freq, competitors
        done_flag.write_text(str(time.time()), encoding="utf-8")
        # per-source intermediates are no longer needed once chunks exist
        for label, _ in target_norms:
            for suffix in ("pairs", "targets"):
                (cdir / f"src_{label}.{suffix}.parquet").unlink(missing_ok=True)
        chunks = sorted(chunk_dir.glob("chunk_*.parquet"))
        all_chunks.extend(chunks)
        if verbose:
            print(f"  [Blocking] {country}: {len(chunks)} chunks checkpointed in {time.time() - t0:.1f}s")
        del s1, targets
        _free_memory()
    return all_chunks


def collect_raw_pairs(
    s1_ids: List[str],
    s1_parquet_path: Path,
    s2_parquet_path: Path,
    s3_parquet_path: Path,
    verbose: bool = True,
    params: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Uncapped candidate pairs in compact form (ids + blocking signals), used to train the
    BlockRanker. Returns: s1_id, candidate_id, name/skel/addr/combo_cos, addr_missing, rule_mask.
    """
    p = _resolve_params(params)
    p["ranker"] = None
    s1_norm = ensure_norm(s1_parquet_path)
    target_norms = [("S2", ensure_norm(s2_parquet_path)), ("S3", ensure_norm(s3_parquet_path))]
    ids_df = pd.DataFrame({"entity_id": list(s1_ids)})
    con = duckdb.connect()
    con.register("sel_ids", ids_df)
    f = str(s1_norm).replace("\\", "/")
    countries = [r[0] for r in con.execute(
        f"SELECT DISTINCT n.country_normalized FROM '{f}' n JOIN sel_ids s ON n.entity_id = s.entity_id "
        f"WHERE n.country_normalized != '' ORDER BY 1").fetchall()]
    con.close()
    keep = ["name_cos", "skel_cos", "addr_cos", "combo_cos", "addr_missing", "rule_mask", "rev_rank"]
    parts = []
    for country in countries:
        s1 = _load_norm(s1_norm, country, ids_df).reset_index(drop=True)
        for label, tpath in target_norms:
            ts = time.time()
            rev = reverse_table(s1_norm, dict(target_norms), label, country, p, verbose)
            idx = TargetIndex(tpath, label, country, p, verbose)
            pairs = _block_source(s1, idx, p, per_s1_cap=None, rev=rev)
            del rev
            if len(pairs):
                out = pairs[keep].copy()
                out["s1_id"] = s1["entity_id"].values[pairs["qi"].values]
                out["candidate_id"] = idx.df["entity_id"].values[pairs["tj"].values]
                parts.append(out)
            if verbose:
                print(f"    [Ranker data] S1 -> {label} ({country}): {len(pairs):,} raw pairs "
                      f"for {len(s1):,} S1 in {time.time() - ts:.1f}s")
            del idx, pairs
            gc.collect()
    if not parts:
        return pd.DataFrame(columns=["s1_id", "candidate_id"] + keep)
    return pd.concat(parts, ignore_index=True)


def generate_candidate_pairs(
    s1_ids: List[str],
    s1_parquet_path: Path,
    s2_parquet_path: Path,
    s3_parquet_path: Path,
    max_candidates_per_s1: Optional[int] = 50,
    verbose: bool = True,
    params: Optional[dict] = None,
) -> pd.DataFrame:
    """Materialize all candidate pairs for the given S1 IDs into one DataFrame."""
    t0 = time.time()
    parts = list(iter_candidate_pairs(s1_ids, s1_parquet_path, s2_parquet_path, s3_parquet_path,
                                      max_candidates_per_s1, verbose, params))
    if not parts:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    out = pd.concat(parts, ignore_index=True)
    if verbose:
        print(f"  [Blocking Complete] {len(out):,} pairs for {out['s1_id'].nunique():,} / {len(s1_ids):,} S1 "
              f"entities ({time.time() - t0:.1f}s)")
    return out
