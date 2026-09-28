"""
Context features - pair evidence that needs the WHOLE Source 1 file, not just the two records.

Motivation (test analysis of run_100k_v1): France names are built from a small vocabulary and often
contain the city ("Lille Club SARL", "Pornic Club SARL", "Saint-Herblain Institut SARL"), so many
different S1 businesses look alike. The pairwise model (trained on US / India, where look-alikes are
~20x rarer) gave 0.99 to e.g. "Defense Sante SAS | 99 Rue Auber" for a target at "99 Rue du Chene
Houpline", and 1.4% of French targets were claimed by 2+ S1 with prob >= 0.86 (US: 0.06%).

Two country-agnostic feature families, learned only from the provided data:

1. Distinctive-token evidence (names and addresses). Word tokens are weighted by their IDF over the
   S1 file of that country, so a shared city word or "club" counts little while a mismatched rare
   word ("institut" vs "developpement", "auber" vs "chene") counts a lot:
   idf-weighted cosine / Jaccard, share of S1 idf mass missing from the target and vice versa.

2. Rival S1 businesses (reverse retrieval). Every target is queried against the whole S1 file
   (char 3-gram TF-IDF on name + address, top RIVAL_K). Rivals are re-scored with the idf evidence
   and the best RIVAL_KEEP are cached per target. For a pair (s1, target) the model then sees how
   s1 compares with the best OTHER S1 business for that target (margin, #better rivals, the rival's
   own name / address evidence). S1 is deduplicated, so a better-matching rival means the target
   most likely belongs to the rival. Computed identically on train, validation and test.

3. Target duplicate structure (v4). Distractor records (targets that belong to no S1 business) are
   single perturbed copies: in the training ground truth only 1.2% of them share (name key, house number)
   with another S2/S3 record and 4.6% share the exact address, vs 32% / 43% of real records (a business
   usually has several S2/S3 records). The test set has ~2x more distractors per S1 than training, so
   "this candidate has no duplicate anywhere" is the signal that separates a look-alike distractor
   ("807 Heights Blvd" vs "1001 Heights Blvd") from a real record. Counted over all S2 + S3 records of
   the country (no labels involved), cached once per dataset.

All heavy artefacts are cached under dataset/norm/ctx/ and reused by every run.
"""

import gc
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import HashingVectorizer
from sparse_dot_topn import sp_matmul_topn

from src.blocking import TargetIndex, _apply_tfidf, _field_text, _hash_counts, _resolve_params
from src.gpu_topk import GPU_AVAILABLE, GpuGuard, GpuTargetMatrix, free_gpu, gpu_topn
from src.progress import pbar
from src.text_norm import NORM_VERSION, ensure_norm

CTX_VERSION = 4       # pair-feature version (v4: target duplicate counts)
RIVAL_VERSION = 3     # rival-table cache version (unchanged by v4: cached tables stay valid)
DUP_VERSION = 1
WORD_VERSION = 3      # word-matrix cache version (independent of the rival algorithm version)
RIVAL_K = 10          # reverse retrieval depth (char 3-gram cosine on name + address)
RIVAL_KEEP = 5        # rivals kept per target after idf re-scoring
RIVAL_FLOOR = 0.25    # minimum retrieval cosine for a rival
RIVAL_QUERY_CHUNK = 50_000
WORD_FEATURES = 2 ** 20
# Words in more than this share of S1 records ("rue", "de", "club", state names) are left out of the
# word retrieval index only (their postings would make the sparse product ~N work per query).
WORD_RETRIEVAL_MAX_DF = 0.01

CONTEXT_COLUMNS = [
    "nm_idf_cos", "nm_idf_jacc", "nm_s1_miss", "nm_t_extra", "nm_miss_abs", "nm_extra_abs",
    "ad_idf_cos", "ad_idf_jacc", "ad_s1_miss", "ad_t_extra", "ad_miss_abs", "ad_extra_abs",
    "ctx_own_score", "rv_has_rival", "rv_best_score", "rv_margin", "rv_best_nm", "rv_best_ad",
    "rv_best_house", "rv_n_better", "rv_n_close",
]
# Other S2/S3 records of the same country sharing a key with the candidate (NaN when the key is empty):
# (name key, house no.), exact address, (name key, city), (name skeleton, house no.), and the same two
# first keys counted in the other source only.
DUP_COLUMNS = ["t_dup_nkh", "t_dup_addr", "t_dup_nkc", "t_dup_skh", "t_dup_xs_nkh", "t_dup_xs_addr"]
CONTEXT_COLUMNS = CONTEXT_COLUMNS + DUP_COLUMNS

# Light extra address cleanup for token evidence (French numbering: "N 26" -> anyascii "ndeg",
# "30 bis" / "30bis" / "30 ter" -> "30").
_ADDR_FIX = [
    (re.compile(r"\bndeg\b"), " no "),
    (re.compile(r"\b(\d+)\s*(bis|ter|quater)\b"), r"\1"),
]


def _addr_tok_text(a: str) -> str:
    for rx, rep in _ADDR_FIX:
        a = rx.sub(rep, a)
    return a


def _word_bin(texts: List[str]) -> sp.csr_matrix:
    hv = HashingVectorizer(analyzer=str.split, n_features=WORD_FEATURES, binary=True, norm=None,
                           alternate_sign=False, dtype=np.float32)
    return hv.transform(texts).tocsr()


def _word_bin_chunked(texts: List[str], chunk: int = 500_000) -> sp.csr_matrix:
    parts = [_word_bin(texts[i:i + chunk]) for i in range(0, len(texts), chunk)]
    return sp.vstack(parts).tocsr() if parts else sp.csr_matrix((0, WORD_FEATURES), dtype=np.float32)


def _rowsum(M: sp.spmatrix) -> np.ndarray:
    return np.asarray(M.sum(axis=1)).ravel().astype(np.float32)


def _weighted(M: sp.csr_matrix, idf: np.ndarray) -> sp.csr_matrix:
    W = M.copy()
    W.data = idf[W.indices].astype(np.float32)
    return W


def idf_stats(A: sp.csr_matrix, B: sp.csr_matrix, idf: np.ndarray) -> Dict[str, np.ndarray]:
    """Row-aligned binary token matrices A (S1 side), B (target side) -> idf evidence per row."""
    Aw, Bw = _weighted(A, idf), _weighted(B, idf)
    inter = _rowsum(Aw.multiply(B))
    ta, tb = _rowsum(Aw), _rowsum(Bw)
    dot = _rowsum(Aw.multiply(Bw))
    na = np.sqrt(_rowsum(Aw.multiply(Aw)))
    nb = np.sqrt(_rowsum(Bw.multiply(Bw)))
    both = (ta > 0) & (tb > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        cos = np.where(both, dot / np.maximum(na * nb, 1e-9), np.nan)
        jac = np.where(both, inter / np.maximum(ta + tb - inter, 1e-9), np.nan)
        s1_miss = np.where(both, (ta - inter) / np.maximum(ta, 1e-9), np.nan)
        t_extra = np.where(both, (tb - inter) / np.maximum(tb, 1e-9), np.nan)
    return {
        "cos": cos.astype(np.float32), "jacc": jac.astype(np.float32),
        "s1_miss": s1_miss.astype(np.float32), "t_extra": t_extra.astype(np.float32),
        "miss_abs": np.where(both, ta - inter, np.nan).astype(np.float32),
        "extra_abs": np.where(both, tb - inter, np.nan).astype(np.float32),
    }


def _word_combo(WN: sp.csr_matrix, WA: sp.csr_matrix, idf_n: np.ndarray, idf_a: np.ndarray) -> sp.csr_matrix:
    """[name | address] idf-weighted word vectors, each half L2-normalised with weight sqrt(0.5):
    the dot product of two rows is ~0.5 * name idf-cosine + 0.5 * address idf-cosine."""
    from sklearn.preprocessing import normalize
    parts = []
    for W, idf in ((WN, idf_n), (WA, idf_a)):
        M = _weighted(W, idf)
        M.eliminate_zeros()
        parts.append(normalize(M, norm="l2", copy=False) * np.float32(np.sqrt(0.5)))
    return sp.hstack(parts, format="csr", dtype=np.float32)


def _ctx_score(nm_cos: np.ndarray, ad_cos: np.ndarray, house_eq: np.ndarray) -> np.ndarray:
    """Cheap evidence score used to rank rivals (missing fields count as 0)."""
    return (0.5 * np.nan_to_num(nm_cos) + 0.4 * np.nan_to_num(ad_cos) + 0.1 * house_eq).astype(np.float32)


def _house_eq(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return ((a == b) & (a != "")).astype(np.float32)


def _sig(*parts) -> str:
    return hashlib.md5(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:8]


class _Side:
    """Rows of one normalized file & country: id -> row, house numbers, binary word matrices."""

    def __init__(self, idx: TargetIndex, cache_dir: Path, verbose: bool):
        self.df = idx.df
        self.n = len(self.df)
        self.row_of = pd.Index(self.df["entity_id"].values)
        self.house = np.asarray(self.df["house_c"].fillna("").values, dtype=object)
        stem = f"{idx.norm_path.stem}__{idx.country}"
        tag = _sig(WORD_VERSION, NORM_VERSION, WORD_FEATURES)
        wn_p, wa_p = cache_dir / f"{stem}__wn__{tag}.npz", cache_dir / f"{stem}__wa__{tag}.npz"
        if wn_p.is_file() and wa_p.is_file():
            self.WN, self.WA = sp.load_npz(wn_p).tocsr(), sp.load_npz(wa_p).tocsr()
        else:
            t0 = time.time()
            self.WN = _word_bin_chunked(self.df["name_core"].fillna("").tolist())
            self.WA = _word_bin_chunked([_addr_tok_text(a) for a in self.df["addr_t"].fillna("").tolist()])
            sp.save_npz(wn_p, self.WN, compressed=False)
            sp.save_npz(wa_p, self.WA, compressed=False)
            if verbose:
                print(f"      [Context] word matrices {stem}: {self.n:,} rows in {time.time() - t0:.1f}s")


class CountryContext:
    """
    S1 file context for one country + lazily loaded target sides (S2 / S3) with cached rival tables.
    """

    def __init__(self, s1_norm: Path, target_norms: Dict[str, Path], country: str,
                 params: Optional[dict] = None, verbose: bool = True):
        self.p = _resolve_params(params)
        self.country = country
        self.verbose = verbose
        self.cache_dir = Path(s1_norm).parent / "ctx"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.s1_idx = TargetIndex(s1_norm, "S1", country, self.p, verbose)
        self.s1 = _Side(self.s1_idx, self.cache_dir, verbose)
        df_n = np.bincount(self.s1.WN.indices, minlength=WORD_FEATURES)
        df_a = np.bincount(self.s1.WA.indices, minlength=WORD_FEATURES)
        n = max(self.s1.n, 1)
        self.idf_n = (np.log((1.0 + n) / (1.0 + df_n)) + 1.0).astype(np.float32)
        self.idf_a = (np.log((1.0 + n) / (1.0 + df_a)) + 1.0).astype(np.float32)
        # retrieval-only idf: ubiquitous words removed
        self.ridf_n = np.where(df_n > WORD_RETRIEVAL_MAX_DF * n, 0.0, self.idf_n).astype(np.float32)
        self.ridf_a = np.where(df_a > WORD_RETRIEVAL_MAX_DF * n, 0.0, self.idf_a).astype(np.float32)
        self.target_norms = dict(target_norms)
        self._targets: Dict[str, dict] = {}
        self._dup_df: Optional[pd.DataFrame] = None

    # ---------------------------------------------------------------- target duplicate counts
    def _dup_table(self) -> pd.DataFrame:
        """Duplicate counts for every S2/S3 record of this country (cached parquet, built with DuckDB)."""
        if self._dup_df is not None:
            return self._dup_df
        norms = [self.target_norms[k] for k in ("S2", "S3") if k in self.target_norms]
        stems = "__".join(Path(p).stem for p in norms)
        path = self.cache_dir / f"tdup__{stems}__{self.country}__v{DUP_VERSION}.parquet"
        if not path.is_file():
            import duckdb
            t0 = time.time()
            con = duckdb.connect()
            from src.resources import cpu_threads
            con.execute(f"PRAGMA threads={cpu_threads()}")
            sel = " UNION ALL ".join(
                f"SELECT entity_id, '{k}' AS src, coalesce(name_key,'') nk, coalesce(name_skel,'') sk, "
                f"coalesce(house_c,'') hc, coalesce(addr_t,'') ad, coalesce(city,'') ci "
                f"FROM '{str(p).replace(chr(92), '/')}' WHERE country_normalized = $c"
                for k, p in zip(("S2", "S3"), norms))
            df = con.execute(f"""
                WITH t AS ({sel}), k AS (
                  SELECT entity_id, src,
                    CASE WHEN nk <> '' AND hc <> '' THEN nk || '|' || hc END k_nkh,
                    CASE WHEN ad <> '' THEN ad END k_addr,
                    CASE WHEN nk <> '' AND ci <> '' THEN nk || '|' || ci END k_nkc,
                    CASE WHEN sk <> '' AND hc <> '' THEN sk || '|' || hc END k_skh
                  FROM t)
                SELECT entity_id,
                  CASE WHEN k_nkh IS NULL THEN NULL ELSE count(*) OVER (PARTITION BY k_nkh) - 1 END t_dup_nkh,
                  CASE WHEN k_addr IS NULL THEN NULL ELSE count(*) OVER (PARTITION BY k_addr) - 1 END t_dup_addr,
                  CASE WHEN k_nkc IS NULL THEN NULL ELSE count(*) OVER (PARTITION BY k_nkc) - 1 END t_dup_nkc,
                  CASE WHEN k_skh IS NULL THEN NULL ELSE count(*) OVER (PARTITION BY k_skh) - 1 END t_dup_skh,
                  CASE WHEN k_nkh IS NULL THEN NULL ELSE count(*) OVER (PARTITION BY k_nkh)
                       - count(*) OVER (PARTITION BY k_nkh, src) END t_dup_xs_nkh,
                  CASE WHEN k_addr IS NULL THEN NULL ELSE count(*) OVER (PARTITION BY k_addr)
                       - count(*) OVER (PARTITION BY k_addr, src) END t_dup_xs_addr
                FROM k""", {"c": self.country}).df()
            con.close()
            for c in DUP_COLUMNS:
                df[c] = pd.to_numeric(df[c], errors="coerce").astype(np.float32)
            tmp = path.with_suffix(".tmp.parquet")
            df.to_parquet(tmp, index=False)
            tmp.replace(path)
            if self.verbose:
                print(f"      [Context] duplicate counts {self.country}: {len(df):,} targets in {time.time() - t0:.1f}s")
        self._dup_df = pd.read_parquet(path)
        return self._dup_df

    def _dups_for(self, side: _Side) -> Dict[str, np.ndarray]:
        dt = self._dup_table()
        pos = pd.Index(dt["entity_id"].values).get_indexer(side.df["entity_id"].values)
        ok = pos >= 0
        out = {}
        for c in DUP_COLUMNS:
            v = np.full(side.n, np.nan, dtype=np.float32)
            v[ok] = dt[c].to_numpy(dtype=np.float32, na_value=np.nan)[pos[ok]]
            out[c] = v
        return out

    # ---------------------------------------------------------------- targets + rival tables
    def target(self, label: str) -> dict:
        if label not in self._targets:
            idx = TargetIndex(self.target_norms[label], label, self.country, self.p, self.verbose)
            side = _Side(idx, self.cache_dir, self.verbose)
            self._targets[label] = {"side": side, "rivals": self._rivals(label, idx, side),
                                    "dups": self._dups_for(side)}
            if all(k in self._targets for k in ("S2", "S3") if k in self.target_norms):
                self._dup_df = None   # both sides aligned: the country table is no longer needed
        return self._targets[label]

    def release(self, label: str) -> None:
        self._targets.pop(label, None)
        gc.collect()

    def _rivals(self, label: str, idx: TargetIndex, side: _Side) -> Dict[str, np.ndarray]:
        tag = _sig(RIVAL_VERSION, NORM_VERSION, RIVAL_K, RIVAL_KEEP, RIVAL_FLOOR, self.p["max_df_frac"],
                   self.p["n_features"], self.s1.n)
        path = self.cache_dir / (f"rivals__{idx.norm_path.stem}__vs__{self.s1_idx.norm_path.stem}"
                                 f"__{self.country}__{tag}.npz")
        if path.is_file():
            z = np.load(path)
            return {k: z[k] for k in z.files}
        t0 = time.time()
        nT = side.n
        R = np.full((nT, RIVAL_KEEP), -1, dtype=np.int32)
        S = np.zeros((nT, RIVAL_KEEP), dtype=np.float32)
        NM = np.zeros((nT, RIVAL_KEEP), dtype=np.float32)
        AD = np.zeros((nT, RIVAL_KEEP), dtype=np.float32)
        H = np.zeros((nT, RIVAL_KEEP), dtype=np.int8)
        # Two retrieval views over the whole S1 file: char 3-grams (robust to typos) and
        # idf-weighted words (robust to rare-fragment domination, e.g. "Loire-Atlantique").
        S1c, idf_c = self.s1_idx.matrix("combo")
        S1w = _word_combo(self.s1.WN, self.s1.WA, self.ridf_n, self.ridf_a)
        index = {"char": S1c, "word": S1w}
        G = {}
        if self.p.get("use_gpu", True) and GPU_AVAILABLE:
            try:
                G = {k: GpuTargetMatrix(m) for k, m in index.items()}
            except Exception as e:
                print(f"      [Context] GPU unavailable for rival retrieval ({e}); using CPU")
                for g in G.values():
                    g.free()
                G = {}
        starts = list(range(0, nT, RIVAL_QUERY_CHUNK))
        guard = GpuGuard(max_failures=2)
        for c0 in pbar(starts, total=len(starts), desc=f"rivals {label}/{self.country}"
                       f"{' (GPU)' if G else ' (CPU)'}", unit="chunk"):
            tdf = idx.df.iloc[c0:c0 + RIVAL_QUERY_CHUNK]
            queries = {
                "char": _apply_tfidf(_hash_counts(_field_text(tdf, "combo"), self.p["n_features"], 1), idf_c),
                "word": _word_combo(side.WN[c0:c0 + RIVAL_QUERY_CHUNK], side.WA[c0:c0 + RIVAL_QUERY_CHUNK],
                                    self.ridf_n, self.ridf_a),
            }
            r_parts, c_parts = [], []
            for view, Q in queries.items():
                rows = None
                if view in G:
                    try:
                        rows, cols, _ = gpu_topn(Q, G[view], RIVAL_K, RIVAL_FLOOR)
                    except Exception:   # GPU out of memory -> CPU for this chunk (and later ones if repeated)
                        rows = None
                        guard.failed(view, G)
                if rows is None:
                    C = sp_matmul_topn(Q, index[view], top_n=RIVAL_K, threshold=RIVAL_FLOOR,
                                       n_threads=self.p["n_jobs"]).tocoo()
                    rows, cols = C.row.astype(np.int64), C.col.astype(np.int64)
                r_parts.append(rows)
                c_parts.append(cols)
            rows, cols = np.concatenate(r_parts), np.concatenate(c_parts)
            if not len(rows):
                continue
            key = np.unique(rows * np.int64(self.s1.n) + cols)
            rows, cols = key // self.s1.n, key % self.s1.n
            t_rows = rows + c0
            nm = idf_stats(self.s1.WN[cols], side.WN[t_rows], self.idf_n)["cos"]
            ad = idf_stats(self.s1.WA[cols], side.WA[t_rows], self.idf_a)["cos"]
            he = _house_eq(self.s1.house[cols], side.house[t_rows])
            sc = _ctx_score(nm, ad, he)
            order = np.lexsort((-sc, t_rows))
            t_rows, cols, sc, nm, ad, he = t_rows[order], cols[order], sc[order], nm[order], ad[order], he[order]
            first = np.searchsorted(t_rows, t_rows, side="left")
            rank = np.arange(len(t_rows)) - first
            keep = rank < RIVAL_KEEP
            tr, rk = t_rows[keep], rank[keep]
            R[tr, rk] = cols[keep]
            S[tr, rk] = sc[keep]
            NM[tr, rk] = np.nan_to_num(nm[keep])
            AD[tr, rk] = np.nan_to_num(ad[keep])
            H[tr, rk] = he[keep].astype(np.int8)
            del queries, r_parts, c_parts, rows, cols, t_rows, nm, ad, he, sc, order, first, rank, keep, tr, rk
            free_gpu()
        for g in G.values():
            g.free()
        out = {"R": R, "S": S, "NM": NM, "AD": AD, "H": H}
        tmp = path.with_suffix(".tmp.npz")
        np.savez(tmp, **out)
        tmp.replace(path)
        if self.verbose:
            print(f"      [Context] rival table {label}/{self.country}: {nT:,} targets in {time.time() - t0:.1f}s")
        return out

    # ---------------------------------------------------------------- pair features
    def pair_features(self, s1_ids: np.ndarray, cand_ids: np.ndarray, batch: int = 1_000_000) -> pd.DataFrame:
        n = len(s1_ids)
        out = {c: np.full(n, np.nan, dtype=np.float32) for c in CONTEXT_COLUMNS}
        s1r_all = self.s1.row_of.get_indexer(s1_ids)
        labels = np.array([c[:2] for c in cand_ids], dtype=object)
        for label in ("S2", "S3"):
            sel_idx = np.flatnonzero((labels == label) & (s1r_all >= 0))
            if not len(sel_idx) or label not in self.target_norms:
                continue
            tg = self.target(label)
            side, rv, dups = tg["side"], tg["rivals"], tg["dups"]
            tr_all = side.row_of.get_indexer(cand_ids[sel_idx])
            ok = tr_all >= 0
            sel_idx, tr_all = sel_idx[ok], tr_all[ok]
            for b0 in range(0, len(sel_idx), batch):
                ii = sel_idx[b0:b0 + batch]
                tr = tr_all[b0:b0 + batch]
                sr = s1r_all[ii]
                nm = idf_stats(self.s1.WN[sr], side.WN[tr], self.idf_n)
                ad = idf_stats(self.s1.WA[sr], side.WA[tr], self.idf_a)
                for k, v in nm.items():
                    out[f"nm_idf_{k}" if k in ("cos", "jacc") else f"nm_{k}"][ii] = v
                for k, v in ad.items():
                    out[f"ad_idf_{k}" if k in ("cos", "jacc") else f"ad_{k}"][ii] = v
                own = _ctx_score(nm["cos"], ad["cos"], _house_eq(self.s1.house[sr], side.house[tr]))
                R, S = rv["R"][tr], rv["S"][tr]
                valid = (R >= 0) & (R != sr[:, None])
                has = valid.any(axis=1)
                first = np.argmax(valid, axis=1)
                ar = np.arange(len(tr))
                best = np.where(has, S[ar, first], 0.0).astype(np.float32)
                out["ctx_own_score"][ii] = own
                out["rv_has_rival"][ii] = has.astype(np.float32)
                out["rv_best_score"][ii] = best
                out["rv_margin"][ii] = own - best
                out["rv_best_nm"][ii] = np.where(has, rv["NM"][tr][ar, first], 0.0)
                out["rv_best_ad"][ii] = np.where(has, rv["AD"][tr][ar, first], 0.0)
                out["rv_best_house"][ii] = np.where(has, rv["H"][tr][ar, first], 0).astype(np.float32)
                out["rv_n_better"][ii] = ((S > own[:, None] + 1e-6) & valid).sum(axis=1).astype(np.float32)
                out["rv_n_close"][ii] = ((S >= own[:, None] - 0.05) & valid).sum(axis=1).astype(np.float32)
                for c in DUP_COLUMNS:
                    out[c][ii] = dups[c][tr]
        return pd.DataFrame(out)


def _countries_of(s1_ids: np.ndarray, s1_norm: Path) -> np.ndarray:
    import duckdb
    con = duckdb.connect()
    con.register("ids", pd.DataFrame({"entity_id": pd.unique(s1_ids)}))
    f = str(s1_norm).replace("\\", "/")
    m = con.execute(f"SELECT n.entity_id, n.country_normalized FROM '{f}' n JOIN ids USING (entity_id)").df()
    con.close()
    return pd.Series(m["country_normalized"].values, index=m["entity_id"].values).reindex(s1_ids).fillna("").values


class ContextBuilder:
    """Adds CONTEXT_COLUMNS to candidate pairs of any split; keeps one country context in memory."""

    def __init__(self, s1_path: Path, s2_path: Path, s3_path: Path, params: Optional[dict] = None,
                 verbose: bool = True):
        self.s1_norm = ensure_norm(s1_path)
        self.t_norms = {"S2": ensure_norm(s2_path), "S3": ensure_norm(s3_path)}
        self.params = {k: v for k, v in (params or {}).items() if k != "ranker"}
        self.verbose = verbose
        self._ctx: Optional[CountryContext] = None

    def country_ctx(self, country: str) -> CountryContext:
        if self._ctx is None or self._ctx.country != country:
            self._ctx = None
            gc.collect()
            self._ctx = CountryContext(self.s1_norm, self.t_norms, country, self.params, self.verbose)
        return self._ctx

    def prepare(self, countries: List[str]) -> None:
        """Build (and cache) all rival tables up front, one country at a time."""
        for c in countries:
            ctx = self.country_ctx(c)
            for label in ("S2", "S3"):
                ctx.target(label)
                ctx.release(label)
        self.close()

    def features(self, s1_ids, cand_ids, countries: Optional[np.ndarray] = None) -> pd.DataFrame:
        s1_ids = np.asarray(s1_ids, dtype=object)
        cand_ids = np.asarray(cand_ids, dtype=object)
        if countries is None:
            countries = _countries_of(s1_ids, self.s1_norm)
        countries = np.asarray(countries, dtype=object)
        out = {c: np.full(len(s1_ids), np.nan, dtype=np.float32) for c in CONTEXT_COLUMNS}
        for c in [x for x in pd.unique(countries) if x]:
            sel = np.flatnonzero(countries == c)
            f = self.country_ctx(c).pair_features(s1_ids[sel], cand_ids[sel])
            for col in CONTEXT_COLUMNS:
                out[col][sel] = f[col].values
        return pd.DataFrame(out)

    def close(self) -> None:
        self._ctx = None
        gc.collect()
