"""
Feature Engineering Engine v2 - Vectorized Pairwise Feature Extraction.

Operates on the blocking v2 output (transliterated / canonicalized fields):
  - RapidFuzz similarities via process.cpdist (multi-threaded, element-wise pairs)
    on full name, legal-stripped core name, phonetic skeleton and canonical address
  - Token Jaccard / overlap on core name and address
  - Numeric-token and house-number agreement (leading zeros stripped)
  - Structured location agreement (state, city) incl. conflicts
  - Retrieval cosines from blocking (name, skeleton, address, combined) + block score
  - Within-S1 context: candidate ranks, gap to the best candidate, candidate counts
  - Blocking rule bits (one column per rule), source and native-script indicators
No country identity features: the test set contains countries unseen in training.
"""

import os
import re
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist

from src.blocking import RULE_NAMES, RULE_BIT
from src.progress import pbar
from src.resources import cpu_threads


def _obj(s):
    """Plain Python-object string array (pyarrow-backed element access is ~10x slower)."""
    return np.array(s.fillna("").astype(str).tolist(), dtype=object)

_DIGITS = re.compile(r"\d+")

# Max rows per worker task in parallel extraction (keeps each pickled task small on Windows)
PARALLEL_TASK_ROWS = 25_000

# Columns _extract_batch reads (only these are shipped to worker processes)
_BATCH_COLS = [
    "s1_name", "target_name", "s1_core", "target_core", "s1_key", "target_key", "s1_skel", "target_skel",
    "s1_addr", "target_addr", "s1_house", "target_house", "s1_state", "target_state", "s1_city",
    "target_city", "name_cos", "skel_cos", "addr_cos", "combo_cos", "block_score", "candidate_source",
    "target_native", "rule_mask",
    # name-frequency signals (were missing here, so parallel extraction silently produced zeros)
    "target_core_freq", "target_core_s1freq", "s1_core_s1freq",
    # competing-S1 signals (see src/blocking.py: _competitor_flags)
    "other_s1_same_name_house", "other_s1_same_name_addr",
]

_FUZZ = [
    # (feature prefix, left col, right col, scorers)
    ("name", "s1_name", "target_name", {"ratio": fuzz.ratio, "tokenset": fuzz.token_set_ratio,
                                          "tokensort": fuzz.token_sort_ratio, "wratio": fuzz.WRatio}),
    ("core", "s1_core", "target_core", {"ratio": fuzz.ratio, "tokenset": fuzz.token_set_ratio,
                                          "partial": fuzz.partial_ratio}),
    ("skel", "s1_skel", "target_skel", {"ratio": fuzz.ratio, "tokenset": fuzz.token_set_ratio}),
    ("addr", "s1_addr", "target_addr", {"ratio": fuzz.ratio, "tokenset": fuzz.token_set_ratio,
                                          "tokensort": fuzz.token_sort_ratio, "partial": fuzz.partial_ratio}),
]

FEATURE_COLUMNS = (
    [f"{p}_{s}" for p, _, _, sc in _FUZZ for s in sc]
    + [
        "name_exact", "core_exact", "key_exact", "skel_exact", "addr_exact",
        "core_jaccard", "core_overlap_ratio", "core_len_ratio", "core_token_diff",
        "first_word_match", "last_word_match", "core_prefix_4",
        "addr_jaccard", "addr_overlap_ratio", "addr_len_ratio",
        "num_jaccard", "num_any_overlap", "num_s1_subset", "has_nums_both",
        "house_match", "has_house_both", "house_conflict",
        "state_match", "has_state_both", "state_conflict",
        "city_match", "has_city_both", "city_in_target_addr",
        "name_cos", "skel_cos", "addr_cos", "combo_cos", "block_score",
        "rank_block", "rank_addr", "rank_name", "gap_block", "gap_addr", "gap_name",
        "n_cands", "n_cands_same_source",
        "is_source3", "target_native",
        "house_edit_sim", "house_num_reldiff", "house_prefix", "num_fuzzy",
        "nospace_ratio", "nospace_partial",
        "s1_addr_empty", "target_addr_empty", "name_only_evidence",
        "n_sib_same_addr", "n_sib_same_core",
        "target_core_logfreq", "rare_name_evidence",
        "s1freq_target_name_log", "s1freq_own_name_log", "name_share_prior",
        "other_s1_same_name_house", "other_s1_same_name_addr",
    ]
    + [f"rule_{r}" for r in RULE_NAMES]
)


def _nonempty(a: np.ndarray) -> np.ndarray:
    return np.fromiter((len(x) > 0 for x in a), dtype=bool, count=len(a))


def _token_stats(left, right):
    n = len(left)
    jac = np.zeros(n, np.float32)
    ovr = np.zeros(n, np.float32)
    first = np.zeros(n, np.float32)
    last = np.zeros(n, np.float32)
    tdiff = np.zeros(n, np.float32)
    for i in range(n):
        ta, tb = left[i].split(), right[i].split()
        if not ta or not tb:
            tdiff[i] = max(len(ta), len(tb))
            continue
        sa, sb = set(ta), set(tb)
        inter = len(sa & sb)
        jac[i] = inter / len(sa | sb)
        ovr[i] = inter / min(len(sa), len(sb))
        first[i] = ta[0] == tb[0]
        last[i] = ta[-1] == tb[-1]
        tdiff[i] = abs(len(sa) - len(sb))
    return jac, ovr, first, last, tdiff


def _num_stats(left, right):
    n = len(left)
    jac = np.zeros(n, np.float32)
    anyo = np.zeros(n, np.float32)
    sub = np.zeros(n, np.float32)
    both = np.zeros(n, np.float32)
    for i in range(n):
        na, nb = set(_DIGITS.findall(left[i])), set(_DIGITS.findall(right[i]))
        if na and nb:
            both[i] = 1.0
            inter = len(na & nb)
            jac[i] = inter / len(na | nb)
            anyo[i] = inter > 0
            sub[i] = na <= nb
    return jac, anyo, sub, both


def _house_stats(left, right):
    """
    House numbers in the data carry digit-level noise (2169 vs 2171, 616 vs 61, 241 vs 141),
    so exact equality is too strict: add edit similarity, relative numeric distance and prefix match.
    """
    n = len(left)
    edit = np.zeros(n, np.float32)
    reldiff = np.ones(n, np.float32)
    prefix = np.zeros(n, np.float32)
    for i in range(n):
        a, b = left[i], right[i]
        if not a or not b:
            continue
        edit[i] = fuzz.ratio(a, b) / 100.0
        prefix[i] = 1.0 if (a.startswith(b) or b.startswith(a)) else 0.0
        da, db = _DIGITS.match(a), _DIGITS.match(b)
        if da and db:
            x, y = int(da.group(0)[:9]), int(db.group(0)[:9])
            reldiff[i] = abs(x - y) / max(x, y, 1)
    return edit, reldiff, prefix


def _num_fuzzy(left, right):
    """Average best edit-similarity of each S1 address number to any target address number."""
    n = len(left)
    out = np.zeros(n, np.float32)
    for i in range(n):
        na, nb = _DIGITS.findall(left[i]), _DIGITS.findall(right[i])
        if na and nb:
            out[i] = sum(max(fuzz.ratio(x, y) for y in nb) for x in na) / (100.0 * len(na))
    return out


def _len_ratio(a, b):
    la = np.fromiter((len(x) for x in a), dtype=np.float32, count=len(a))
    lb = np.fromiter((len(x) for x in b), dtype=np.float32, count=len(b))
    return np.where(np.maximum(la, lb) > 0, np.minimum(la, lb) / np.maximum(np.maximum(la, lb), 1), 0).astype(np.float32)


def add_group_features(df: pd.DataFrame) -> pd.DataFrame:
    """Within-S1 context features. Requires every S1 group to be complete in df."""
    out = pd.DataFrame(index=df.index)
    g = df.groupby("s1_id", sort=False)
    name_sim = np.maximum(df["name_cos"].values, df["skel_cos"].values)
    tmp = pd.DataFrame({"s1_id": df["s1_id"].values, "block": df["block_score"].values,
                        "addr": df["addr_cos"].values, "name": name_sim}, index=df.index)
    tg = tmp.groupby("s1_id", sort=False)
    for col in ["block", "addr", "name"]:
        out[f"rank_{col}"] = tg[col].rank(ascending=False, method="min").astype(np.float32).values
        out[f"gap_{col}"] = (tg[col].transform("max").values - tmp[col].values).astype(np.float32)
    out["n_cands"] = g["candidate_id"].transform("size").astype(np.float32).values
    out["n_cands_same_source"] = df.groupby(["s1_id", "candidate_source"], sort=False)["candidate_id"] \
        .transform("size").astype(np.float32).values
    # Sibling support: other candidates of the same S1 sharing this exact address / core name.
    # True matches of one S1 are variants of the same business, so they tend to cluster.
    t_addr = _obj(df["target_addr"])
    t_core = _obj(df["target_core"])
    same_addr = df.groupby(["s1_id", "target_addr"], sort=False)["candidate_id"].transform("size").values - 1
    same_core = df.groupby(["s1_id", "target_core"], sort=False)["candidate_id"].transform("size").values - 1
    out["n_sib_same_addr"] = np.where(t_addr != "", same_addr, 0).astype(np.float32)
    out["n_sib_same_core"] = np.where(t_core != "", same_core, 0).astype(np.float32)
    return out


class FeatureExtractor:
    """
    Stateless pairwise feature extractor (kept as a class for pipeline/joblib compatibility).
    """
    version = 2

    def __init__(self, tfidf_ngram_range=(3, 4), max_features=50_000, workers: int = None):
        self.workers = workers
        self.is_fitted = True

    def fit_tfidfs(self, names_corpus: Optional[pd.Series] = None, addrs_corpus: Optional[pd.Series] = None):
        """No-op: TF-IDF cosines now come from the blocking stage."""
        return self

    def save(self, filepath: str):
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: str) -> "FeatureExtractor":
        return joblib.load(filepath)

    def extract_features(self, pairs_df: pd.DataFrame, batch_size: int = 500_000,
                         group_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        group_df: precomputed add_group_features() rows aligned with pairs_df (use when pairs_df is a
        subsample of the full candidate set, e.g. negative-subsampled training pairs).
        """
        if len(pairs_df) == 0:
            return pd.DataFrame(columns=FEATURE_COLUMNS)
        pairs_df = pairs_df.reset_index(drop=True)
        grp = add_group_features(pairs_df) if group_df is None else group_df.reset_index(drop=True)
        n_jobs = cpu_threads()
        batches = None
        if len(pairs_df) >= 500_000 and n_jobs > 1:
            # Pairwise features are independent per row: split across processes. Workers use
            # single-threaded cpdist to avoid oversubscribing cores. Tasks are kept small
            # (<= PARALLEL_TASK_ROWS rows): large pickled string frames make Windows pipes fail
            # with WinError 1450 ("insufficient system resources").
            from joblib import Parallel, delayed
            need = [c for c in _BATCH_COLS if c in pairs_df.columns]
            size = min(batch_size, PARALLEL_TASK_ROWS, (len(pairs_df) + n_jobs - 1) // n_jobs)
            try:
                worker = FeatureExtractor(workers=1)
                gen = Parallel(n_jobs=n_jobs, backend="loky", return_as="generator", pre_dispatch="2*n_jobs")(
                    delayed(worker._extract_batch)(pairs_df[need].iloc[b0:b0 + size])
                    for b0 in range(0, len(pairs_df), size))
                n_tasks = (len(pairs_df) + size - 1) // size
                batches = list(pbar(gen, total=n_tasks, desc=f"features ({n_jobs} CPU procs)", unit="batch"))
            except Exception as e:  # never lose a long run to a transport error: fall back to serial
                print(f"    [Features] Parallel extraction failed ({type(e).__name__}: {str(e)[:120]}); "
                      f"falling back to single-process extraction for this chunk")
                batches = None
        if batches is None:
            starts = range(0, len(pairs_df), batch_size)
            it = pbar(starts, total=len(starts), desc="features", unit="batch") if len(starts) > 1 else starts
            batches = [self._extract_batch(pairs_df.iloc[b0:b0 + batch_size]) for b0 in it]
        out = pd.concat(batches, ignore_index=True)
        for c in grp.columns:
            out[c] = grp[c].values
        return out[FEATURE_COLUMNS]

    def _extract_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        f = {}
        cols = {c: _obj(df[c]) for c in
                ["s1_name", "target_name", "s1_core", "target_core", "s1_key", "target_key",
                 "s1_skel", "target_skel", "s1_addr", "target_addr", "s1_house", "target_house",
                 "s1_state", "target_state", "s1_city", "target_city"]}

        for prefix, lc, rc, scorers in _FUZZ:
            a, b = cols[lc], cols[rc]
            both = _nonempty(a) & _nonempty(b)
            for sname, scorer in scorers.items():
                v = cpdist(list(a), list(b), scorer=scorer, workers=self.workers or cpu_threads(), dtype=np.float32) / 100.0
                f[f"{prefix}_{sname}"] = np.where(both, v, 0.0).astype(np.float32)

        for name, lc, rc in [("name_exact", "s1_name", "target_name"), ("core_exact", "s1_core", "target_core"),
                             ("key_exact", "s1_key", "target_key"), ("skel_exact", "s1_skel", "target_skel"),
                             ("addr_exact", "s1_addr", "target_addr")]:
            a, b = cols[lc], cols[rc]
            f[name] = ((a == b) & _nonempty(a)).astype(np.float32)

        jac, ovr, first, last, tdiff = _token_stats(cols["s1_core"], cols["target_core"])
        f.update(core_jaccard=jac, core_overlap_ratio=ovr, first_word_match=first,
                 last_word_match=last, core_token_diff=tdiff)
        f["core_len_ratio"] = _len_ratio(cols["s1_core"], cols["target_core"])
        a4 = np.array([x.replace(" ", "")[:4] for x in cols["s1_core"]], dtype=object)
        b4 = np.array([x.replace(" ", "")[:4] for x in cols["target_core"]], dtype=object)
        f["core_prefix_4"] = ((a4 == b4) & np.fromiter((len(x) == 4 for x in a4), bool, len(a4))).astype(np.float32)

        jac, ovr, _, _, _ = _token_stats(cols["s1_addr"], cols["target_addr"])
        f.update(addr_jaccard=jac, addr_overlap_ratio=ovr)
        f["addr_len_ratio"] = _len_ratio(cols["s1_addr"], cols["target_addr"])

        njac, nany, nsub, nboth = _num_stats(cols["s1_addr"], cols["target_addr"])
        f.update(num_jaccard=njac, num_any_overlap=nany, num_s1_subset=nsub, has_nums_both=nboth)

        for key, lc, rc in [("house", "s1_house", "target_house"), ("state", "s1_state", "target_state"),
                            ("city", "s1_city", "target_city")]:
            a, b = cols[lc], cols[rc]
            both = _nonempty(a) & _nonempty(b)
            f[f"{key}_match"] = ((a == b) & both).astype(np.float32)
            f[f"has_{key}_both"] = both.astype(np.float32)
            if key != "city":
                f[f"{key}_conflict"] = ((a != b) & both).astype(np.float32)
        h_edit, h_rel, h_pre = _house_stats(cols["s1_house"], cols["target_house"])
        f.update(house_edit_sim=h_edit, house_num_reldiff=h_rel, house_prefix=h_pre)
        f["num_fuzzy"] = _num_fuzzy(cols["s1_addr"], cols["target_addr"])

        # Space-free name comparison: "easternheartlandhealth com" vs "eastern heartland health"
        a_ns = [x.replace(" ", "") for x in cols["s1_core"]]
        b_ns = [x.replace(" ", "") for x in cols["target_core"]]
        both = _nonempty(cols["s1_core"]) & _nonempty(cols["target_core"])
        for sname, scorer in [("nospace_ratio", fuzz.ratio), ("nospace_partial", fuzz.partial_ratio)]:
            v = cpdist(a_ns, b_ns, scorer=scorer, workers=self.workers or cpu_threads(), dtype=np.float32) / 100.0
            f[sname] = np.where(both, v, 0.0).astype(np.float32)

        s1_ae = ~_nonempty(cols["s1_addr"])
        t_ae = ~_nonempty(cols["target_addr"])
        f["s1_addr_empty"] = s1_ae.astype(np.float32)
        f["target_addr_empty"] = t_ae.astype(np.float32)
        f["name_only_evidence"] = np.where(s1_ae | t_ae, f["core_tokenset"], 0.0).astype(np.float32)

        # Name rarity: an identical name used by only 1-3 targets is a match ~95% of the time even
        # without an address, while one shared by 50+ targets almost never is.
        freq = (df["target_core_freq"].fillna(0).values.astype(np.float32)
                if "target_core_freq" in df.columns else np.zeros(len(df), np.float32))
        f["target_core_logfreq"] = np.log1p(freq).astype(np.float32)
        f["rare_name_evidence"] = (f["core_tokenset"] / np.log2(2.0 + freq)).astype(np.float32)

        # S1-side name frequency (S1 is deduplicated): a target whose name is used by exactly one S1
        # business belongs to it ~95-99% of the time even with no address; ~1/n when n S1 share it.
        def _col(name):
            return (df[name].fillna(0).values.astype(np.float32) if name in df.columns
                    else np.zeros(len(df), np.float32))
        tf1, sf1 = _col("target_core_s1freq"), _col("s1_core_s1freq")
        f["s1freq_target_name_log"] = np.log1p(tf1).astype(np.float32)
        f["s1freq_own_name_log"] = np.log1p(sf1).astype(np.float32)
        f["other_s1_same_name_house"] = _col("other_s1_same_name_house")
        f["other_s1_same_name_addr"] = _col("other_s1_same_name_addr")
        f["name_share_prior"] = np.where(f["core_exact"] > 0.5, 1.0 / np.maximum(tf1, 1.0),
                                         f["core_tokenset"] / np.maximum(sf1, 1.0)).astype(np.float32)

        f["city_in_target_addr"] = np.fromiter(
            (1.0 if c and c in t else 0.0 for c, t in zip(cols["s1_city"], cols["target_addr"])),
            dtype=np.float32, count=len(df))

        for c in ["name_cos", "skel_cos", "addr_cos", "combo_cos", "block_score"]:
            f[c] = df[c].values.astype(np.float32)
        f["is_source3"] = (df["candidate_source"].values == "S3").astype(np.float32)
        f["target_native"] = df["target_native"].values.astype(np.float32)
        mask = df["rule_mask"].values
        for r in RULE_NAMES:
            f[f"rule_{r}"] = ((mask & RULE_BIT[r]) != 0).astype(np.float32)

        return pd.DataFrame(f)
