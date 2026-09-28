"""
Evaluation Metrics - Macro-Averaged F_0.5 Metric and Optimal Threshold Tuning.
Implements Section 14 of the Business Entity Resolution Architecture.

Decision layer:
  1. Target exclusivity: every S2/S3 record belongs to at most one S1 entity (verified on the
     training ground truth), so each candidate is kept only for its highest-probability S1.
  2. Two-level threshold: an S1's top-1 remaining candidate is accepted if prob >= t1, every
     other candidate if prob >= t2.
"""

from typing import Dict, Set, List, Tuple, Any, Optional
import numpy as np
import pandas as pd


def compute_entity_f05(pred_ids: Set[str], true_ids: Set[str]) -> float:
    """
    Compute F_0.5 score for a single Source 1 entity.
    Includes singleton scoring (true_ids == empty set).
    """
    if not true_ids and not pred_ids:
        return 1.0  # Correct singleton
    if not true_ids and pred_ids:
        return 0.0  # False merge on singleton
    if true_ids and not pred_ids:
        return 0.0  # Missed match

    tp = len(pred_ids & true_ids)
    if tp == 0:
        return 0.0

    fp = len(pred_ids - true_ids)
    fn = len(true_ids - pred_ids)

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    f05 = (1.25 * precision * recall) / (0.25 * precision + recall)
    return float(f05)


def evaluate_predictions(
    predicted_matches: Dict[str, Set[str]],
    ground_truth_map: Dict[str, Set[str]],
    all_eval_s1_ids: List[str],
) -> Dict[str, float]:
    """
    Evaluate predicted matches against ground truth across all evaluation S1 entities.
    Returns:
      macro_f05, precision, recall, singleton_accuracy, total_matches_predicted
    """
    total_entities = len(all_eval_s1_ids)
    if total_entities == 0:
        return {"macro_f05": 0.0, "precision": 0.0, "recall": 0.0, "singleton_accuracy": 0.0}

    f05_scores = []
    total_tp = 0
    total_fp = 0
    total_fn = 0
    singletons_total = 0
    singletons_correct = 0

    for s1_id in all_eval_s1_ids:
        true_ids = ground_truth_map.get(s1_id, set())
        pred_ids = predicted_matches.get(s1_id, set())

        score = compute_entity_f05(pred_ids, true_ids)
        f05_scores.append(score)

        if not true_ids:
            singletons_total += 1
            if not pred_ids:
                singletons_correct += 1
        else:
            tp = len(pred_ids & true_ids)
            fp = len(pred_ids - true_ids)
            fn = len(true_ids - pred_ids)
            total_tp += tp
            total_fp += fp
            total_fn += fn

    macro_f05 = float(np.mean(f05_scores))
    overall_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    overall_rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    singleton_acc = singletons_correct / singletons_total if singletons_total > 0 else 1.0

    return {
        "macro_f05": macro_f05,
        "precision": overall_prec,
        "recall": overall_rec,
        "singleton_accuracy": singleton_acc,
        "total_entities": total_entities,
        "total_matches_predicted": total_tp + total_fp,
        "singletons_correct": singletons_correct,
        "singletons_total": singletons_total,
    }


# ------------------------------------------------------------------------------
# Decision layer
# ------------------------------------------------------------------------------
def prepare_decisions(scored: pd.DataFrame, exclusive_targets: bool = True) -> pd.DataFrame:
    """
    Input columns: s1_id, candidate_id, prob (optionally label).
    Applies target exclusivity and adds per-S1 rank (0 = top-1) of the surviving candidates.
    """
    df = scored
    if exclusive_targets and len(df):
        df = (df.sort_values(["candidate_id", "prob", "s1_id"], ascending=[True, False, True], kind="stable")
                .drop_duplicates("candidate_id", keep="first"))
    df = df.sort_values(["s1_id", "prob"], ascending=[True, False], kind="stable").reset_index(drop=True)
    df["s1_rank"] = df.groupby("s1_id", sort=False).cumcount().astype(np.int32)
    return df


def accept_mask(dec: pd.DataFrame, t1: float, t2: float, max_matches_per_s1: int = 10) -> np.ndarray:
    prob = dec["prob"].values
    rank = dec["s1_rank"].values
    return np.where(rank == 0, prob >= t1, prob >= t2) & (rank < max_matches_per_s1)


def decide_matches(scored: pd.DataFrame, t1: float, t2: float, max_matches_per_s1: int = 10,
                   exclusive_targets: bool = True) -> Dict[str, List[str]]:
    """Final S1 -> [matched ids] map (ordered by probability)."""
    dec = prepare_decisions(scored, exclusive_targets)
    acc = dec[accept_mask(dec, t1, t2, max_matches_per_s1)]
    return acc.groupby("s1_id", sort=False)["candidate_id"].apply(list).to_dict()


def _fast_eval(dec: pd.DataFrame, accepted: np.ndarray, s1_codes: np.ndarray, n_true: np.ndarray,
               fp_weight: float = 1.0) -> Dict[str, float]:
    """
    Vectorized Macro F0.5 over all eval entities (entities without candidates predict nothing).
    fp_weight > 1 counts every false match as that many (test-calibrated selection: the test set has
    more distractor targets than validation, so a false match is more likely there).
    """
    n = len(n_true)
    pred = np.bincount(s1_codes[accepted], minlength=n)
    tp = np.bincount(s1_codes[accepted & dec["label"].values], minlength=n)
    single = n_true == 0
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(pred > 0, tp / np.maximum(tp + fp_weight * (pred - tp), 1e-9), 0.0)
        r = np.where(n_true > 0, tp / np.maximum(n_true, 1), 0.0)
        f = np.where(tp > 0, 1.25 * p * r / (0.25 * p + r), 0.0)
    f = np.where(single, (pred == 0).astype(float), f)
    tp_ns, pred_ns = tp[~single].sum(), pred[~single].sum()
    return {
        "macro_f05": float(f.mean()),
        "precision": float(tp_ns / pred_ns) if pred_ns else 0.0,
        "recall": float(tp_ns / n_true.sum()) if n_true.sum() else 0.0,
        "singleton_accuracy": float((pred[single] == 0).mean()) if single.any() else 1.0,
        "total_matches_predicted": int(pred.sum()),
    }


def _eval_frame(val_pairs_df, val_probs, ground_truth_map, val_s1_ids, exclusive_targets):
    df_eval = val_pairs_df[["s1_id", "candidate_id", "rule"]].copy()
    df_eval["prob"] = np.asarray(val_probs, dtype=np.float32)
    df_eval.loc[df_eval["rule"] == "exact_name_addr", "prob"] = 1.0
    df_eval["label"] = [c in ground_truth_map.get(s, ()) for s, c in zip(df_eval["s1_id"], df_eval["candidate_id"])]
    dec = prepare_decisions(df_eval, exclusive_targets)
    ids = pd.Index(list(val_s1_ids))
    codes = ids.get_indexer(dec["s1_id"])
    keep = codes >= 0
    dec = dec[keep].reset_index(drop=True)
    codes = codes[keep]
    n_true = np.array([len(ground_truth_map.get(s, ())) for s in ids], dtype=np.int64)
    return dec, codes, n_true


def candidate_recall(val_pairs_df: pd.DataFrame, ground_truth_map: Dict[str, Set[str]], val_s1_ids: List[str]) -> Dict[str, float]:
    """Blocking recall ceiling: share of true links present in the candidate set."""
    ids = set(val_s1_ids)
    total = sum(len(ground_truth_map.get(s, ())) for s in ids)
    sub = val_pairs_df[val_pairs_df["s1_id"].isin(ids)]
    hit = sum(c in ground_truth_map.get(s, ()) for s, c in zip(sub["s1_id"], sub["candidate_id"]))
    return {
        "candidate_recall": hit / total if total else 0.0,
        "avg_candidates_per_s1": len(sub) / max(len(ids), 1),
    }


def sweep_thresholds(
    val_pairs_df: pd.DataFrame,
    val_probs: np.ndarray,
    ground_truth_map: Dict[str, Set[str]],
    val_s1_ids: List[str],
    min_thresh: float = 0.20,
    max_thresh: float = 0.95,
    step: float = 0.02,
    max_matches_per_s1: int = 10,
    exclusive_targets: bool = True,
) -> Tuple[float, pd.DataFrame]:
    """
    Single-threshold sweep (t1 = t2) maximizing Macro F_0.5.
    Returns:
      (optimal_threshold, sweep_df)
    """
    dec, codes, n_true = _eval_frame(val_pairs_df, val_probs, ground_truth_map, val_s1_ids, exclusive_targets)
    records = []
    for t in np.arange(min_thresh, max_thresh + 1e-5, step):
        t_val = round(float(t), 4)
        m = _fast_eval(dec, accept_mask(dec, t_val, t_val, max_matches_per_s1), codes, n_true)
        m["threshold"] = t_val
        records.append(m)
    sweep_df = pd.DataFrame(records)
    cols = ["threshold", "macro_f05", "precision", "recall", "singleton_accuracy", "total_matches_predicted"]
    sweep_df = sweep_df[cols]
    best = float(sweep_df.loc[sweep_df["macro_f05"].idxmax(), "threshold"])
    return best, sweep_df


def sweep_thresholds_2d(
    val_pairs_df: pd.DataFrame,
    val_probs: np.ndarray,
    ground_truth_map: Dict[str, Set[str]],
    val_s1_ids: List[str],
    min_thresh: float = 0.10,
    max_thresh: float = 0.95,
    step: float = 0.02,
    max_matches_per_s1: int = 10,
    exclusive_targets: bool = True,
    fp_weight: float = 1.0,
) -> Tuple[Tuple[float, float], pd.DataFrame]:
    """
    Two-level sweep over (t1 for the top-1 candidate, t2 for the others) maximizing Macro F_0.5.
    Returns:
      ((t1, t2), sweep_df)
    """
    dec, codes, n_true = _eval_frame(val_pairs_df, val_probs, ground_truth_map, val_s1_ids, exclusive_targets)
    grid = [round(float(t), 4) for t in np.arange(min_thresh, max_thresh + 1e-5, step)]
    records = []
    for t1 in grid:
        for t2 in grid:
            acc = accept_mask(dec, t1, t2, max_matches_per_s1)
            m = _fast_eval(dec, acc, codes, n_true)
            m["macro_f05_fpw"] = (_fast_eval(dec, acc, codes, n_true, fp_weight)["macro_f05"]
                                  if fp_weight != 1.0 else m["macro_f05"])
            m["t1"], m["t2"] = t1, t2
            records.append(m)
    sweep_df = pd.DataFrame(records)
    cols = ["t1", "t2", "macro_f05", "macro_f05_fpw", "precision", "recall", "singleton_accuracy",
            "total_matches_predicted"]
    # ranked by the (possibly false-match-weighted) objective; macro_f05 is always the plain metric
    sweep_df = sweep_df[cols].sort_values(["macro_f05_fpw", "macro_f05"], ascending=False).reset_index(drop=True)
    best = sweep_df.iloc[0]
    return (float(best["t1"]), float(best["t2"])), sweep_df


def entity_f05_vector(val_pairs_df: pd.DataFrame, val_probs: np.ndarray, ground_truth_map: Dict[str, Set[str]],
                      val_s1_ids: List[str], t1: float, t2: float, max_matches_per_s1: int = 10,
                      exclusive_targets: bool = True) -> np.ndarray:
    """Per-entity F0.5 (aligned with val_s1_ids) at thresholds (t1, t2) - for subset diagnostics."""
    dec, codes, n_true = _eval_frame(val_pairs_df, val_probs, ground_truth_map, val_s1_ids, exclusive_targets)
    acc = accept_mask(dec, t1, t2, max_matches_per_s1)
    n = len(n_true)
    pred = np.bincount(codes[acc], minlength=n)
    tp = np.bincount(codes[acc & dec["label"].values], minlength=n)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(pred > 0, tp / np.maximum(pred, 1), 0.0)
        r = np.where(n_true > 0, tp / np.maximum(n_true, 1), 0.0)
        f = np.where(tp > 0, 1.25 * p * r / (0.25 * p + r), 0.0)
    return np.where(n_true == 0, (pred == 0).astype(float), f)
