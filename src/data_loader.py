"""
Data Loader - Loads Parquet datasets, parses Ground Truth, and performs 70-30 S1 Entity Split.
"""

import random
from pathlib import Path
from typing import Dict, Set, Tuple, List, Optional

import duckdb
import pandas as pd


def load_ground_truth(gt_path: Path) -> Dict[str, Set[str]]:
    """
    Load train_ground_truth.tsv into a dictionary:
      { source1_entity_id: {matched_target_id_1, matched_target_id_2, ...} }
    Singletons map to an empty set.
    """
    gt_map: Dict[str, Set[str]] = {}
    if not Path(gt_path).is_file():
        raise FileNotFoundError(f"Ground truth file not found: {gt_path}")

    df_gt = pd.read_csv(gt_path, sep="\t", dtype=str, keep_default_na=False)
    for row in df_gt.itertuples(index=False):
        s1_id = str(row.source1_entity_id).strip()
        matched_str = str(row.matched_entity_ids).strip()
        if matched_str:
            gt_map[s1_id] = {x.strip() for x in matched_str.split(",") if x.strip()}
        else:
            gt_map[s1_id] = set()

    return gt_map


def get_all_s1_ids(s1_parquet_path: Path, limit: Optional[int] = None) -> List[str]:
    """Retrieve ordered list of all Source 1 entity IDs from Parquet."""
    con = duckdb.connect()
    limit_clause = f"LIMIT {limit}" if limit else ""
    s1_file = str(s1_parquet_path).replace("\\", "/")
    query = f"""
        SELECT entity_id 
        FROM '{s1_file}' 
        ORDER BY entity_id 
        {limit_clause}
    """
    df = con.execute(query).df()
    return df["entity_id"].tolist()


def split_s1_entities(
    s1_parquet_path: Path,
    train_ratio: float = 0.70,
    val_ratio: float = 0.30,
    random_seed: int = 42,
    sample_size: Optional[int] = None,
    sample_seed: Optional[int] = None,
    exclude_ids: Optional[set] = None,
) -> Tuple[List[str], List[str]]:
    """
    Perform a strict Source 1 Entity-Level Split (default 70-30).
    Guarantees that all candidate pairs for any given S1 entity belong
    strictly to either the training set or the validation set (zero data leakage).

    Returns:
      (train_s1_ids, val_s1_ids)
    """
    if sample_seed is None and not exclude_ids:
        all_ids = get_all_s1_ids(s1_parquet_path, limit=sample_size)   # first N by entity_id (legacy)
    else:
        # Random sample of N S1 from the whole file, optionally excluding S1 used by other runs, so
        # several runs can be trained on disjoint samples and ensembled (--sample-seed / --exclude-runs).
        pool = get_all_s1_ids(s1_parquet_path)
        if exclude_ids:
            pool = [i for i in pool if i not in exclude_ids]
        pool_rng = random.Random(1000003 + (sample_seed or 0))
        all_ids = pool_rng.sample(pool, min(sample_size, len(pool))) if sample_size else pool
    
    rng = random.Random(random_seed)
    shuffled_ids = list(all_ids)
    rng.shuffle(shuffled_ids)

    n_total = len(shuffled_ids)
    n_train = int(n_total * train_ratio)

    train_ids = sorted(shuffled_ids[:n_train])
    val_ids = sorted(shuffled_ids[n_train:])

    return train_ids, val_ids


def get_distinct_countries(parquet_path: Path) -> List[str]:
    """Dynamically discover all unique country codes present in a Parquet dataset."""
    con = duckdb.connect()
    p_file = str(parquet_path).replace("\\", "/")
    query = f"""
        SELECT DISTINCT country_normalized
        FROM '{p_file}'
        WHERE country_normalized != ''
        ORDER BY country_normalized
    """
    results = con.execute(query).fetchall()
    return [r[0] for r in results]
