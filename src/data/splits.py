from __future__ import annotations

import hashlib

import pandas as pd


def assign_split_group(sample_id: str, train_ratio: float = 0.7, val_ratio: float = 0.2) -> str:
    key = int(hashlib.sha256(sample_id.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    if key < train_ratio:
        return "train"
    if key < train_ratio + val_ratio:
        return "val"
    return "test"


def apply_splits(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    existing = out.get("split_group")
    if existing is None:
        out["split_group"] = out["sample_id"].apply(assign_split_group)
        return out
    out["split_group"] = out.apply(
        lambda row: row["split_group"] if row.get("split_group") in {"train", "val", "test"} else assign_split_group(row["sample_id"]),
        axis=1,
    )
    return out
