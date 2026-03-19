from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def verify_manifest(df: pd.DataFrame) -> dict[str, Any]:
    missing_files: list[str] = []
    broken_labels: list[str] = []
    empty_annotations = 0
    empty_masks = 0

    for _, row in df.iterrows():
        for field in ("image_path", "label_path", "mask_path", "video_path"):
            value = row.get(field)
            if value and not Path(value).exists():
                missing_files.append(value)
        if row.get("label_path") and not row.get("classes_mapped"):
            broken_labels.append(str(row["label_path"]))
        if row.get("label_path") and len(row.get("classes_mapped") or []) == 0:
            empty_annotations += 1
        if row.get("mask_path") == "":
            empty_masks += 1

    leakage = 0
    if not df.empty:
        leakage = int((df.groupby("sample_id")["split_group"].nunique() > 1).sum())

    class_counts: dict[str, int] = {}
    for classes in df.get("classes_mapped", pd.Series(dtype=object)).tolist():
        for class_name in classes or []:
            class_counts[class_name] = class_counts.get(class_name, 0) + 1

    return {
        "samples_total": int(len(df)),
        "datasets": df["source_dataset"].value_counts().to_dict() if not df.empty else {},
        "task_types": df["task_type"].value_counts().to_dict() if not df.empty else {},
        "class_counts": class_counts,
        "missing_files_count": len(missing_files),
        "missing_files_examples": missing_files[:20],
        "broken_labels_count": len(broken_labels),
        "broken_labels_examples": broken_labels[:20],
        "split_leakage_groups": leakage,
        "empty_annotations": empty_annotations,
        "empty_masks": empty_masks,
    }
