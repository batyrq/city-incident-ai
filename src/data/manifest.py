from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from src.schemas.common import DatasetRecord
from src.utils.io import dump_yaml, ensure_dir


MANIFEST_COLUMNS = [
    "sample_id",
    "source_dataset",
    "task_type",
    "image_path",
    "label_path",
    "mask_path",
    "video_path",
    "video_id",
    "frame_id",
    "camera_id",
    "event_id",
    "width",
    "height",
    "classes_raw",
    "classes_mapped",
    "is_negative",
    "weather",
    "time_of_day",
    "split_group",
    "roi_tags",
]


def records_to_dataframe(records: list[DatasetRecord]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(columns=MANIFEST_COLUMNS)
    return pd.DataFrame([record.model_dump() for record in records], columns=MANIFEST_COLUMNS)


def save_manifest(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    df.to_parquet(path, index=False)


def build_dataset_stats(df: pd.DataFrame) -> dict[str, Any]:
    classes_counter: Counter[str] = Counter()
    task_counter: Counter[str] = Counter()
    per_dataset: dict[str, int] = defaultdict(int)
    mask_images = 0
    val_samples = 0
    anomaly_normal_images = 0

    for _, row in df.iterrows():
        task_counter[row["task_type"]] += 1
        per_dataset[row["source_dataset"]] += 1
        for class_name in row["classes_mapped"] or []:
            classes_counter[class_name] += 1
        if row.get("mask_path"):
            mask_images += 1
        if row.get("split_group") == "val":
            val_samples += 1
        if row["task_type"] == "anomaly" and row["is_negative"]:
            anomaly_normal_images += 1

    return {
        "images_total": int(len(df)),
        "positive_instances_total": int(sum(classes_counter.values())),
        "positive_instances_per_class": dict(classes_counter),
        "classes_present": sorted(classes_counter.keys()),
        "num_classes_present": len(classes_counter),
        "mask_images": mask_images,
        "val_samples": val_samples,
        "task_counts": dict(task_counter),
        "dataset_counts": dict(per_dataset),
        "anomaly_normal_images": anomaly_normal_images,
        "has_train_split": bool((df["split_group"] == "train").any()) if not df.empty else False,
        "has_val_split": bool((df["split_group"] == "val").any()) if not df.empty else False,
    }


def export_yolo_data_files(df: pd.DataFrame, output_dir: Path) -> None:
    ensure_dir(output_dir)
    spec = {
        "path": str(output_dir.resolve()),
        "train": "manifest_train.txt",
        "val": "manifest_val.txt",
        "names": ["fire", "smoke", "flood_water", "traffic_incident", "vehicle"],
    }
    train_images = [
        str(Path(path).resolve())
        for path in df.loc[(df["split_group"] == "train") & df["image_path"].notna(), "image_path"].tolist()
    ]
    val_images = [
        str(Path(path).resolve())
        for path in df.loc[(df["split_group"] == "val") & df["image_path"].notna(), "image_path"].tolist()
    ]
    (output_dir / "manifest_train.txt").write_text("\n".join(train_images), encoding="utf-8")
    (output_dir / "manifest_val.txt").write_text("\n".join(val_images), encoding="utf-8")
    dump_yaml(spec, output_dir / "yolo_detect_data.yaml")
    dump_yaml(spec, output_dir / "yolo_seg_data.yaml")
