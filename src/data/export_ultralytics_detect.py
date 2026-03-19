from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.adapters.yolo_utils import copy_or_link, detect_class_names
from src.utils.io import dump_yaml, ensure_dir


def export_ultralytics_detect_dataset(
    manifest_df: pd.DataFrame,
    output_dir: str | Path,
    labels_config: dict[str, Any],
    use_symlink: bool = True,
) -> dict[str, Any]:
    output_root = ensure_dir(Path(output_dir))
    detect_df = manifest_df[(manifest_df["task_type"] == "detection") & manifest_df["image_path"].notna()].copy() if not manifest_df.empty else pd.DataFrame()
    summary = {"exported_images": 0, "exported_labels": 0, "skipped": 0, "output_dir": str(output_root), "link_mode": "symlink" if use_symlink else "copy"}
    if detect_df.empty:
        return summary

    train_list: list[str] = []
    val_list: list[str] = []
    for split_name in ("train", "val"):
        ensure_dir(output_root / "images" / split_name)
        ensure_dir(output_root / "labels" / split_name)

    for _, row in detect_df.iterrows():
        split_name = row["split_group"] if row["split_group"] in {"train", "val"} else "train"
        image_src = Path(row["image_path"])
        label_src = Path(row["label_path"]) if row.get("label_path") else None
        if not image_src.exists() or label_src is None or not label_src.exists():
            summary["skipped"] += 1
            continue
        image_dst = output_root / "images" / split_name / image_src.name
        label_dst = output_root / "labels" / split_name / f"{image_src.stem}.txt"
        copy_or_link(image_src, image_dst, use_symlink=use_symlink)
        copy_or_link(label_src, label_dst, use_symlink=False)
        resolved_path = str(image_dst.resolve())
        if split_name == "train":
            train_list.append(resolved_path)
        else:
            val_list.append(resolved_path)
        summary["exported_images"] += 1
        summary["exported_labels"] += 1

    (output_root / "train.txt").write_text("\n".join(train_list), encoding="utf-8")
    (output_root / "val.txt").write_text("\n".join(val_list), encoding="utf-8")
    dump_yaml(
        {
            "path": str(output_root.resolve()),
            "train": str((output_root / "train.txt").resolve()),
            "val": str((output_root / "val.txt").resolve()),
            "names": {idx: name for idx, name in enumerate(detect_class_names(labels_config))},
        },
        output_root / "dataset.yaml",
    )
    return summary
