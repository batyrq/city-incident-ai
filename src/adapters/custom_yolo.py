from __future__ import annotations

import logging
from pathlib import Path

from src.adapters.base import DatasetAdapter
from src.adapters.yolo_utils import (
    canonical_label_path,
    create_record,
    detect_class_names,
    detect_name_to_idx,
    load_dataset_yaml,
    map_raw_name,
    resolve_names_map,
)
from src.schemas.common import DatasetRecord
from src.utils.io import ensure_dir


LOGGER = logging.getLogger(__name__)


class CustomYoloAdapter(DatasetAdapter):
    def validate_raw(self, raw_root: Path) -> tuple[bool, list[str]]:
        required = [
            raw_root / "images" / "train",
            raw_root / "images" / "val",
            raw_root / "labels" / "train",
            raw_root / "labels" / "val",
        ]
        missing = [str(path) for path in required if not path.exists()]
        self.debug_summary = {
            "dataset_name": self.dataset_name,
            "layout_recognized": len(missing) == 0,
            "layout_name": "images_labels_split_dirs",
            "data_present": False,
            "reason": "required split directories missing" if missing else None,
            "missing_paths": missing,
        }
        return (len(missing) == 0, missing)

    def prepare(self, config: dict, labels_config: dict) -> list[DatasetRecord]:
        raw_root = Path(config["raw_root"])
        valid, missing = self.validate_raw(raw_root)
        if not valid:
            LOGGER.warning("custom_yolo adapter skipped because layout is incomplete: %s", missing)
            return []

        dataset_yaml = load_dataset_yaml(raw_root)
        names_map = resolve_names_map(dataset_yaml) or {idx: name for idx, name in enumerate(detect_class_names(labels_config))}
        detect_idx = detect_name_to_idx(labels_config)
        interim_root = ensure_dir(Path(config["target_interim_layout"]))
        records: list[DatasetRecord] = []
        label_files_found = 0
        split_counts = {"train": 0, "val": 0}
        class_counts: dict[str, int] = {}
        negatives = 0

        for split_name in ("train", "val"):
            image_dir = raw_root / "images" / split_name
            label_dir = raw_root / "labels" / split_name
            for image_path in sorted(path for path in image_dir.rglob("*") if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}):
                source_label = label_dir / f"{image_path.stem}.txt"
                target_label = canonical_label_path(interim_root, split_name, image_path)
                raw_classes: list[str] = []
                mapped_classes: list[str] = []
                target_lines: list[str] = []

                if source_label.exists():
                    label_files_found += 1
                    for line in source_label.read_text(encoding="utf-8").splitlines():
                        parts = line.strip().split()
                        if len(parts) < 5:
                            continue
                        try:
                            class_id = int(float(parts[0]))
                        except ValueError:
                            continue
                        raw_name = names_map.get(class_id, str(class_id))
                        mapped_name = map_raw_name(raw_name, labels_config)
                        if mapped_name in {None, "__negative__"}:
                            continue
                        if mapped_name not in detect_idx:
                            LOGGER.warning("custom_yolo class '%s' mapped to unsupported detect class; skipping", mapped_name)
                            continue
                        raw_classes.append(raw_name)
                        mapped_classes.append(mapped_name)
                        class_counts[mapped_name] = class_counts.get(mapped_name, 0) + 1
                        target_lines.append(" ".join([str(detect_idx[mapped_name]), *parts[1:5]]))

                target_label.write_text("\n".join(target_lines), encoding="utf-8")
                record = create_record(config["dataset_name"], config["task_type"], image_path, target_label, raw_classes, mapped_classes, split_name)
                if record.is_negative:
                    negatives += 1
                split_counts[split_name] = split_counts.get(split_name, 0) + 1
                records.append(record)
        if not records:
            LOGGER.warning("custom_yolo adapter found the expected directories but no usable image/label pairs.")
        self.debug_summary = {
            "dataset_name": self.dataset_name,
            "layout_recognized": valid,
            "layout_name": "images_labels_split_dirs",
            "data_present": len(records) > 0,
            "reason": None if records else "split directories exist but no usable image/label pairs were found",
            "names_map": names_map,
            "images_found": len(records),
            "label_files_found": label_files_found,
            "class_counts": class_counts,
            "empty_or_negative_samples": negatives,
            "split_counts": split_counts,
            "missing_paths": missing,
        }
        return records
