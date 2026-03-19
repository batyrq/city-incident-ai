from __future__ import annotations

import logging
from pathlib import Path

from src.adapters.base import DatasetAdapter
from src.adapters.yolo_utils import (
    canonical_label_path,
    create_record,
    detect_name_to_idx,
    image_extensions,
    load_dataset_yaml,
    map_raw_name,
    resolve_names_map,
)
from src.schemas.common import DatasetRecord
from src.utils.io import ensure_dir


LOGGER = logging.getLogger(__name__)


class DFireAdapter(DatasetAdapter):
    def validate_raw(self, raw_root: Path) -> tuple[bool, list[str]]:
        image_dirs = [raw_root / "images", raw_root / "images" / "train", raw_root / "train" / "images"]
        label_dirs = [raw_root / "labels", raw_root / "labels" / "train", raw_root / "train" / "labels"]
        has_images = any(path.exists() for path in image_dirs)
        has_labels = any(path.exists() for path in label_dirs)
        missing: list[str] = []
        if not has_images:
            missing.append(str(raw_root / "images"))
        if not has_labels:
            missing.append(str(raw_root / "labels"))
        self.debug_summary = {
            "dataset_name": self.dataset_name,
            "layout_recognized": has_images and has_labels,
            "layout_name": "unknown",
            "data_present": False,
            "reason": "required directories missing" if missing else None,
            "missing_paths": missing,
        }
        return (has_images and has_labels, missing)

    def prepare(self, config: dict, labels_config: dict) -> list[DatasetRecord]:
        raw_root = Path(config["raw_root"])
        valid, missing = self.validate_raw(raw_root)
        if not valid:
            LOGGER.warning("DFire adapter skipped because dataset is missing: %s", missing)
            return []

        dataset_yaml = load_dataset_yaml(raw_root)
        names_map = resolve_names_map(dataset_yaml) or {0: "smoke", 1: "fire"}
        detect_idx = detect_name_to_idx(labels_config)
        interim_root = ensure_dir(Path(config["target_interim_layout"]))
        records: list[DatasetRecord] = []
        label_files_found = 0
        fire_positives = 0
        smoke_positives = 0
        negatives = 0
        split_counts = {"train": 0, "val": 0, "test": 0}
        layout_name, split_dirs = self._discover_layout(raw_root)

        for split_name, image_dir, label_dir in split_dirs:
            if not image_dir.exists():
                continue
            for image_path in sorted(path for path in image_dir.rglob("*") if path.suffix.lower() in image_extensions()):
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
                            LOGGER.warning("DFire class '%s' is unsupported for detect export; skipping label row.", mapped_name)
                            continue
                        raw_classes.append(raw_name)
                        mapped_classes.append(mapped_name)
                        if mapped_name == "fire":
                            fire_positives += 1
                        if mapped_name == "smoke":
                            smoke_positives += 1
                        target_lines.append(" ".join([str(detect_idx[mapped_name]), *parts[1:5]]))

                target_label.write_text("\n".join(target_lines), encoding="utf-8")
                record = create_record(config["dataset_name"], config["task_type"], image_path, target_label, raw_classes, mapped_classes, split_name)
                if record.is_negative:
                    negatives += 1
                split_counts[split_name] = split_counts.get(split_name, 0) + 1
                records.append(record)
        if not records:
            LOGGER.warning("DFire adapter found the expected directories but no usable images.")
        self.debug_summary = {
            "dataset_name": self.dataset_name,
            "layout_recognized": valid,
            "layout_name": layout_name,
            "data_present": len(records) > 0,
            "reason": None if records else "layout exists but no images were found under supported DFire folders",
            "names_map": names_map,
            "images_found": len(records),
            "label_files_found": label_files_found,
            "fire_positives": fire_positives,
            "smoke_positives": smoke_positives,
            "empty_or_negative_samples": negatives,
            "split_counts": split_counts,
            "missing_paths": missing,
        }
        return records

    def _discover_layout(self, raw_root: Path) -> tuple[str, list[tuple[str, Path, Path]]]:
        candidates = [
            ("train", raw_root / "images" / "train", raw_root / "labels" / "train"),
            ("val", raw_root / "images" / "val", raw_root / "labels" / "val"),
            ("test", raw_root / "images" / "test", raw_root / "labels" / "test"),
        ]
        alt_candidates = [
            ("train", raw_root / "train" / "images", raw_root / "train" / "labels"),
            ("val", raw_root / "val" / "images", raw_root / "val" / "labels"),
            ("test", raw_root / "test" / "images", raw_root / "test" / "labels"),
        ]
        available = [item for item in candidates if item[1].exists()]
        if available:
            return "images_split_dirs", available
        alt_available = [item for item in alt_candidates if item[1].exists()]
        if alt_available:
            return "root_split_dirs", alt_available
        return "flat_images_labels", [("train", raw_root / "images", raw_root / "labels")]
