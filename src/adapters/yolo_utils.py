from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

import yaml

from src.schemas.common import DatasetRecord
from src.utils.io import ensure_dir


LOGGER = logging.getLogger(__name__)

try:
    import cv2
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    cv2 = None


def load_dataset_yaml(raw_root: Path) -> dict[str, Any]:
    for candidate in (raw_root / "dataset.yaml", raw_root / "data.yaml"):
        if candidate.exists():
            with candidate.open("r", encoding="utf-8") as handle:
                return yaml.safe_load(handle) or {}
    return {}


def resolve_names_map(dataset_yaml: dict[str, Any]) -> dict[int, str]:
    names = dataset_yaml.get("names", {})
    if isinstance(names, list):
        return {idx: str(name) for idx, name in enumerate(names)}
    if isinstance(names, dict):
        return {int(idx): str(name) for idx, name in names.items()}
    return {}


def build_mapping_lookup(labels_config: dict[str, Any]) -> dict[str, str]:
    mapping = {str(key).strip().lower(): str(value).strip() for key, value in labels_config.get("mapping", {}).items()}
    for item in labels_config.get("hard_negative_aliases", []):
        mapping[str(item).strip().lower()] = "__hard_negative__"
    for item in labels_config.get("ignore_aliases", []):
        mapping[str(item).strip().lower()] = "__ignore__"
    mapping["none"] = "__negative__"
    mapping["negative"] = "__negative__"
    return mapping


def detect_class_names(labels_config: dict[str, Any]) -> list[str]:
    return list(labels_config.get("detect_classes", ["fire", "smoke", "flood_water", "traffic_incident"]))


def detect_name_to_idx(labels_config: dict[str, Any]) -> dict[str, int]:
    return {name: idx for idx, name in enumerate(detect_class_names(labels_config))}


def map_raw_name(raw_name: str, labels_config: dict[str, Any]) -> str | None:
    lookup = build_mapping_lookup(labels_config)
    normalized = raw_name.strip().lower()
    mapped = lookup.get(normalized, normalized)
    if mapped in {"__ignore__", "__hard_negative__"}:
        return None
    if mapped == "__negative__":
        return "__negative__"
    return mapped


def image_size(image_path: Path) -> tuple[int | None, int | None]:
    if cv2 is None:
        return None, None
    image = cv2.imread(str(image_path))
    if image is None:
        return None, None
    height, width = image.shape[:2]
    return width, height


def canonical_label_path(interim_root: Path, split_name: str, image_path: Path) -> Path:
    target = interim_root / "labels" / split_name / f"{image_path.stem}.txt"
    ensure_dir(target.parent)
    return target


def image_extensions() -> tuple[str, ...]:
    return (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def copy_or_link(src: Path, dst: Path, use_symlink: bool = True) -> str:
    ensure_dir(dst.parent)
    if dst.exists():
        return "existing"
    if use_symlink:
        try:
            dst.symlink_to(src.resolve())
            return "symlink"
        except OSError:
            pass
    shutil.copy2(src, dst)
    return "copy"


def create_record(
    dataset_name: str,
    task_type: str,
    image_path: Path,
    label_path: Path,
    raw_classes: list[str],
    mapped_classes: list[str],
    split_name: str,
) -> DatasetRecord:
    width, height = image_size(image_path)
    return DatasetRecord(
        sample_id=f"{dataset_name}:{split_name}:{image_path.stem}",
        source_dataset=dataset_name,
        task_type=task_type,
        image_path=str(image_path),
        label_path=str(label_path),
        width=width,
        height=height,
        classes_raw=raw_classes,
        classes_mapped=mapped_classes,
        is_negative=len(mapped_classes) == 0,
        split_group=split_name,
    )
