from __future__ import annotations

import base64
from pathlib import Path

from src.adapters.custom_yolo import CustomYoloAdapter
from src.adapters.dfire import DFireAdapter
from src.utils.io import load_yaml


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    png_bytes = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO7Z0KsAAAAASUVORK5CYII="
    )
    path.write_bytes(png_bytes)


def test_custom_yolo_adapter_without_dataset_yaml_uses_canonical_order(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw" / "custom_yolo"
    for split in ("train", "val"):
        _write_image(raw_root / "images" / split / f"{split}_sample.png")
        (raw_root / "labels" / split).mkdir(parents=True, exist_ok=True)
        (raw_root / "labels" / split / f"{split}_sample.txt").write_text("0 0.5 0.5 0.25 0.25\n", encoding="utf-8")

    adapter = CustomYoloAdapter.from_config(
        {
            "dataset_name": "custom_yolo",
            "task_type": "detection",
            "access_mode": "local_only",
            "can_auto_download": False,
            "expected_raw_layout": [],
            "target_interim_layout": str(tmp_path / "interim" / "custom_yolo"),
            "label_mapping_supported": True,
            "notes": "",
            "raw_root": str(raw_root),
        }
    )
    records = adapter.prepare({"dataset_name": "custom_yolo", "task_type": "detection", "raw_root": str(raw_root), "target_interim_layout": str(tmp_path / "interim" / "custom_yolo")}, load_yaml("configs/labels.yaml"))
    assert len(records) == 2
    assert records[0].classes_mapped == ["fire"]
    assert Path(records[0].label_path).exists()


def test_dfire_adapter_maps_fire_smoke_and_negative(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw" / "dfire"
    _write_image(raw_root / "images" / "train" / "fire.png")
    _write_image(raw_root / "images" / "val" / "negative.png")
    (raw_root / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (raw_root / "labels" / "val").mkdir(parents=True, exist_ok=True)
    (raw_root / "labels" / "train" / "fire.txt").write_text("1 0.5 0.5 0.5 0.5\n0 0.4 0.4 0.2 0.2\n", encoding="utf-8")
    (raw_root / "dataset.yaml").write_text("names: ['smoke', 'fire']\n", encoding="utf-8")

    adapter = DFireAdapter.from_config(
        {
            "dataset_name": "dfire",
            "task_type": "detection",
            "access_mode": "manual_download",
            "can_auto_download": False,
            "expected_raw_layout": [],
            "target_interim_layout": str(tmp_path / "interim" / "dfire"),
            "label_mapping_supported": True,
            "notes": "",
            "raw_root": str(raw_root),
        }
    )
    records = adapter.prepare({"dataset_name": "dfire", "task_type": "detection", "raw_root": str(raw_root), "target_interim_layout": str(tmp_path / "interim" / "dfire")}, load_yaml("configs/labels.yaml"))
    assert len(records) == 2
    assert sorted(records[0].classes_mapped) == ["fire", "smoke"]
    negative_record = next(record for record in records if record.image_path.endswith("negative.png"))
    assert negative_record.is_negative is True
