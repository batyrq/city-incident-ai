from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.schemas.common import DatasetRecord


@dataclass
class DatasetAdapter:
    dataset_name: str
    task_type: str
    access_mode: str
    can_auto_download: bool
    expected_raw_layout: list[str]
    target_interim_layout: str
    label_mapping_supported: bool
    notes: str

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "DatasetAdapter":
        return cls(
            dataset_name=config["dataset_name"],
            task_type=config["task_type"],
            access_mode=config["access_mode"],
            can_auto_download=config["can_auto_download"],
            expected_raw_layout=config.get("expected_raw_layout", []),
            target_interim_layout=config["target_interim_layout"],
            label_mapping_supported=config.get("label_mapping_supported", False),
            notes=config.get("notes", ""),
        )

    def expected_paths(self, raw_root: Path) -> list[Path]:
        return [raw_root / item for item in self.expected_raw_layout]

    def validate_raw(self, raw_root: Path) -> tuple[bool, list[str]]:
        missing = [str(path) for path in self.expected_paths(raw_root) if not path.exists()]
        return (len(missing) == 0, missing)

    def prepare(self, config: dict[str, Any], labels_config: dict[str, Any]) -> list[DatasetRecord]:
        valid, _ = self.validate_raw(Path(config["raw_root"]))
        if not valid:
            return []
        return []
