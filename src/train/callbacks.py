from __future__ import annotations

from pathlib import Path
from typing import Any

from src.utils.io import ensure_dir, write_json


def save_training_stub_metrics(save_dir: str | Path, payload: dict[str, Any]) -> None:
    target = ensure_dir(Path(save_dir))
    write_json(payload, target / "metrics.json")
