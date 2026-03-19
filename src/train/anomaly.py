from __future__ import annotations

from pathlib import Path
from typing import Any

from src.schemas.common import TrainingDecision
from src.train.callbacks import save_training_stub_metrics
from src.utils.env import has_package
from src.utils.io import ensure_dir, write_json


def train_anomaly_branch(config: dict[str, Any], decision: TrainingDecision) -> dict[str, Any]:
    save_dir = ensure_dir(Path(config["save_dir"]))
    result = {"branch": config["branch"], "task_type": "anomaly", "decision": decision.decision, "status": decision.status, "trained": False, "notes": []}
    if decision.decision == "skip_branch":
        result["notes"].append("Anomaly branch skipped because anomaly-ready data is missing or too small.")
        save_training_stub_metrics(save_dir, result)
        return result
    if not has_package("anomalib"):
        result["notes"].append("anomalib is not installed; saved stub only.")
        save_training_stub_metrics(save_dir, result)
        return result
    result["notes"].append("Anomalib integration point is ready; dataset-specific datamodule is still required.")
    write_json(result, save_dir / "train_result.json")
    return result
