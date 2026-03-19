from __future__ import annotations

from pathlib import Path
from typing import Any

from src.schemas.common import TrainingDecision
from src.train.callbacks import save_training_stub_metrics
from src.utils.env import has_package
from src.utils.io import ensure_dir, write_json


def train_seg(config: dict[str, Any], decision: TrainingDecision) -> dict[str, Any]:
    save_dir = ensure_dir(Path(config["save_dir"]))
    result = {"branch": config["branch"], "task_type": "segmentation", "decision": decision.decision, "status": decision.status, "trained": False, "notes": []}
    if decision.decision in {"use_pretrained_only", "skip_branch"}:
        result["notes"].append("Segmentation training not launched.")
        save_training_stub_metrics(save_dir, result)
        return result
    if not has_package("ultralytics"):
        result["notes"].append("ultralytics is not installed; wrote stub metrics only.")
        save_training_stub_metrics(save_dir, result)
        return result
    from ultralytics import YOLO

    epochs = min(config["epochs"], 3) if decision.decision == "train_smoke_test" else config["epochs"]
    model = YOLO(config["model_name"])
    model.train(
        data=config["data_yaml_path"],
        imgsz=config["imgsz"],
        epochs=epochs,
        project=str(save_dir.parent),
        name=save_dir.name,
        patience=config.get("patience", 20),
        batch=config.get("batch", -1),
        device=config.get("device", "auto"),
        amp=config.get("amp", True),
        resume=config.get("resume", False),
    )
    result["trained"] = True
    write_json(result, save_dir / "train_result.json")
    return result
