from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from src.schemas.common import TrainingDecision
from src.train.callbacks import save_training_stub_metrics
from src.utils.env import has_package
from src.utils.io import dump_yaml, ensure_dir, write_json


def train_detect(config: dict[str, Any], decision: TrainingDecision) -> dict[str, Any]:
    save_dir = ensure_dir(Path(config["save_dir"]))
    result = {"branch": config["branch"], "task_type": "detection", "decision": decision.decision, "status": decision.status, "trained": False, "notes": []}
    if decision.decision in {"use_pretrained_only", "skip_branch"}:
        result["notes"].append("Training skipped by policy; use pretrained/stub path.")
        save_training_stub_metrics(save_dir, result)
        return result

    data_yaml_path = Path(config["data_yaml_path"])
    if not data_yaml_path.exists():
        result["notes"].append(f"Detect dataset yaml not found: {data_yaml_path}")
        save_training_stub_metrics(save_dir, result)
        return result
    if not has_package("ultralytics"):
        result["notes"].append("ultralytics is not installed; wrote stub metrics only.")
        save_training_stub_metrics(save_dir, result)
        return result

    from ultralytics import YOLO

    epochs = min(config["epochs"], 3) if decision.decision == "train_smoke_test" else config["epochs"]
    train_data_yaml = data_yaml_path
    if decision.decision == "train_smoke_test":
        result["notes"].append("Smoke-test mode active; not a full benchmark.")
        train_data_yaml = _build_smoke_subset_yaml(
            data_yaml_path=data_yaml_path,
            save_dir=save_dir,
            subset_fraction=float(config.get("smoke_test_subset_fraction", 0.1)),
        )

    model = YOLO(config["model_name"])
    train_output = model.train(
        data=str(train_data_yaml),
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
    result["artifacts"] = {
        "save_dir": str(save_dir),
        "best_ckpt": str(save_dir / "weights" / "best.pt"),
        "last_ckpt": str(save_dir / "weights" / "last.pt"),
    }
    result["train_output_type"] = type(train_output).__name__
    write_json(result, save_dir / "train_result.json")
    return result


def _build_smoke_subset_yaml(data_yaml_path: Path, save_dir: Path, subset_fraction: float) -> Path:
    with data_yaml_path.open("r", encoding="utf-8") as handle:
        data_yaml = yaml.safe_load(handle) or {}
    subset_fraction = min(max(subset_fraction, 0.01), 1.0)
    train_subset = _subset_txt(Path(data_yaml["train"]), save_dir / "smoke_train.txt", subset_fraction)
    val_subset = _subset_txt(Path(data_yaml["val"]), save_dir / "smoke_val.txt", max(subset_fraction, 0.25))
    smoke_yaml = dict(data_yaml)
    smoke_yaml["train"] = str(train_subset.resolve())
    smoke_yaml["val"] = str(val_subset.resolve())
    smoke_yaml["smoke_test"] = True
    target = save_dir / "smoke_dataset.yaml"
    dump_yaml(smoke_yaml, target)
    return target


def _subset_txt(source_txt: Path, output_txt: Path, fraction: float) -> Path:
    lines = [line.strip() for line in source_txt.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        output_txt.write_text("", encoding="utf-8")
        return output_txt
    keep = max(1, int(len(lines) * fraction))
    output_txt.write_text("\n".join(lines[:keep]), encoding="utf-8")
    return output_txt
