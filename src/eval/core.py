from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.env import has_package
from src.utils.io import ensure_dir, write_json


def evaluate_checkpoint(config: dict, checkpoint: str | None = None) -> dict:
    start = time.perf_counter()
    result = {
        "branch": config.get("branch"),
        "checkpoint": checkpoint,
        "object_metrics": {"mAP50": None, "mAP50_95": None, "precision": None, "recall": None, "per_class": {}},
        "video_alert_metrics": {
            "event_precision": None,
            "event_recall": None,
            "false_alerts_per_camera_hour": None,
            "detection_delay": None,
            "duplicate_alert_rate": None,
        },
        "speed_metrics": {"fps": None, "mean_latency_ms": None, "p95_latency_ms": None, "gpu_memory_mb": None, "throughput_batch1": None, "per_stream_estimate": None},
        "notes": [],
    }
    checkpoint_path = Path(checkpoint) if checkpoint else None
    if config.get("task_type") != "detection":
        result["notes"].append("Real eval is only wired for detection in this iteration.")
    elif not checkpoint_path or not checkpoint_path.exists():
        result["notes"].append("Checkpoint file is missing; evaluation skipped honestly.")
    elif not Path(config.get("data_yaml_path", "")).exists():
        result["notes"].append("Dataset yaml for detection eval is missing.")
    elif not has_package("ultralytics"):
        result["notes"].append("Ultralytics is not installed; evaluation skipped.")
    else:
        from ultralytics import YOLO

        model = YOLO(str(checkpoint_path))
        metrics = model.val(
            data=config["data_yaml_path"],
            imgsz=config.get("imgsz", 640),
            batch=config.get("batch", 1),
            device=config.get("device", "auto"),
            split="val",
            plots=True,
        )
        result["raw_metrics"] = _coerce_metrics(metrics)
        result["object_metrics"] = _parse_detect_metrics(metrics)
        result["notes"].append("Ultralytics detection validation completed.")
    if "raw_metrics" not in result:
        result["notes"].append("No runnable checkpoint evaluation was available; returning stub report without fabricated metrics.")
    result["runtime_seconds"] = round(time.perf_counter() - start, 3)
    ensure_dir(Path("reports"))
    write_json(result, Path("reports") / f"eval_{config.get('branch', 'unknown')}.json")
    pd.DataFrame([result]).to_csv(Path("reports") / f"eval_{config.get('branch', 'unknown')}.csv", index=False)
    return result


def _coerce_metrics(metrics: Any) -> dict[str, Any]:
    if hasattr(metrics, "results_dict"):
        try:
            return {str(key): float(value) if isinstance(value, (int, float)) else value for key, value in metrics.results_dict.items()}
        except Exception:
            return {"results_dict_repr": repr(metrics.results_dict)}
    return {"repr": repr(metrics)}


def _parse_detect_metrics(metrics: Any) -> dict[str, Any]:
    box = getattr(metrics, "box", None)
    parsed = {
        "mAP50": getattr(box, "map50", None),
        "mAP50_95": getattr(box, "map", None),
        "precision": getattr(box, "mp", None),
        "recall": getattr(box, "mr", None),
        "per_class": {},
    }
    names = getattr(metrics, "names", {}) or {}
    if box is not None and hasattr(box, "maps"):
        maps = list(getattr(box, "maps", []))
        parsed["per_class"] = {str(names.get(idx, idx)): score for idx, score in enumerate(maps)}
    return parsed
