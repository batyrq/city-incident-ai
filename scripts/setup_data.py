from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.adapters.registry import get_adapter
from src.data.export_ultralytics_detect import export_ultralytics_detect_dataset
from src.data.manifest import build_dataset_stats, export_yolo_data_files, records_to_dataframe, save_manifest
from src.data.splits import apply_splits
from src.train.policy import should_train_branch
from src.utils.io import load_yaml, write_json
from src.utils.logging import setup_logging


LOGGER = logging.getLogger("setup_data")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare datasets, manifest, stats and training plan.")
    parser.add_argument("--config", required=True, help="Path to datasets YAML config.")
    parser.add_argument("--labels", default="configs/labels.yaml", help="Path to labels mapping YAML.")
    parser.add_argument("--policy", default="configs/training_policy.yaml", help="Path to training policy YAML.")
    return parser.parse_args()


def append_manual_instructions(dataset_cfg: dict[str, Any], present: bool) -> str:
    expected_layout = "\n".join(f"  - {item}" for item in dataset_cfg.get("expected_raw_layout", []))
    extra = ""
    if dataset_cfg["dataset_name"] == "dfire":
        extra = (
            "- supported_layout_examples:\n"
            "  - datasets/raw/dfire/images/train + labels/train\n"
            "  - datasets/raw/dfire/train/images + train/labels\n"
            "- required_files: image files plus matching YOLO txt labels when positives exist\n"
            "- supported_splits: train, val, optional test\n"
        )
    if dataset_cfg["dataset_name"] == "custom_yolo":
        extra = (
            "- supported_layout_examples:\n"
            "  - datasets/raw/custom_yolo/images/train + labels/train\n"
            "  - datasets/raw/custom_yolo/images/val + labels/val\n"
            "- dataset_yaml: optional\n"
            "- required_files: image files and YOLO txt labels in mirrored split folders\n"
        )
    return (
        f"## {dataset_cfg['dataset_name']}\n"
        f"- access_mode: {dataset_cfg['access_mode']}\n"
        f"- present_locally: {'yes' if present else 'no'}\n"
        f"- raw_root: `{dataset_cfg['raw_root']}`\n"
        f"- expected_layout:\n{expected_layout}\n"
        f"{extra}"
        f"- notes: {dataset_cfg.get('notes', '')}\n"
    )


def enrich_training_decision(branch: str, decision_payload: dict[str, Any], stats: dict[str, Any], policy_config: dict[str, Any]) -> dict[str, Any]:
    enriched = dict(decision_payload)
    task_name = "segmentation" if "seg" in branch else "anomaly" if branch in {"efficientad_s", "patchcore"} else "detection"
    thresholds = policy_config.get(task_name, {})
    failing_checks: list[str] = []
    if task_name == "detection":
        checks = [
            ("has_train_split", bool(stats.get("has_train_split")), True),
            ("has_val_split", bool(stats.get("has_val_split")), True),
            ("min_images_total", stats.get("images_total", 0), thresholds.get("min_images_total")),
            ("min_positive_instances_total", stats.get("positive_instances_total", 0), thresholds.get("min_positive_instances_total")),
            ("min_val_samples", stats.get("val_samples", 0), thresholds.get("min_val_samples")),
            ("min_classes_present", stats.get("num_classes_present", 0), thresholds.get("min_classes_present")),
        ]
        per_class = stats.get("positive_instances_per_class", {})
        rare_ok = sum(1 for value in per_class.values() if value >= thresholds.get("min_positive_instances_per_class", 0))
        checks.append(("min_positive_instances_per_class_classes", rare_ok, thresholds.get("min_classes_present")))
        for name, actual, expected in checks:
            if expected is True and actual is not True:
                failing_checks.append(name)
            elif isinstance(expected, int) and actual < expected:
                failing_checks.append(name)
    enriched["thresholds_snapshot"] = thresholds
    enriched["first_failing_gate"] = failing_checks[0] if failing_checks else None
    enriched["failing_checks"] = failing_checks
    return enriched


def ensure_expected_layout(dataset_cfg: dict[str, Any]) -> None:
    raw_root = Path(dataset_cfg["raw_root"])
    raw_root.mkdir(parents=True, exist_ok=True)
    for item in dataset_cfg.get("expected_raw_layout", []):
        target = raw_root / item
        if str(item).endswith("/"):
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    setup_logging()
    config = load_yaml(args.config)
    labels_config = load_yaml(args.labels)
    policy_config = load_yaml(args.policy)

    all_records = []
    manual_sections = [
        "# Manual Download",
        "",
        "Generated by `scripts/setup_data.py`.",
        "",
        "Datasets that require login, forms, Kaggle, Google Drive or manual access are listed below.",
        "",
    ]
    dataset_reports: list[dict[str, Any]] = []

    for dataset_cfg in config.get("datasets", []):
        if not dataset_cfg.get("enabled", True):
            continue
        adapter = get_adapter(dataset_cfg)
        raw_root = Path(dataset_cfg["raw_root"])
        if dataset_cfg["access_mode"] in {"manual_download", "local_only"}:
            ensure_expected_layout(dataset_cfg)
        present, missing = adapter.validate_raw(raw_root)
        if dataset_cfg["access_mode"] != "auto_download":
            manual_sections.append(append_manual_instructions(dataset_cfg, present))
        if dataset_cfg["access_mode"] == "auto_download" and not dataset_cfg.get("can_auto_download", False):
            LOGGER.warning("Dataset %s declared auto_download without stable implementation; skipping download.", dataset_cfg["dataset_name"])
        if not present:
            LOGGER.warning("Dataset %s is missing expected files: %s", dataset_cfg["dataset_name"], missing)
        records = adapter.prepare(dataset_cfg, labels_config)
        debug_summary = getattr(adapter, "debug_summary", None)
        if debug_summary:
            write_json(debug_summary, Path("reports") / f"{dataset_cfg['dataset_name']}_adapter_debug.json")
        dataset_reports.append(
            {
                "dataset_name": dataset_cfg["dataset_name"],
                "task_type": dataset_cfg["task_type"],
                "access_mode": dataset_cfg["access_mode"],
                "present": present,
                "missing": missing,
                "records_prepared": len(records),
                "debug_report": str(Path("reports") / f"{dataset_cfg['dataset_name']}_adapter_debug.json") if debug_summary else None,
            }
        )
        all_records.extend(records)

    Path("docs/MANUAL_DOWNLOAD.md").parent.mkdir(parents=True, exist_ok=True)
    Path("docs/MANUAL_DOWNLOAD.md").write_text("\n".join(manual_sections), encoding="utf-8")

    df = records_to_dataframe(all_records)
    if not df.empty:
        df = apply_splits(df)
    save_manifest(df, Path("datasets/unified/manifest.parquet"))
    export_yolo_data_files(df, Path("datasets/unified"))
    detect_export_summary = export_ultralytics_detect_dataset(df, Path("datasets/unified/detect_yolo"), labels_config)

    overall_stats = build_dataset_stats(df)
    stats_by_task: dict[str, Any] = {"overall": overall_stats}
    for task_name in ("detection", "segmentation", "anomaly"):
        stats_by_task[task_name] = build_dataset_stats(df[df["task_type"] == task_name]) if not df.empty else build_dataset_stats(pd.DataFrame())
    write_json(stats_by_task, "reports/dataset_stats.json")
    write_json(dataset_reports, "reports/dataset_reports.json")
    write_json(detect_export_summary, "reports/detect_export_summary.json")

    plan = []
    for branch in ["yolo11n_detect", "yolo11s_detect", "yolo11m_detect", "rtdetr_l_detect", "yolo11n_seg", "yolo11s_seg", "efficientad_s", "patchcore"]:
        task_stats = stats_by_task["segmentation" if "seg" in branch else "anomaly" if branch in {"efficientad_s", "patchcore"} else "detection"]
        decision = should_train_branch(branch, task_stats, {"policy_path": args.policy})
        plan.append(enrich_training_decision(branch, decision.model_dump(), task_stats, policy_config))
    write_json(plan, "reports/training_plan.json")


if __name__ == "__main__":
    main()
