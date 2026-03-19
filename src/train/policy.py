from __future__ import annotations

from pathlib import Path
from typing import Any

from src.schemas.common import TrainingDecision
from src.utils.io import load_yaml, write_json


def _branch_task_type(branch_name: str) -> str:
    if "seg" in branch_name:
        return "segmentation"
    if branch_name in {"efficientad_s", "patchcore", "padim"} or "anomaly" in branch_name:
        return "anomaly"
    return "detection"


def should_train_branch(
    branch_name: str,
    dataset_stats: dict[str, Any],
    config: dict[str, Any],
    force_smoke_test: bool = False,
    force_full_train: bool = False,
) -> TrainingDecision:
    task_type = _branch_task_type(branch_name)
    policy = load_yaml(config["policy_path"])
    thresholds = policy[task_type]
    defaults = policy.get("defaults", {})
    reasons: list[str] = []

    if force_full_train and not defaults.get("allow_force_full_train", False):
        reasons.append("force_full_train requested but disabled by policy")
        force_full_train = False

    if not dataset_stats.get("has_train_split") or not dataset_stats.get("has_val_split"):
        decision = "use_pretrained_only" if task_type != "anomaly" else "skip_branch"
        reasons.append("valid train/val split not found")
        return TrainingDecision(
            branch=branch_name,
            decision=decision,
            reasons=reasons,
            stats_snapshot=dataset_stats,
            status="pretrained_only" if decision == "use_pretrained_only" else "skipped",
        )

    if force_smoke_test:
        reasons.append("force_smoke_test requested explicitly")
        return TrainingDecision(
            branch=branch_name,
            decision="train_smoke_test",
            reasons=reasons,
            stats_snapshot=dataset_stats,
            status="smoke_test_only",
        )

    if task_type == "detection":
        return _decide_detection(branch_name, dataset_stats, thresholds, force_full_train)
    if task_type == "segmentation":
        return _decide_segmentation(branch_name, dataset_stats, thresholds, force_full_train)
    return _decide_anomaly(branch_name, dataset_stats, thresholds, force_full_train)


def _decide_detection(branch_name: str, stats: dict[str, Any], thresholds: dict[str, Any], force_full_train: bool) -> TrainingDecision:
    reasons: list[str] = []
    classes_present = stats.get("classes_present", [])
    per_class = stats.get("positive_instances_per_class", {})
    rare_ok = sum(1 for value in per_class.values() if value >= thresholds["min_positive_instances_per_class"])
    full_ok = all(
        [
            stats.get("images_total", 0) >= thresholds["min_images_total"],
            stats.get("positive_instances_total", 0) >= thresholds["min_positive_instances_total"],
            stats.get("val_samples", 0) >= thresholds["min_val_samples"],
            stats.get("num_classes_present", 0) >= thresholds["min_classes_present"],
            rare_ok >= thresholds["min_classes_present"],
        ]
    )
    smoke_ok = all(
        [
            stats.get("images_total", 0) >= thresholds["smoke_min_images_total"],
            stats.get("positive_instances_total", 0) >= thresholds["smoke_min_positive_instances_total"],
            stats.get("val_samples", 0) >= thresholds["smoke_min_val_samples"],
        ]
    )
    if force_full_train and full_ok:
        reasons.extend(["manual force_full_train accepted", "thresholds still satisfied"])
        return TrainingDecision(branch=branch_name, decision="train_full", reasons=reasons, stats_snapshot=stats, status="trained")
    if full_ok:
        reasons.extend(["valid train/val split found", f"{len(classes_present)} classes present", "positive instances exceed threshold"])
        return TrainingDecision(branch=branch_name, decision="train_full", reasons=reasons, stats_snapshot=stats, status="trained")
    if smoke_ok:
        reasons.extend(["dataset exists but is below full-train thresholds", "running smoke-test only is feasible"])
        return TrainingDecision(branch=branch_name, decision="train_smoke_test", reasons=reasons, stats_snapshot=stats, status="smoke_test_only")
    reasons.extend(["insufficient detection data for honest fine-tuning", "pretrained detector is a safer default"])
    return TrainingDecision(branch=branch_name, decision="use_pretrained_only", reasons=reasons, stats_snapshot=stats, status="pretrained_only")


def _decide_segmentation(branch_name: str, stats: dict[str, Any], thresholds: dict[str, Any], force_full_train: bool) -> TrainingDecision:
    reasons: list[str] = []
    has_masks = stats.get("mask_images", 0) > 0
    full_ok = all(
        [
            has_masks,
            stats.get("images_total", 0) >= thresholds["min_images_total"],
            stats.get("mask_images", 0) >= thresholds["min_mask_images"],
            stats.get("val_samples", 0) >= thresholds["min_val_samples"],
        ]
    )
    smoke_ok = all(
        [
            has_masks,
            stats.get("images_total", 0) >= thresholds["smoke_min_images_total"],
            stats.get("mask_images", 0) >= thresholds["smoke_min_mask_images"],
            stats.get("val_samples", 0) >= thresholds["smoke_min_val_samples"],
        ]
    )
    if force_full_train and full_ok:
        reasons.append("manual force_full_train accepted")
        return TrainingDecision(branch=branch_name, decision="train_full", reasons=reasons, stats_snapshot=stats, status="trained")
    if full_ok:
        reasons.extend(["mask labels available in sufficient quantity", "segmentation split looks valid"])
        return TrainingDecision(branch=branch_name, decision="train_full", reasons=reasons, stats_snapshot=stats, status="trained")
    if smoke_ok:
        reasons.append("mask labels exist but volume is small")
        return TrainingDecision(branch=branch_name, decision="train_smoke_test", reasons=reasons, stats_snapshot=stats, status="smoke_test_only")
    if not has_masks:
        reasons.append("bbox-only data detected; segmentation training disabled")
        return TrainingDecision(branch=branch_name, decision="use_pretrained_only", reasons=reasons, stats_snapshot=stats, status="pretrained_only")
    reasons.append("segmentation data insufficient")
    return TrainingDecision(branch=branch_name, decision="skip_branch", reasons=reasons, stats_snapshot=stats, status="skipped")


def _decide_anomaly(branch_name: str, stats: dict[str, Any], thresholds: dict[str, Any], force_full_train: bool) -> TrainingDecision:
    reasons: list[str] = []
    full_ok = all(
        [
            stats.get("images_total", 0) >= thresholds["min_images_total"],
            stats.get("anomaly_normal_images", 0) >= thresholds["min_anomaly_normal_images"],
            stats.get("val_samples", 0) >= thresholds["min_val_samples"],
        ]
    )
    smoke_ok = all([stats.get("images_total", 0) >= thresholds["smoke_min_images_total"], stats.get("val_samples", 0) >= thresholds["smoke_min_val_samples"]])
    if force_full_train and full_ok:
        reasons.append("manual force_full_train accepted")
        return TrainingDecision(branch=branch_name, decision="train_full", reasons=reasons, stats_snapshot=stats, status="trained")
    if full_ok:
        reasons.extend(["normal-only anomaly dataset is sufficient", "ROI-ready split is present"])
        return TrainingDecision(branch=branch_name, decision="train_full", reasons=reasons, stats_snapshot=stats, status="trained")
    if smoke_ok:
        reasons.append("only anomaly smoke-test is justified")
        return TrainingDecision(branch=branch_name, decision="train_smoke_test", reasons=reasons, stats_snapshot=stats, status="smoke_test_only")
    reasons.extend(["anomaly-ready dataset absent or too small", "creating skeleton only; fake training is blocked"])
    return TrainingDecision(branch=branch_name, decision="skip_branch", reasons=reasons, stats_snapshot=stats, status="skipped")


def save_training_decision(decision: TrainingDecision, output_path: str | Path) -> None:
    write_json(decision.model_dump(), output_path)
