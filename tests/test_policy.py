from src.train.policy import should_train_branch


def test_detection_policy_uses_pretrained_when_data_is_tiny() -> None:
    decision = should_train_branch(
        "yolo11s_detect",
        {
            "has_train_split": True,
            "has_val_split": True,
            "images_total": 5,
            "positive_instances_total": 5,
            "positive_instances_per_class": {"fire": 5},
            "classes_present": ["fire"],
            "num_classes_present": 1,
            "val_samples": 1,
        },
        {"policy_path": "configs/training_policy.yaml"},
    )
    assert decision.decision == "use_pretrained_only"


def test_anomaly_policy_skips_without_normal_data() -> None:
    decision = should_train_branch(
        "efficientad_s",
        {
            "has_train_split": True,
            "has_val_split": True,
            "images_total": 20,
            "anomaly_normal_images": 0,
            "val_samples": 2,
        },
        {"policy_path": "configs/training_policy.yaml"},
    )
    assert decision.decision == "skip_branch"


def test_detection_policy_allows_full_train_when_thresholds_met() -> None:
    decision = should_train_branch(
        "yolo11s_detect",
        {
            "has_train_split": True,
            "has_val_split": True,
            "images_total": 200,
            "positive_instances_total": 320,
            "positive_instances_per_class": {"fire": 100, "smoke": 80},
            "classes_present": ["fire", "smoke"],
            "num_classes_present": 2,
            "val_samples": 25,
        },
        {"policy_path": "configs/training_policy.yaml"},
    )
    assert decision.decision == "train_full"
