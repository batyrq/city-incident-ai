from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.data.manifest import build_dataset_stats
from src.train.anomaly import train_anomaly_branch
from src.train.detect import train_detect
from src.train.policy import save_training_decision, should_train_branch
from src.train.seg import train_seg
from src.utils.io import load_yaml, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one branch if policy gate allows it.")
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    manifest = pd.read_parquet(config["manifest_path"]) if Path(config["manifest_path"]).exists() else pd.DataFrame()
    task_df = manifest[manifest["task_type"] == config["task_type"]] if not manifest.empty else manifest
    stats = build_dataset_stats(task_df)
    decision = should_train_branch(
        branch_name=config["branch"],
        dataset_stats=stats,
        config=config,
        force_smoke_test=config.get("force_smoke_test", False),
        force_full_train=config.get("force_full_train", False),
    )
    save_training_decision(decision, Path(config["save_dir"]) / "policy_decision.json")
    if config["task_type"] == "detection":
        result = train_detect(config, decision)
    elif config["task_type"] == "segmentation":
        result = train_seg(config, decision)
    else:
        result = train_anomaly_branch(config, decision)
    write_json(result, Path(config["save_dir"]) / "result_summary.json")
    print(result)


if __name__ == "__main__":
    main()
