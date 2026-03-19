from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.train.policy import should_train_branch
from src.utils.io import load_yaml, read_json, write_json


def build_leaderboard(matrix_config_path: str, dataset_stats_path: str = "reports/dataset_stats.json") -> list[dict[str, Any]]:
    matrix = load_yaml(matrix_config_path)
    stats = read_json(dataset_stats_path) if Path(dataset_stats_path).exists() else {}
    policy_path = "configs/training_policy.yaml"
    rows: list[dict[str, Any]] = []
    for section in ("detection", "segmentation", "anomaly"):
        for item in matrix.get(section, []):
            decision = should_train_branch(item["branch"], stats.get(section, stats), {"policy_path": policy_path})
            rows.append(
                {
                    "branch": item["branch"],
                    "task_type": item["task_type"],
                    "model_name": item["model_name"],
                    "status": decision.status,
                    "decision": decision.decision,
                    "reasons": " | ".join(decision.reasons),
                    "trained_result_path": str(Path("runs") / "train" / item["branch"] / "train_result.json"),
                }
            )
    Path("reports").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv("reports/leaderboard.csv", index=False)
    write_json(rows, "reports/leaderboard.json")
    return rows
