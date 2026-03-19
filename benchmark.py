from __future__ import annotations

import argparse

from src.benchmark.runner import build_leaderboard


def main() -> None:
    parser = argparse.ArgumentParser(description="Build benchmark leaderboard respecting policy gates.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset-stats", default="reports/dataset_stats.json")
    args = parser.parse_args()
    rows = build_leaderboard(args.config, args.dataset_stats)
    print(f"Leaderboard rows: {len(rows)}")


if __name__ == "__main__":
    main()
