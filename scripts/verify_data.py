from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.validation import verify_manifest
from src.utils.io import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify prepared unified manifest.")
    parser.add_argument("--manifest", required=True, help="Path to manifest parquet.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_parquet(args.manifest)
    report = verify_manifest(df)
    write_json(report, "reports/data_verification.json")
    print("Data verification summary")
    for key, value in report.items():
        print(f"- {key}: {value}")


if __name__ == "__main__":
    main()
