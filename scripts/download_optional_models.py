from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.env import has_package
from src.utils.io import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Opt-in download for optional pretrained assets.")
    parser.add_argument("--include-sam2", action="store_true", help="Attempt SAM2 checkpoint preparation note.")
    return parser.parse_args()


def main() -> None:
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    summary = {"ultralytics_installed": has_package("ultralytics"), "sam2_requested": False, "notes": []}
    if has_package("ultralytics"):
        summary["notes"].append("Ultralytics package available. Weights are usually auto-fetched lazily by the library.")
    else:
        summary["notes"].append("Ultralytics not installed; skipping optional weight prefetch.")
    args = parse_args()
    if args.include_sam2:
        summary["sam2_requested"] = True
        summary["notes"].append("SAM2 checkpoints often require manual selection/licensing/version control; no hardcoded download performed.")
    write_json(summary, models_dir / "download_summary.json")
    print(summary)


if __name__ == "__main__":
    main()
