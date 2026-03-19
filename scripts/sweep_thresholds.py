from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.io import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep inference thresholds for offline experiments.")
    parser.add_argument("--config", default="configs/inference_rtsp.yaml")
    args = parser.parse_args()
    config = load_yaml(args.config)
    for score, iou in itertools.product([0.25, 0.35, 0.5], [0.45, 0.5, 0.6]):
        print({"score_threshold": score, "iou_threshold": iou, "camera_count": len(config.get("streams", []))})


if __name__ == "__main__":
    main()
