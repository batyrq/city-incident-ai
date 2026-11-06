"""Download the open incident datasets (no login) into data/incidents/raw/. See data/incidents/SOURCES.md for sources and licences.

  python -m eval.incidents_download
"""
from __future__ import annotations

import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

RAW = Path("data/incidents/raw")
FILES = {
    "hanvithSai/gavin-dfire-raw": ["train.tar", "test.tar"],  # D-Fire mirror, CC BY 4.0
    "justjuu/traffic-accident-cctv-object-detection": ["data/train-00000-of-00002.parquet", "data/train-00001-of-00002.parquet",
                                                       "data/validation-00000-of-00001.parquet", "data/test-00000-of-00001.parquet"],  # CC0
    "zinnia82/flood-binary-hnm-benchmark": ["data/train-00000-of-00003.parquet", "data/train-00001-of-00003.parquet",
                                            "data/train-00002-of-00003.parquet", "data/val-00000-of-00001.parquet"],
}


def main() -> None:
    for repo, files in FILES.items():
        for f in files:
            p = hf_hub_download(repo_id=repo, filename=f, repo_type="dataset", local_dir=str(RAW / repo.replace("/", "__")))
            print("ok", repo, f, Path(p).stat().st_size // 10**6, "MB", flush=True)


if __name__ == "__main__":
    sys.exit(main())
