from __future__ import annotations

import argparse

from src.eval.core import evaluate_checkpoint
from src.utils.io import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate checkpoint and export reports.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", "--ckpt", dest="checkpoint", default=None)
    args = parser.parse_args()
    config = load_yaml(args.config)
    result = evaluate_checkpoint(config, args.checkpoint)
    print(result)


if __name__ == "__main__":
    main()
