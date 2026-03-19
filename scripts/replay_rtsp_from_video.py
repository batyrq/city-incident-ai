from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Stub helper to replay a local video into an RTSP server.")
    parser.add_argument("--source", required=True)
    parser.add_argument("--target", default="rtsp://127.0.0.1:8554/live")
    args = parser.parse_args()
    print("Replay helper is intentionally a stub. Use ffmpeg or mediamtx locally, for example:")
    print(f"ffmpeg -re -stream_loop -1 -i {args.source} -c copy -f rtsp {args.target}")


if __name__ == "__main__":
    main()
