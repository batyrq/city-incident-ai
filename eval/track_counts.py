"""Unique track IDs per dominant class from a run's predictions.jsonl -> JSON."""
from __future__ import annotations
import argparse, collections, json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("predictions", nargs="+", help="predictions.jsonl paths")
    ap.add_argument("--out", default="reports/metrics/track_counts.json")
    args = ap.parse_args()
    out = {}
    for path in args.predictions:
        tracks = collections.defaultdict(collections.Counter)
        for line in open(path):
            row = json.loads(line)
            for det in row["detections"]:
                if det.get("track_id") is not None:
                    tracks[int(det["track_id"])][det["class_name"]] += 1
        by_class = collections.Counter(c.most_common(1)[0][0] for c in tracks.values())
        out[path] = {"unique_tracks": len(tracks), "max_id": max(tracks) if tracks else 0, "by_dominant_class": dict(by_class)}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
