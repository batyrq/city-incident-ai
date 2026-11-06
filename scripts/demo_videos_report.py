"""Run the pipeline on every clip in demo_videos/ (recommended settings) and write the results table.

  python scripts/demo_videos_report.py     -> reports/metrics/demo_videos.json + demo_videos/README.md

All numbers in the README come from these runs (summary + events JSON written by demo.run_demo), nothing is typed by hand.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from demo import run_demo  # noqa: E402
from demo_presets import DIR, LICENCE  # noqa: E402
from demo_presets import PRESETS as ALL_PRESETS  # noqa: E402

PRESETS = [p for p in ALL_PRESETS if "/" not in p["file"]]  # the six street clips; incident clips have their own report


def probe(path: Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    fps, n = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    diffs, prev = [], None
    while len(diffs) < 400:  # mean absolute frame-to-frame change: time-lapses score high on this
        ok, fr = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(cv2.resize(fr, (320, 180)), cv2.COLOR_BGR2GRAY).astype(np.float32)
        if prev is not None:
            diffs.append(float(np.abs(g - prev).mean()))
        prev = g
    return {"width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), "fps": round(fps, 2),
            "seconds": round(n / fps, 1), "size_mb": round(path.stat().st_size / 1e6, 2), "mean_frame_diff": round(float(np.mean(diffs)), 2)}


def main() -> None:
    if "--render-only" in sys.argv:  # re-render README from the saved JSON without re-running the pipeline
        saved = json.loads((ROOT / "reports/metrics/demo_videos.json").read_text())
        notes = {p["id"]: p["note"] for p in PRESETS}
        rows = [{**r, "note": notes[r["id"]]} for r in saved["clips"]]
        (ROOT / DIR / "README.md").write_text(render(rows, saved["timelapse_reference_frame_diff"]))
        return
    rows = []
    for p in PRESETS:
        path = ROOT / DIR / p["file"]
        info = probe(path)
        result, _ = run_demo(str(path), pose=p["pose"], max_frames=p["max_frames"] or None, run_suffix="_demoset")
        events = json.loads((ROOT / "reports" / f"{result['run_name']}_events.json").read_text())
        rows.append({**{k: p[k] for k in ("id", "file", "label", "pose", "max_frames", "shows", "note", "source", "origin")}, **info,
                     "frames_processed": result["frames_processed"], "unique_tracks": result.get("tracked_objects_total"),
                     "classes_seen": result.get("classes_seen"), "pipeline_fps": result.get("pipeline_fps"), "device": result["device"],
                     "events": [{"type": e["event_type"], "start_s": e["start_time_sec"], "tracks": len(e.get("related_track_ids", []))} for e in events]})
        print(p["id"], rows[-1]["events"], rows[-1]["unique_tracks"], rows[-1]["pipeline_fps"], flush=True)
    ref = ROOT / "data/videos/mixkit-4251.mp4"  # the time-lapse rejected earlier, as a reference for the frame-diff measure
    ref_diff = probe(ref)["mean_frame_diff"] if ref.exists() else None
    (ROOT / "reports/metrics/demo_videos.json").write_text(json.dumps({"timelapse_reference_frame_diff": ref_diff, "clips": rows}, indent=2))
    (ROOT / DIR / "README.md").write_text(render(rows, ref_diff))
    print("wrote", DIR + "/README.md")


def render(rows: list[dict], ref_diff: float | None) -> str:
    def ev(r):
        return "; ".join(f"`{e['type']}` at {e['start_s']} s ({e['tracks']} tracks)" for e in r["events"]) or "none fired"

    lines = ["# Demo videos", "",
             "Six short clips for uploading into the **Try it live** section (`make run`, http://localhost:8000). All are free Mixkit videos "
             f"({LICENCE}), not time-lapses (see the check below), trimmed to at most 20 s, 720p H.264, no audio.",
             "Pick one from the dropdown in the live demo (it also sets the recommended pose switch), or upload the file from this folder.", "",
             "| File | What it shows | Recommended settings | Result of the real run | Source / licence |", "|---|---|---|---|---|"]
    for r in rows:
        settings = ("pose **on**" if r["pose"] else "pose off") + ", " + ("whole clip" if not r["max_frames"] else f"max frames {r['max_frames']}")
        res = (f"{ev(r)}. {r['unique_tracks']} unique track{'' if r['unique_tracks'] == 1 else 's'}, {r['frames_processed']} frames, {r['pipeline_fps']:.0f} FPS "
               f"(whole pipeline, {r['device']}). Classes: {', '.join(r['classes_seen'])}.")
        lines.append(f"| `{r['file']}` ({r['seconds']} s, {r['size_mb']} MB) | {r['shows']}{(' **Note:** ' + r['note']) if r['note'] else ''} | {settings} | {res} | [{r['origin']}]({r['source']}) |")
    lines += ["", "## Notes", "",
              "- Results are produced by `python scripts/demo_videos_report.py` (default config `configs/infer_pretrained_yolo11s.yaml`, device auto, "
              "so MPS with `yolo11s.pt` on this Mac); FPS is end-to-end pipeline FPS of that run and varies with the machine and with pose on/off. "
              "The raw numbers are in `reports/metrics/demo_videos.json`.",
              "- Time-lapse check (it rules out time-lapses, not slow motion): mean absolute frame-to-frame change (0-255 grey levels, first 400 frames): "
              + ", ".join(f"{r['id']} {r['mean_frame_diff']}" for r in rows)
              + (f". For reference, the time-lapse rejected earlier (`data/videos/mixkit-4251.mp4`) scores {ref_diff} on the same measure." if ref_diff else "."),
              "- Rule thresholds were not changed for any clip (`configs/infer_pretrained_yolo11s.yaml`: `stopped_vehicle` needs 15 s without motion, "
              "`crowding` needs 12 people).",
              "- Mixkit's licence allows free commercial and non-commercial use without attribution; the source page of each clip is linked above.", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
