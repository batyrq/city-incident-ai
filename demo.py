"""One-command demo: annotated H.264 video (boxes + track IDs, trails, event banner, FPS/latency HUD, optional pose).

  python demo.py --source data/videos/mixkit-61.mp4 [--pose] [--max-frames N] [--device mps|cpu] [--out out.mp4]

Uses the normal pipeline (src.infer.runtime.run_demo_video); only the drawing overlay (src/infer/overlay.py) and the
H.264 re-encode are added. Defaults to yolo11s.pt on MPS when available, otherwise the repo's yolo11s.onnx on CPU.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from src.infer.runtime import run_demo_video
from src.utils.io import load_yaml, write_json


def pick_backend(device: str | None, weights: str | None) -> tuple[str, str]:
    if device and weights:
        return device, weights
    try:
        import torch
        mps = bool(torch.backends.mps.is_available())
    except Exception:
        mps = False
    if device == "mps" or (device is None and mps):
        w = weights or "yolo11s.pt"
        if mps and Path(w).exists():
            return "mps", w
        if device == "mps":
            sys.exit(f"--device mps requested but MPS or {w} is unavailable")
    return device or "cpu", weights or "yolo11s.onnx"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True)
    ap.add_argument("--config", default="configs/infer_pretrained_yolo11s.yaml")
    ap.add_argument("--out", default=None, help="H.264 output path (default outputs/<clip>_demo.mp4)")
    ap.add_argument("--device", default=None, help="mps or cpu (default: mps + yolo11s.pt if available, else cpu + yolo11s.onnx)")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--pose", action="store_true", help="overlay yolo11n-pose skeletons on people")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--classes", default=None, help="comma-separated class names, or 'all'")
    args = ap.parse_args()

    device, weights = pick_backend(args.device, args.weights)
    cfg = load_yaml(args.config)
    stem = Path(args.source).stem + ("_pose" if args.pose else "")
    cfg["run_name"] = f"demo_{stem}"
    cfg["report_path"] = f"reports/demo_{stem}_summary.json"
    cfg["model_weights"] = weights
    cfg.setdefault("runtime", {})["device"] = device
    if args.max_frames is not None:
        cfg["runtime"]["max_frames"] = args.max_frames
    if args.classes:
        cfg["classes"] = [c.strip() for c in args.classes.split(",") if c.strip()]
    cfg["viz"] = {"enabled": True, "pose": args.pose, "trail_seconds": 1.0, "banner_seconds": 2.0}
    print(f"device={device} weights={weights} pose={args.pose}")

    result = run_demo_video(args.source, cfg, model_path=weights)
    write_json(result, Path(cfg["report_path"]))
    if result.get("status") != "completed":
        sys.exit(f"pipeline did not complete: {result.get('skipped_reason') or result.get('status')}")
    raw = result["output_video_path"]
    out = Path(args.out or f"outputs/{Path(args.source).stem}{'_pose' if args.pose else ''}_demo.mp4")
    out.parent.mkdir(parents=True, exist_ok=True)
    if shutil.which("ffmpeg"):
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", raw, "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-crf", "20", "-preset", "fast", "-movflags", "+faststart", str(out)], check=True)
    else:
        shutil.copy(raw, out)
        print("ffmpeg not found: output is mp4v, not H.264")
    print(f"frames={result['frames_processed']} events={result['events_detected']} pipeline_fps={result.get('pipeline_fps')}")
    print(f"video   : {out}\nsummary : {cfg['report_path']}\ntracker_fallback: {result['tracker_fallback']}")


if __name__ == "__main__":
    main()
