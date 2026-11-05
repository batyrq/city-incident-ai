"""Live demo: upload a short video (or pick a bundled clip), run the existing pipeline, see the annotated video + events + stats.

  make app      # or: python app.py   -> http://127.0.0.1:7860

All processing is demo.run_demo (the same pipeline as demo.py / infer_video.py); this file only builds the UI.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

for _k in ("NO_PROXY", "no_proxy"):  # Gradio checks its own server at startup; a system HTTP proxy must not intercept that
    os.environ[_k] = ",".join(filter(None, [os.environ.get(_k), "127.0.0.1", "localhost"]))

import gradio as gr

from demo import run_demo

CLIPS = sorted(str(p) for p in Path("data/videos").glob("*.mp4"))
EVENT_COLS = ["event_type", "start_time_sec", "end_time_sec", "tracks_involved", "max_confidence", "source"]


def process(upload, bundled, pose, device, max_frames, progress=gr.Progress()):
    source = upload or bundled
    if not source:
        raise gr.Error("Upload a video or pick a bundled clip.")
    progress(0, desc="loading models")
    try:
        result, out = run_demo(source, device=None if device == "auto" else device, pose=bool(pose), max_frames=int(max_frames) or None,
                               progress_callback=lambda i, n: progress(i / max(n, 1), desc=f"frame {i}/{n}"), run_suffix="_app")
    except Exception as exc:
        raise gr.Error(f"Pipeline failed: {exc}") from exc
    events = json.loads(Path(f"reports/{result['run_name']}_events.json").read_text())
    rows = [[e.get("event_type"), e.get("start_time_sec"), e.get("end_time_sec"), len(e.get("related_track_ids", [])),
             (e.get("confidence_summary") or {}).get("max"), e.get("source")] for e in events]
    stats = {
        "frames_processed": result["frames_processed"], "device": result["device"], "weights": result["weights"], "pose_overlay": bool(pose),
        "pipeline_fps": result.get("pipeline_fps"), "detect_track_mean_ms": result.get("mean_latency_ms"),
        "unique_tracks": result.get("tracked_objects_total"), "classes_seen": result.get("classes_seen"),
        "events_detected": result.get("events_detected"), "tracker_fallback": result.get("tracker_fallback"),
    }
    return str(out), rows, stats


def build() -> gr.Blocks:
    with gr.Blocks(title="city-incident-ai live demo") as demo:
        gr.Markdown("# city-incident-ai · live demo\nUpload a short clip or pick a bundled one. The existing pipeline runs "
                    "(YOLO11s detection, ByteTrack, event rules) and returns an annotated video, the events and run statistics.")
        with gr.Row():
            with gr.Column():
                upload = gr.Video(label="Upload a video", sources=["upload"])
                bundled = gr.Dropdown(CLIPS, value=next((c for c in CLIPS if "mixkit-61" in c), CLIPS[0] if CLIPS else None), label="…or a bundled clip (used when nothing is uploaded)")
                pose = gr.Checkbox(False, label="Pose skeletons (yolo11n-pose)")
                device = gr.Radio(["auto", "mps", "cpu"], value="auto", label="Device (auto = MPS with yolo11s.pt if available, else CPU ONNX)")
                max_frames = gr.Slider(0, 1000, value=150, step=10, label="Max frames (0 = whole video)")
                run = gr.Button("Run pipeline", variant="primary")
            with gr.Column():
                video = gr.Video(label="Annotated video", autoplay=True)
                events = gr.Dataframe(headers=EVENT_COLS, label="Events (empty = no rule fired)", interactive=False)
                stats = gr.JSON(label="Run summary")
        run.click(process, [upload, bundled, pose, device, max_frames], [video, events, stats])
    return demo


if __name__ == "__main__":
    build().queue().launch(server_name="127.0.0.1", server_port=7860)
