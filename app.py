"""Live demo: upload a short video (or pick a bundled clip), run the existing pipeline, see the annotated video + events + stats.

  make run                     # whole website incl. this demo at http://localhost:8000 (server.py)
  python app.py                # this demo alone at http://127.0.0.1:7860

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

FONT = ["-apple-system", "BlinkMacSystemFont", "SF Pro Text", "Inter", "Segoe UI", "Roboto", "sans-serif"]  # same stack as the results site
THEME = gr.themes.Base(primary_hue="teal", neutral_hue="slate", font=FONT, font_mono=["ui-monospace", "SFMono-Regular", "Menlo", "monospace"]).set(
    body_background_fill="#0b0d10", body_background_fill_dark="#0b0d10", body_text_color="#e8ecf1", body_text_color_dark="#e8ecf1",
    background_fill_primary="#12161b", background_fill_primary_dark="#12161b", background_fill_secondary="#0f1318", background_fill_secondary_dark="#0f1318",
    block_background_fill="#12161b", block_background_fill_dark="#12161b", block_border_color="#222a33", block_border_color_dark="#222a33",
    border_color_primary="#222a33", border_color_primary_dark="#222a33", input_background_fill="#0f1318", input_background_fill_dark="#0f1318",
    button_primary_background_fill="#5eead4", button_primary_background_fill_dark="#5eead4", button_primary_background_fill_hover="#7ff0de",
    button_primary_background_fill_hover_dark="#7ff0de", button_primary_text_color="#04201c", button_primary_text_color_dark="#04201c",
    color_accent_soft="#12332f", color_accent_soft_dark="#12332f", slider_color="#5eead4", slider_color_dark="#5eead4",
    checkbox_background_color_selected="#5eead4", checkbox_background_color_selected_dark="#5eead4", checkbox_border_color_selected="#5eead4",
    checkbox_border_color_selected_dark="#5eead4", checkbox_label_text_color_selected="#e8ecf1", checkbox_label_text_color_selected_dark="#e8ecf1")
CSS = "footer{display:none !important} .gradio-container{background:#0b0d10 !important;max-width:1120px !important;min-height:0 !important} body,html{min-height:0 !important} h1{letter-spacing:-.02em}"
JS = "() => { document.body.classList.add('dark'); }"  # always dark, like the site
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
    with gr.Blocks(title="city-incident-ai live demo", fill_width=True) as demo:
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
    build().queue().launch(server_name="127.0.0.1", server_port=7860, theme=THEME, css=CSS, js=JS)
