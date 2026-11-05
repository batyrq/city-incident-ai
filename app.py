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
from demo_presets import DEFAULT_PRESET, DIR, PRESETS

FONT = ["-apple-system", "BlinkMacSystemFont", "SF Pro Text", "Inter", "Segoe UI", "Roboto", "sans-serif"]  # same stack as the results site
ACC, ACC_HOVER, INK, MUTED, LINE, SOFT = "#0f766e", "#115e59", "#1f2328", "#586069", "#e1e6ec", "#f6f8fa"  # same tokens as the results site (AA on white)
_light = dict(  # every value is applied to the light and the *_dark variant, so the demo stays light even if the OS is in dark mode
    body_background_fill="#ffffff", body_text_color=INK, body_text_color_subdued=MUTED, background_fill_primary="#ffffff", background_fill_secondary=SOFT,
    block_background_fill="#ffffff", block_border_color=LINE, block_label_background_fill="#ffffff", block_label_text_color=MUTED, block_title_text_color=INK,
    border_color_primary=LINE, input_background_fill="#ffffff", input_border_color=LINE, table_even_background_fill="#ffffff", table_odd_background_fill=SOFT,
    button_primary_background_fill=ACC, button_primary_background_fill_hover=ACC_HOVER, button_primary_text_color="#ffffff", button_primary_border_color=ACC,
    button_secondary_background_fill=SOFT, button_secondary_text_color=INK, color_accent_soft="#e6f4f1", slider_color=ACC,
    checkbox_background_color="#ffffff", checkbox_background_color_selected=ACC, checkbox_border_color_selected=ACC, checkbox_label_text_color=INK,
    checkbox_label_text_color_selected=INK, checkbox_label_background_fill=SOFT, checkbox_label_background_fill_selected="#e6f4f1",
    link_text_color=ACC, link_text_color_hover=ACC_HOVER, link_text_color_visited=ACC)
THEME = gr.themes.Base(primary_hue="teal", neutral_hue="slate", font=FONT, font_mono=["ui-monospace", "SFMono-Regular", "Menlo", "monospace"]).set(
    **{k: v for k, v in _light.items()}, **{k + "_dark": v for k, v in _light.items()})
CSS = ("footer{display:none !important} .gradio-container{background:#fff !important;max-width:1120px !important;min-height:0 !important;padding:18px !important} .gradio-container .main{padding:0 !important}"
       " body,html{min-height:0 !important;background:#fff !important} h1{letter-spacing:-.02em} .prose, .prose *{color:" + INK + "}"
       " .preset-note{color:" + MUTED + ";font-size:13.5px;line-height:1.45;margin:2px 0 0} .preset-note p{margin:0;color:" + MUTED + "}")
JS = "() => { document.body.classList.remove('dark'); }"  # always light, like the site
PRESET_DIR = Path(__file__).parent / DIR
CHOICES = [(p["label"], str(PRESET_DIR / p["file"])) for p in PRESETS if (PRESET_DIR / p["file"]).exists()]
BY_PATH = {str(PRESET_DIR / p["file"]): p for p in PRESETS}
DEFAULT_CLIP = next((str(PRESET_DIR / p["file"]) for p in PRESETS if p["id"] == DEFAULT_PRESET and (PRESET_DIR / p["file"]).exists()), CHOICES[0][1] if CHOICES else None)


def preset_info(path):
    """Selecting a preset sets the recommended pose switch and frame limit and shows what the clip is for."""
    p = BY_PATH.get(path)
    if not p:
        return gr.update(), gr.update(), ""
    note = p["shows"] + (" " + p["note"] if p["note"] else "")
    return bool(p["pose"]), int(p["max_frames"]), note


EVENT_COLS = ["event", "start (s)", "end (s)", "tracks", "max conf"]


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
             (e.get("confidence_summary") or {}).get("max")] for e in events]
    stats = {
        "frames_processed": result["frames_processed"], "device": result["device"], "weights": result["weights"], "pose_overlay": bool(pose),
        "pipeline_fps": result.get("pipeline_fps"), "detect_track_mean_ms": result.get("mean_latency_ms"),
        "unique_tracks": result.get("tracked_objects_total"), "classes_seen": result.get("classes_seen"),
        "events_detected": result.get("events_detected"), "tracker_fallback": result.get("tracker_fallback"),
    }
    return str(out), rows, stats


def build() -> gr.Blocks:
    with gr.Blocks(title="city-incident-ai live demo", fill_width=True) as demo:
        gr.Markdown("Pick an example clip or upload your own. The existing pipeline runs (YOLO11s detection, ByteTrack, event rules) and returns an annotated video, the events and run statistics.")
        with gr.Row():
            with gr.Column():
                upload = gr.Video(label="Upload a video", sources=["upload"])
                bundled = gr.Dropdown(CHOICES, value=DEFAULT_CLIP, label="…or an example clip (used when nothing is uploaded)")
                note = gr.Markdown(preset_info(DEFAULT_CLIP)[2], elem_classes="preset-note")
                pose = gr.Checkbox(bool(preset_info(DEFAULT_CLIP)[0]), label="Pose skeletons (yolo11n-pose)")
                device = gr.Radio(["auto", "mps", "cpu"], value="auto", label="Device (auto = MPS with yolo11s.pt if available, else CPU ONNX)")
                max_frames = gr.Slider(0, 1000, value=int(preset_info(DEFAULT_CLIP)[1]), step=10, label="Max frames (0 = whole video)")
                run = gr.Button("Run pipeline", variant="primary")
            with gr.Column():
                video = gr.Video(label="Annotated video", autoplay=True)
                events = gr.Dataframe(headers=EVENT_COLS, label="Events (empty = no rule fired)", interactive=False, wrap=True, column_widths=["30%", "17%", "17%", "16%", "20%"])
                stats = gr.JSON(label="Run summary")
        bundled.change(preset_info, bundled, [pose, max_frames, note])
        run.click(process, [upload, bundled, pose, device, max_frames], [video, events, stats])
    return demo


if __name__ == "__main__":
    build().queue().launch(server_name="127.0.0.1", server_port=7860, theme=THEME, css=CSS, js=JS)
