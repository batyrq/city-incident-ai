"""Speed vs accuracy benchmark of detector variants on this machine (batch size 1).

  python -m eval.benchmark_speed                      # all variants -> reports/metrics/speed.json, docs/speed_vs_ap.png
  python -m eval.benchmark_speed --variants fp32_onnx_cpu,int8_onnx_cpu

Per variant:
  * model-only latency: wall time of the backend forward pass (AutoBackend.forward, with torch.mps.synchronize
    around it on MPS), 300 frames of data/videos/mixkit-61.mp4 after 20 warm-up frames. Excludes letterbox/NMS.
  * detector total: full ultralytics predict() call (preprocess + forward + NMS), same frames.
  * pipeline FPS: the normal demo path (run_demo_video: decode + detect/track + rules + draw + write), first 300 frames.
  * accuracy: MOT17-02/04/09 FRCNN folders, every 5th frame (frames 1, 6, 11, ...), person only, conf floor 0.01,
    NMS IoU 0.5: AP50 (same matcher/AP code as eval_detection.py) and recall at conf 0.35 by GT box height.
"""
from __future__ import annotations

import argparse
import json
import platform
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from eval.eval_detection import match_sequence, merge, summarize
from eval.mot_common import SEQS, image_path, load_gt, seq_info

CLIP = "data/videos/mixkit-61.mp4"
WARMUP, TIMED, ACC_STRIDE, N_CALIB = 20, 300, 5, 100
WORK = Path("data/bench_models")  # gitignored (data/)
CFG = "configs/infer_pretrained_yolo11s.yaml"
OUT_JSON, OUT_PNG = Path("reports/metrics/speed.json"), Path("docs/speed_vs_ap.png")

# name -> (label, weights path, imgsz, device)
VARIANTS = {
    "fp32_pt_cpu": ("yolo11s PyTorch CPU", WORK / "yolo11s.pt", 640, "cpu"),
    "fp32_pt_mps": ("yolo11s PyTorch MPS", WORK / "yolo11s.pt", 640, "mps"),
    "fp32_onnx_cpu": ("yolo11s ONNX CPU (repo)", Path("yolo11s.onnx"), 640, "cpu"),
    "fp32_onnx_coreml": ("yolo11s ONNX CoreML EP", Path("yolo11s.onnx"), 640, "mps"),
    "int8_onnx_cpu": ("yolo11s ONNX INT8 static QDQ CPU", WORK / "yolo11s_int8_qdq.onnx", 640, "cpu"),
    "int8dyn_onnx_cpu": ("yolo11s ONNX INT8 dynamic CPU", WORK / "yolo11s_int8_dyn.onnx", 640, "cpu"),
    "n_onnx_cpu": ("yolo11n ONNX CPU", WORK / "yolo11n.onnx", 640, "cpu"),
    "s960_onnx_cpu": ("yolo11s ONNX CPU imgsz 960", WORK / "yolo11s_960.onnx", 960, "cpu"),
}


# ----------------------------------------------------------------------------- model preparation
def prepare_models(only: set[str]) -> dict[str, str]:
    """Build the derived models. Returns {variant: note}; a missing file means that variant was skipped."""
    from ultralytics import YOLO
    notes: dict[str, str] = {}
    WORK.mkdir(parents=True, exist_ok=True)
    for pt in ("yolo11s.pt", "yolo11n.pt"):  # work on copies: export writes next to the .pt and must not touch repo yolo11s.onnx
        if not (WORK / pt).exists():
            shutil.copy(pt, WORK / pt)
    if "n_onnx_cpu" in only and not (WORK / "yolo11n.onnx").exists():
        YOLO(str(WORK / "yolo11n.pt")).export(format="onnx", imgsz=640, dynamic=False, simplify=True)
    if "s960_onnx_cpu" in only and not (WORK / "yolo11s_960.onnx").exists():
        out = YOLO(str(WORK / "yolo11s.pt")).export(format="onnx", imgsz=960, dynamic=False, simplify=True)
        Path(out).rename(WORK / "yolo11s_960.onnx")
    if "int8_onnx_cpu" in only and not (WORK / "yolo11s_int8_qdq.onnx").exists():
        notes["int8_onnx_cpu"] = quantize_static_qdq()
    if "int8dyn_onnx_cpu" in only and not (WORK / "yolo11s_int8_dyn.onnx").exists():
        notes["int8dyn_onnx_cpu"] = quantize_dynamic_model()
    return notes


def _copy_metadata(src: str, dst: str) -> None:
    import onnx
    m_src, m_dst = onnx.load(src), onnx.load(dst)
    del m_dst.metadata_props[:]
    for p in m_src.metadata_props:
        e = m_dst.metadata_props.add(); e.key, e.value = p.key, p.value
    onnx.save(m_dst, dst)


def calibration_frames() -> list[np.ndarray]:
    """~100 MOT17 frames from the three sequences, offset 3 within the stride so they never coincide with eval frames."""
    import cv2
    from ultralytics.data.augment import LetterBox
    lb = LetterBox((640, 640), auto=False, stride=32)
    per = N_CALIB // len(SEQS)
    out = []
    for seq in SEQS:
        n = seq_info(seq)["length"]
        cand = [f for f in range(1, n + 1) if (f - 1) % ACC_STRIDE == 2]  # frames 3, 8, 13, ...
        idx = np.linspace(0, len(cand) - 1, per + (N_CALIB - per * len(SEQS) if seq == SEQS[-1] else 0)).astype(int)
        for i in idx:
            img = cv2.imread(str(image_path(seq, cand[i])))
            x = lb(image=img)[:, :, ::-1].transpose(2, 0, 1)  # BGR->RGB, HWC->CHW
            out.append(np.ascontiguousarray(x, dtype=np.float32)[None] / 255.0)
    return out


def quantize_static_qdq() -> str:
    from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
    from onnxruntime.quantization.shape_inference import quant_pre_process
    pre = str(WORK / "yolo11s_prep.onnx")
    quant_pre_process("yolo11s.onnx", pre)
    import onnx
    from onnx import version_converter
    m = onnx.load(pre)  # repo model is opset 12; per-channel DequantizeLinear (axis attr) needs >= 13
    onnx.save(version_converter.convert_version(m, 17), pre)
    frames = calibration_frames()

    class Reader(CalibrationDataReader):
        def __init__(self): self.it = iter(frames)
        def get_next(self):
            x = next(self.it, None)
            return None if x is None else {"images": x}

    dst = str(WORK / "yolo11s_int8_qdq.onnx")
    # Detect-head decode (DFL, box arithmetic, concat of boxes 0-640 with class scores 0-1, sigmoid) stays FP32:
    # quantizing it with one shared activation scale rounds every class score to 0 (observed: AP50 0.000).
    exclude = [n.name for n in onnx.load(pre).graph.node
               if n.name.startswith("/model.23/") and not (n.op_type in ("Conv", "Sigmoid", "Mul") and ("/cv2." in n.name or "/cv3." in n.name))]
    quantize_static(pre, dst, Reader(), quant_format=QuantFormat.QDQ, activation_type=QuantType.QInt8,
                    weight_type=QuantType.QInt8, per_channel=True, nodes_to_exclude=exclude)
    _copy_metadata("yolo11s.onnx", dst)
    return (f"static QDQ, QInt8 act/weights, per-channel, {len(frames)} MOT17 calibration frames (min-max); "
            f"{len(exclude)} detect-head decode nodes kept FP32 (without this, class scores all quantize to 0 and AP50 is 0.000)")


def quantize_dynamic_model() -> str:
    from onnxruntime.quantization import QuantType, quantize_dynamic
    dst = str(WORK / "yolo11s_int8_dyn.onnx")
    quantize_dynamic("yolo11s.onnx", dst, weight_type=QuantType.QUInt8)
    _copy_metadata("yolo11s.onnx", dst)
    return "dynamic quantization, QUInt8 weights"


# ----------------------------------------------------------------------------- measurement
class ForwardTimer:
    """Times AutoBackend.forward (the model call only). Syncs MPS so asynchronous GPU work is counted."""

    def __init__(self):
        import torch
        from ultralytics.nn.autobackend import AutoBackend
        self.torch, self.cls, self.orig, self.times = torch, AutoBackend, AutoBackend.forward, []
        timer = self

        def forward(this, *a, **k):
            mps = this.device.type == "mps"
            if mps: timer.torch.mps.synchronize()
            t0 = time.perf_counter()
            out = timer.orig(this, *a, **k)
            if mps: timer.torch.mps.synchronize()
            timer.times.append((time.perf_counter() - t0) * 1000)
            return out
        AutoBackend.forward = forward

    def close(self):
        self.cls.forward = self.orig


def pct(xs: list[float]) -> dict:
    s = sorted(xs)
    return {"mean_ms": statistics.fmean(s), "p50_ms": s[len(s) // 2], "p95_ms": s[min(len(s) - 1, int(round(0.95 * (len(s) - 1))))], "n": len(s)}


def measure_latency(model, imgsz: int, device: str) -> dict:
    import cv2
    cap = cv2.VideoCapture(CLIP)
    timer = ForwardTimer()
    total, fwd = [], []
    try:
        for i in range(WARMUP + TIMED):
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"{CLIP} ended after {i} frames")
            timer.times.clear()
            t0 = time.perf_counter()
            model.predict(frame, conf=0.35, iou=0.5, imgsz=imgsz, device=device, classes=[0], verbose=False)
            dt = (time.perf_counter() - t0) * 1000
            if i >= WARMUP:
                total.append(dt); fwd.append(timer.times[-1])
    finally:
        timer.close(); cap.release()
    return {"model_only": pct(fwd), "detector_total_incl_pre_nms": pct(total)}


def measure_accuracy(model, imgsz: int, device: str) -> dict:
    import cv2
    parts, nframes = [], 0
    for seq in SEQS:
        frames = list(range(1, seq_info(seq)["length"] + 1, ACC_STRIDE))
        rows = []
        for fr in frames:
            res = model.predict(cv2.imread(str(image_path(seq, fr))), conf=0.01, iou=0.5, imgsz=imgsz, device=device, classes=[0], verbose=False)[0]
            b = res.boxes
            if len(b):
                rows.append(np.column_stack([np.full(len(b), fr), b.xyxy.cpu().numpy(), b.conf.cpu().numpy()]))
        dets = np.concatenate(rows) if rows else np.zeros((0, 6))
        parts.append(match_sequence(seq, dets, frames)); nframes += len(frames)
    s = summarize(merge(parts)); s.pop("_curve", None)
    return {"frames_evaluated": nframes, "n_gt_boxes": s["n_gt_boxes"], "AP50": s["AP50"], "AP50_95": s["AP50_95"],
            "at_conf_0.35": s["at_conf_0.35"], "recall_by_height_px": s["recall_by_height_px"]}


def measure_pipeline(weights: Path, imgsz: int, device: str, name: str) -> dict:
    from src.infer.runtime import run_demo_video
    from src.utils.io import load_yaml
    cfg = load_yaml(CFG)
    cfg["run_name"] = f"bench_{name}"
    cfg["report_path"] = str(WORK / f"bench_{name}_summary.json")
    cfg["alerts_root"] = str(WORK / "alerts")
    cfg["model_weights"] = str(weights)
    cfg["runtime"].update(max_frames=TIMED, imgsz=imgsz, device=device)
    res = run_demo_video(CLIP, cfg, model_path=str(weights))
    return {"frames": res["frames_processed"], "wall_seconds": res["pipeline_wall_seconds"], "fps": res["pipeline_fps"],
            "detect_track_mean_ms": res["mean_latency_ms"], "tracker_fallback": res["tracker_fallback"], "events": res["events_detected"]}


def environment() -> dict:
    import onnxruntime, torch, ultralytics
    sh = lambda *c: subprocess.run(c, capture_output=True, text=True).stdout.strip()
    return {"chip": sh("sysctl", "-n", "machdep.cpu.brand_string"), "cpu_cores": int(sh("sysctl", "-n", "hw.ncpu")),
            "ram_gb": round(int(sh("sysctl", "-n", "hw.memsize")) / 2**30, 1), "macos": sh("sw_vers", "-productVersion"),
            "python": platform.python_version(), "torch": torch.__version__, "onnxruntime": onnxruntime.__version__,
            "onnxruntime_providers": onnxruntime.get_available_providers(), "ultralytics": ultralytics.__version__,
            "torch_mps_available": bool(torch.backends.mps.is_available())}


def run_variant(key: str) -> dict:
    from ultralytics import YOLO
    label, weights, imgsz, device = VARIANTS[key]
    if not Path(weights).exists():
        return {"label": label, "status": "skipped", "reason": f"{weights} was not produced"}
    model = YOLO(str(weights), task="detect")
    r = {"label": label, "weights": str(weights), "imgsz": imgsz, "device": device, "size_mb": round(Path(weights).stat().st_size / 2**20, 1)}
    t = time.time()
    r["latency"] = measure_latency(model, imgsz, device)
    print(f"  latency {r['latency']['model_only']['p50_ms']:.1f} ms p50 ({time.time() - t:.0f}s)", flush=True)
    r["accuracy"] = measure_accuracy(model, imgsz, device)
    print(f"  AP50 {r['accuracy']['AP50']:.3f}", flush=True)
    del model
    r["pipeline"] = measure_pipeline(Path(weights), imgsz, device, key)
    print(f"  pipeline {r['pipeline']['fps']} fps", flush=True)
    r["status"] = "ok"
    return r


def plot(results: dict) -> None:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pts = [(v["latency"]["model_only"]["p50_ms"], v["accuracy"]["AP50"], v["label"]) for v in results.values() if v.get("status") == "ok"]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    # the 8 points cluster tightly, so points are numbered and explained in a legend instead of labelled in place
    for i, (x, y, lab) in enumerate(sorted(pts), 1):
        ax.scatter(x, y, s=160, zorder=3, label=f"{i}  {lab}  ({x:.1f} ms, AP50 {y:.3f})")
        ax.annotate(str(i), (x, y), ha="center", va="center", fontsize=8, color="white", zorder=4, fontweight="bold")
    ax.set_xlim(0, 140)
    ax.legend(loc="upper left", fontsize=7.5, frameon=True, labelspacing=0.8)
    ax.set_xlabel("model-only latency, p50 (ms, batch 1)"); ax.set_ylabel("person AP50, MOT17-02/04/09 (every 5th frame)")
    ax.grid(alpha=.3); ax.set_title("Speed vs accuracy (this Mac)")
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True); fig.tight_layout(); fig.savefig(OUT_PNG, dpi=140)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--plot-only", action="store_true", help="redraw docs/speed_vs_ap.png from reports/metrics/speed.json")
    args = ap.parse_args()
    if args.plot_only:
        plot(json.loads(OUT_JSON.read_text())["variants"]); return
    keys = [k for k in args.variants.split(",") if k]
    notes = prepare_models(set(keys))
    prev = json.loads(OUT_JSON.read_text())["variants"] if OUT_JSON.exists() else {}
    results = dict(prev)
    for k in keys:
        print(k, flush=True)
        try:
            results[k] = run_variant(k)
        except Exception as exc:  # record, don't hide
            results[k] = {"label": VARIANTS[k][0], "status": "failed", "reason": f"{type(exc).__name__}: {exc}"}
            print("  FAILED", results[k]["reason"], flush=True)
        if k in notes:
            results[k]["quantization"] = notes[k]
        results = {k2: results[k2] for k2 in VARIANTS if k2 in results}
        OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
        OUT_JSON.write_text(json.dumps({"environment": environment(), "protocol": {
            "clip": CLIP, "warmup_frames": WARMUP, "timed_frames": TIMED, "accuracy_frame_stride": ACC_STRIDE,
            "accuracy_conf_floor": 0.01, "recall_conf": 0.35, "nms_iou": 0.5, "batch": 1,
            "model_only_definition": "AutoBackend.forward wall time, MPS synchronised; excludes letterbox and NMS",
            "pipeline_definition": "run_demo_video frame loop, first 300 frames, tracker detector_conf 0.1 / display 0.35, annotated mp4 written"},
            "variants": results}, indent=2))
    plot(results)
    print("wrote", OUT_JSON, OUT_PNG)


if __name__ == "__main__":
    main()
