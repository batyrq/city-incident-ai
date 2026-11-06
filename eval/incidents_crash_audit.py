"""Does the crash score follow the crash, or the camera style? Scores B (CLIP zero-shot), C (probe) and D (Enos-123 YOLO11x) every 0.5 s on the
dashcam crash clips and on two dashcam/CCTV controls in which I saw no collision (c4: highway dashcam, c5: street CCTV; their titles say collision, so treat as unverified).
  python -m eval.incidents_crash_audit -> reports/metrics/incident_crash_audit.json
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from huggingface_hub import hf_hub_download  # noqa: E402
from ultralytics import YOLO  # noqa: E402

from eval.incidents_eval import CLIP_PROMPTS, PRETRAINED, device  # noqa: E402
from src.infer.incidents import IncidentScorer, load_detector_cfg  # noqa: E402

det = load_detector_cfg("configs/incident_detectors.json")
cfgC = {**det, "classes": {"crash": {"approach": "probe", "probe_file": "models/incident_probe_crash.npz", "threshold": 0.5}}}
cfgB = {**det, "classes": {"crash": {"approach": "clip_zeroshot", "threshold": 0.5}}}
sC, sB = IncidentScorer(cfgC), IncidentScorer(cfgB)
sp = PRETRAINED["crash"]
yD = YOLO(hf_hub_download(sp["repo"], sp["file"], local_dir=str(ROOT / "models/pretrained/Enos-123_accident-yolo11x")))
clips = {"crash_dashcam_highway (crash 2-7 s)": ROOT / "demo_videos/incidents/crash_dashcam_highway.mp4",
         "crash_dashcam_truck (crash at 8.5 s)": ROOT / "demo_videos/incidents/crash_dashcam_truck.mp4",
         "crash_aftermath_truck (static aftermath)": ROOT / "demo_videos/incidents/crash_aftermath_truck.mp4",
         "control c4 dashcam highway, no collision seen": ROOT / "data/incidents/raw/videos/commons/c4.webm",
         "control c5 street CCTV, no collision seen": ROOT / "data/incidents/raw/videos/commons/c5.webm"}
out = {}
for name, p in clips.items():
    cap = cv2.VideoCapture(str(p)); fps = cap.get(5) or 25; step = max(1, round(fps * 0.5)); i = 0; rows = {"t": [], "B": [], "C": [], "D": []}
    while True:
        ok, fr = cap.read()
        if not ok or i / fps > 20:
            break
        if i % step == 0:
            rows["t"].append(round(i / fps, 1)); rows["B"].append(round(sB.score(fr)["crash"]["score"], 3)); rows["C"].append(round(sC.score(fr)["crash"]["score"], 3))
            r = yD.predict(fr, conf=0.005, imgsz=640, device=device(), verbose=False)[0]
            c = r.boxes.cls.cpu().numpy() if len(r.boxes) else np.array([]); cf = r.boxes.conf.cpu().numpy() if len(r.boxes) else np.array([])
            rows["D"].append(round(float(cf[c == 0].max()) if (c == 0).any() else 0.0, 3))
        i += 1
    out[name] = {"series_0.5s": rows, **{f"mean_{k}": round(float(np.mean(rows[k])), 3) for k in "BCD"}}
    print(name, {k: out[name][f"mean_{k}"] for k in "BCD"}, flush=True)
thr = {k: json.loads((ROOT / "reports/metrics/incidents.json").read_text())["crash"][k]["threshold"] for k in "BCD"}
(ROOT / "reports/metrics/incident_crash_audit.json").write_text(json.dumps({"image_level_thresholds": thr, "clips": out}, indent=1))
