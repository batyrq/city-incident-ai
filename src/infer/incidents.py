"""Incident detection (fire, smoke, flood, crash) for the video pipeline.

Image-level scorers (the per-class approach and threshold come from configs/incident_detectors.json, which
eval/incidents_eval.py writes from the validation split; see reports/metrics/incidents.json for the test numbers):

  yolo           YOLO11n fine-tuned on D-Fire (fire, smoke): score = highest box confidence of the class
  probe          logistic regression on frozen CLIP ViT-B/16 image embeddings (flood, crash): score = probability
  clip_zeroshot  CLIP text prompts, softmax mass on the incident prompts (no training)

IncidentMonitor turns a stream of sampled-frame scores into events: a class fires when at least k of the last n
sampled frames are above its threshold (temporal smoothing), then stays quiet for a cooldown.
"""
from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
EVENT_NAME = {"fire": "fire", "smoke": "smoke", "flood": "flood", "crash": "crash_candidate"}
DEFAULTS = {"enabled": False,  # library default; the demo config (configs/infer_pretrained_yolo11s.yaml) turns it on
             "detectors_file": "configs/incident_detectors.json", "sample_seconds": 0.5, "k": 3, "n": 5, "cooldown_seconds": 10.0,
            "legacy_iou_collision": False}


def load_detector_cfg(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    return json.loads((p if p.is_absolute() else ROOT / p).read_text())


class IncidentScorer:
    """Scores one BGR frame for every configured class. Models are loaded once and shared by the classes that need them."""

    def __init__(self, detectors: dict[str, Any], device: str | None = None):
        import torch
        self.cfg = detectors
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.classes = list(detectors["classes"].keys())
        self._yolo = None
        self._clip = None
        self._probes: dict[str, tuple[np.ndarray, float]] = {}
        self._text: dict[str, np.ndarray] = {}
        for c, spec in detectors["classes"].items():
            if spec["approach"] == "probe":
                z = np.load(ROOT / spec["probe_file"])
                self._probes[c] = (z["coef"].astype(np.float32), float(z["intercept"]))

    def _need_yolo(self):
        if self._yolo is None:
            from ultralytics import YOLO
            self._yolo = YOLO(str(ROOT / self.cfg["yolo_weights"]))
        return self._yolo

    def _need_clip(self):
        if self._clip is None:
            import open_clip
            import torch
            m, _, pre = open_clip.create_model_and_transforms("ViT-B-16-quickgelu", pretrained="openai")
            self._clip = (m.to(self.device).eval(), pre, open_clip.get_tokenizer("ViT-B-16-quickgelu"), torch)
        return self._clip

    def _embed(self, frame_bgr: np.ndarray) -> np.ndarray:
        from PIL import Image
        m, pre, _, torch = self._need_clip()
        img = Image.fromarray(frame_bgr[:, :, ::-1])
        with torch.no_grad():
            f = m.encode_image(pre(img).unsqueeze(0).to(self.device))
            f = f / f.norm(dim=-1, keepdim=True)
        return f[0].cpu().numpy()

    def _zeroshot(self, cls: str, emb: np.ndarray) -> float:
        if cls not in self._text:
            m, _, tok, torch = self._need_clip()
            pos, neg = self.cfg["clip_prompts"][cls]
            with torch.no_grad():
                t = m.encode_text(tok(pos + neg).to(self.device))
                self._text[cls] = (t / t.norm(dim=-1, keepdim=True)).cpu().numpy()
        lg = 100.0 * self._text[cls] @ emb
        p = np.exp(lg - lg.max())
        p /= p.sum()
        return float(p[: len(self.cfg["clip_prompts"][cls][0])].sum())

    def score(self, frame_bgr: np.ndarray) -> dict[str, dict[str, Any]]:
        """-> {class: {"score": float, "box": [x1, y1, x2, y2] | None}}"""
        out: dict[str, dict[str, Any]] = {}
        emb = None
        yres = None
        for c, spec in self.cfg["classes"].items():
            ap = spec["approach"]
            if ap == "yolo":
                if yres is None:
                    yres = self._need_yolo().predict(frame_bgr, conf=0.005, imgsz=640, device=self.device, verbose=False)[0]
                cid = spec["yolo_class_id"]
                best, box = 0.0, None
                if len(yres.boxes):
                    cls_ids = yres.boxes.cls.cpu().numpy()
                    conf = yres.boxes.conf.cpu().numpy()
                    xyxy = yres.boxes.xyxy.cpu().numpy()
                    idx = np.where(cls_ids == cid)[0]
                    if len(idx):
                        j = idx[np.argmax(conf[idx])]
                        best, box = float(conf[j]), [float(v) for v in xyxy[j]]
                out[c] = {"score": best, "box": box}
            else:
                if emb is None:
                    emb = self._embed(frame_bgr)
                if ap == "probe":
                    coef, b = self._probes[c]
                    s = float(1.0 / (1.0 + np.exp(-(float(coef @ emb) + b))))
                else:
                    s = self._zeroshot(c, emb)
                out[c] = {"score": s, "box": None}
        return out


class IncidentMonitor:
    """k-of-n smoothing + cooldown per class. Feed it one scores dict per sampled frame."""

    def __init__(self, detectors: dict[str, Any], k: int = 3, n: int = 5, cooldown_seconds: float = 10.0):
        self.thr = {c: float(s["threshold"]) for c, s in detectors["classes"].items()}
        self.k, self.n, self.cooldown = k, n, cooldown_seconds
        self.hist = {c: deque(maxlen=n) for c in self.thr}
        self.last_fire: dict[str, float] = {}

    def update(self, scores: dict[str, dict[str, Any]], t_sec: float) -> list[dict[str, Any]]:
        events = []
        for c, thr in self.thr.items():
            s = scores[c]["score"]
            self.hist[c].append((s >= thr, s, scores[c].get("box")))
            pos = [h for h in self.hist[c] if h[0]]
            if len(pos) >= self.k and t_sec - self.last_fire.get(c, -1e9) >= self.cooldown:
                self.last_fire[c] = t_sec
                best = max(pos, key=lambda h: h[1])
                events.append({"class": c, "event_type": EVENT_NAME[c], "t_sec": round(t_sec, 3), "confidence": round(float(np.mean([h[1] for h in pos])), 4),
                               "max_score": round(float(best[1]), 4), "box": best[2], "threshold": thr, "positives_in_window": len(pos), "window": len(self.hist[c])})
        return events
