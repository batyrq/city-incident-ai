"""Optional demo overlay drawn on top of the pipeline's own detections (config key ``viz``).

Boxes (class + track ID), ~1 s motion trails per track, an event banner, an FPS/latency HUD and optional pose
skeletons. It only consumes the detection dicts the pipeline already produced; no detection/tracking/rule logic here.
"""
from __future__ import annotations

from collections import deque
from typing import Any

import cv2

# COCO-17 skeleton (keypoint index pairs)
SKELETON = [(15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12), (5, 6), (5, 7), (6, 8),
            (7, 9), (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6)]
PALETTE = [(255, 99, 71), (60, 179, 113), (65, 105, 225), (255, 193, 37), (186, 85, 211), (0, 206, 209),
           (255, 127, 80), (154, 205, 50), (255, 105, 180), (100, 149, 237)]  # RGB


def _color(track_id: int | None) -> tuple[int, int, int]:
    r, g, b = PALETTE[(track_id or 0) % len(PALETTE)]
    return (b, g, r)  # OpenCV BGR


class Overlay:
    MAX_GAP = 3  # frames

    def __init__(self, viz: dict[str, Any], fps: float, device: str | None = None):
        self.trail_len = max(2, int(round(float(viz.get("trail_seconds", 1.0)) * fps)))
        self.banner_frames = max(1, int(round(float(viz.get("banner_seconds", 2.0)) * fps)))
        self.trails: dict[int, deque] = {}
        self.last_seen: dict[int, int] = {}
        self.banner: tuple[str, int] | None = None  # (text, last frame to show)
        self.latencies: deque = deque(maxlen=30)
        self.pose = None
        self.pose_device = device
        if viz.get("pose"):
            from ultralytics import YOLO
            self.pose = YOLO(str(viz.get("pose_weights", "yolo11n-pose.pt")))

    def draw(self, frame: Any, detections: list[dict[str, Any]], events: list[str], frame_idx: int, latency_ms: float) -> Any:
        img = frame.copy()
        self.latencies.append(latency_ms)
        if self.pose is not None:
            self._draw_pose(img)
        for det in detections:
            tid = det.get("track_id")
            x1, y1, x2, y2 = (int(det[k]) for k in ("x1", "y1", "x2", "y2"))
            if tid is not None:
                trail = self.trails.setdefault(int(tid), deque(maxlen=self.trail_len))
                trail.append((frame_idx, (int((x1 + x2) / 2), int(y2))))  # bottom-centre = ground contact point
                self.last_seen[int(tid)] = frame_idx
        for tid in [t for t, last in self.last_seen.items() if frame_idx - last > self.MAX_GAP]:
            self.trails.pop(tid, None); self.last_seen.pop(tid, None)  # track not visible any more: drop its trail
        for tid, pts in self.trails.items():
            pts = [p for p in pts if frame_idx - p[0] <= self.trail_len]
            for i in range(1, len(pts)):
                if pts[i][0] - pts[i - 1][0] <= self.MAX_GAP:  # don't connect points separated by missed frames
                    cv2.line(img, pts[i - 1][1], pts[i][1], _color(tid), max(2, int(2 + 3 * i / len(pts))), cv2.LINE_AA)
        for det in detections:
            tid = det.get("track_id")
            x1, y1, x2, y2 = (int(det[k]) for k in ("x1", "y1", "x2", "y2"))
            col = _color(tid)
            cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
            label = f"{det['class_name']} #{tid}" if tid is not None else str(det["class_name"])
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), col, -1)
            cv2.putText(img, label, (x1 + 2, max(th, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        if events:
            self.banner = ("EVENT: " + ", ".join(sorted(set(events))), frame_idx + self.banner_frames)
        if self.banner and frame_idx <= self.banner[1]:
            w = img.shape[1]
            cv2.rectangle(img, (0, 0), (w, 38), (0, 0, 200), -1)
            cv2.putText(img, self.banner[0], (12, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        mean = sum(self.latencies) / len(self.latencies)
        hud = f"detect+track {mean:.1f} ms | {1000.0 / mean:.0f} FPS (model loop) | frame {frame_idx}"
        cv2.rectangle(img, (0, img.shape[0] - 26), (cv2.getTextSize(hud, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0][0] + 16, img.shape[0]), (0, 0, 0), -1)
        cv2.putText(img, hud, (8, img.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        return img

    def _draw_pose(self, img: Any) -> None:
        kwargs = {"verbose": False, "conf": 0.35, "imgsz": 640}
        if self.pose_device:
            kwargs["device"] = self.pose_device
        res = self.pose.predict(img, **kwargs)[0]
        if res.keypoints is None or res.keypoints.xy is None:
            return
        kxy, kcf = res.keypoints.xy.cpu().numpy(), res.keypoints.conf
        kcf = kcf.cpu().numpy() if kcf is not None else None
        for p in range(len(kxy)):
            ok = lambda i: (kcf is None or kcf[p][i] > 0.4) and kxy[p][i][0] > 0 and kxy[p][i][1] > 0
            for a, b in SKELETON:
                if ok(a) and ok(b):
                    cv2.line(img, tuple(int(v) for v in kxy[p][a]), tuple(int(v) for v in kxy[p][b]), (0, 255, 255), 2, cv2.LINE_AA)
            for i in range(17):
                if ok(i):
                    cv2.circle(img, tuple(int(v) for v in kxy[p][i]), 3, (0, 128, 255), -1, cv2.LINE_AA)
