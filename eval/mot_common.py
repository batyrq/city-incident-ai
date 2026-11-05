"""Shared helpers for MOT17 evaluation (GT loading, IoU, detector wrapper)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

SEQS = ["MOT17-02", "MOT17-04", "MOT17-09"]
DETECTOR_VARIANT = "FRCNN"  # gt.txt is identical for DPM/FRCNN/SDP; only det/det.txt differs (unused here)
MOT_ROOT = Path("data/mot/MOT17/train")
IGNORE_CLASSES = (2, 7, 8, 12)  # person on vehicle, static person, distractor, reflection
PERSON = 1


def seq_dir(seq: str) -> Path:
    return MOT_ROOT / f"{seq}-{DETECTOR_VARIANT}"


def seq_info(seq: str) -> dict[str, int | float]:
    info = {}
    for line in (seq_dir(seq) / "seqinfo.ini").read_text().splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            info[k.strip()] = v.strip()
    return {"frame_rate": float(info["frameRate"]), "length": int(info["seqLength"]),
            "width": int(info["imWidth"]), "height": int(info["imHeight"])}


def image_path(seq: str, frame: int) -> Path:
    return seq_dir(seq) / "img1" / f"{frame:06d}.jpg"


def load_gt(seq: str) -> dict[str, np.ndarray]:
    """gt.txt columns: frame,id,x,y,w,h,consider,class,visibility. Returns xyxy arrays.

    valid  : class==1 and consider==1  -> the ground truth we are scored against
    ignore : class in IGNORE_CLASSES   -> predictions overlapping these are not penalised
    """
    raw = np.loadtxt(seq_dir(seq) / "gt" / "gt.txt", delimiter=",")
    frame, tid = raw[:, 0].astype(int), raw[:, 1].astype(int)
    xyxy = np.stack([raw[:, 2], raw[:, 3], raw[:, 2] + raw[:, 4], raw[:, 3] + raw[:, 5]], 1)
    consider, cls, vis = raw[:, 6], raw[:, 7].astype(int), raw[:, 8]
    valid = (cls == PERSON) & (consider == 1)
    ignore = np.isin(cls, IGNORE_CLASSES)

    def pack(mask):
        return {"frame": frame[mask], "id": tid[mask], "xyxy": xyxy[mask], "vis": vis[mask], "cls": cls[mask]}

    out = {"valid": pack(valid), "ignore": pack(ignore)}
    out["stats"] = {
        "rows_total": int(len(raw)),
        "rows_valid": int(valid.sum()),
        "rows_ignore_regions": int(ignore.sum()),
        "rows_person_consider0": int(((cls == PERSON) & (consider != 1)).sum()),
        "rows_other_classes": int((~valid & ~ignore & ~((cls == PERSON) & (consider != 1))).sum()),
        "gt_identities": int(len(np.unique(tid[valid]))),
    }
    return out


def per_frame(block: dict[str, np.ndarray]) -> dict[int, dict[str, np.ndarray]]:
    frames: dict[int, dict[str, np.ndarray]] = {}
    order = np.argsort(block["frame"], kind="stable")
    f = block["frame"][order]
    bounds = np.flatnonzero(np.diff(f)) + 1
    for idx in np.split(order, bounds):
        if len(idx):
            frames[int(block["frame"][idx[0]])] = {k: v[idx] for k, v in block.items()}
    return frames


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between xyxy arrays (N,4) and (M,4)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


def load_detector(weights: str = "yolo11s.onnx"):
    from ultralytics import YOLO
    return YOLO(weights, task="detect")
