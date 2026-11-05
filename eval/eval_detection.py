"""Person-detection evaluation of the demo detector (YOLO11s ONNX) on MOT17.

Usage:
  python eval/eval_detection.py                 # run detector (conf 0.01) and evaluate
  python eval/eval_detection.py --use-cache     # re-evaluate from data/mot/cache/dets_*.npz
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from eval.mot_common import SEQS, image_path, iou_matrix, load_detector, load_gt, per_frame, seq_info

IOU_THRESHOLDS = [round(0.5 + 0.05 * i, 2) for i in range(10)]
IGNORE_IOU = 0.5
DEMO_CONF = 0.35
CACHE = Path("data/mot/cache")


def run_detector(seq: str, conf: float, imgsz: int, iou_nms: float) -> tuple[np.ndarray, dict]:
    import cv2
    model = load_detector()
    n = seq_info(seq)["length"]
    rows, times = [], []
    for fr in range(1, n + 1):
        img = cv2.imread(str(image_path(seq, fr)))
        t0 = time.perf_counter()
        res = model.predict(source=img, conf=conf, iou=iou_nms, imgsz=imgsz, classes=[0], verbose=False)[0]
        times.append((time.perf_counter() - t0) * 1000)
        b = res.boxes
        if len(b):
            xyxy, cf = b.xyxy.cpu().numpy(), b.conf.cpu().numpy()
            rows.append(np.column_stack([np.full(len(cf), fr), xyxy, cf]))
    dets = np.concatenate(rows) if rows else np.zeros((0, 6))
    return dets, {"mean_ms_per_frame": float(np.mean(times)), "frames": n}


def match_sequence(seq: str, dets: np.ndarray) -> dict:
    """Greedy per-frame matching (descending confidence) at every IoU threshold.

    Returns per-prediction records and per-GT match info at IoU 0.5.
    """
    gt = load_gt(seq)
    gt_f, ign_f = per_frame(gt["valid"]), per_frame(gt["ignore"])
    det_f = per_frame({"frame": dets[:, 0].astype(int), "xyxy": dets[:, 1:5], "conf": dets[:, 5]}) if len(dets) else {}
    recs = {t: {"conf": [], "status": []} for t in IOU_THRESHOLDS}  # status: 1 TP, 0 FP, -1 ignored
    gt_info = {"vis": [], "height": [], "matched_conf": []}  # matched_conf: conf of matching pred at IoU 0.5 or -1
    for fr in range(1, seq_info(seq)["length"] + 1):
        g = gt_f.get(fr); ig = ign_f.get(fr); d = det_f.get(fr)
        gx = g["xyxy"] if g else np.zeros((0, 4))
        ng = len(gx)
        if g:
            gt_info["vis"].extend(g["vis"].tolist())
            gt_info["height"].extend((gx[:, 3] - gx[:, 1]).tolist())
        mconf = np.full(ng, -1.0)
        if d is None:
            gt_info["matched_conf"].extend(mconf.tolist())
            continue
        order = np.argsort(-d["conf"], kind="stable")
        px, pc = d["xyxy"][order], d["conf"][order]
        ious = iou_matrix(px, gx)
        ign = iou_matrix(px, ig["xyxy"]).max(1) if ig is not None and len(ig["xyxy"]) else np.zeros(len(px))
        for t in IOU_THRESHOLDS:
            matched = np.zeros(ng, bool)
            for p in range(len(px)):
                if ng:
                    row = np.where(matched, -1.0, ious[p]); j = int(np.argmax(row))
                else:
                    row, j = np.array([-1.0]), 0
                if ng and row[j] >= t:
                    matched[j] = True; status = 1
                    if t == 0.5:
                        mconf[j] = pc[p]
                elif ign[p] > IGNORE_IOU:
                    status = -1
                else:
                    status = 0
                recs[t]["conf"].append(float(pc[p])); recs[t]["status"].append(status)
        gt_info["matched_conf"].extend(mconf.tolist())
    return {"recs": recs, "gt": gt_info, "n_gt": int(len(gt["valid"]["frame"])), "gt_stats": gt["stats"]}


def merge(parts: list[dict]) -> dict:
    out = {"recs": {t: {"conf": [], "status": []} for t in IOU_THRESHOLDS}, "gt": {"vis": [], "height": [], "matched_conf": []}, "n_gt": 0}
    for p in parts:
        for t in IOU_THRESHOLDS:
            out["recs"][t]["conf"] += p["recs"][t]["conf"]; out["recs"][t]["status"] += p["recs"][t]["status"]
        for k in out["gt"]:
            out["gt"][k] += p["gt"][k]
        out["n_gt"] += p["n_gt"]
    return out


def pr_curve(conf: list[float], status: list[int], n_gt: int):
    conf, status = np.asarray(conf), np.asarray(status)
    keep = status >= 0
    conf, status = conf[keep], status[keep]
    order = np.argsort(-conf, kind="stable")
    conf, tp = conf[order], (status[order] == 1).astype(float)
    ctp, cfp = np.cumsum(tp), np.cumsum(1 - tp)
    rec = ctp / max(n_gt, 1)
    prec = ctp / np.maximum(ctp + cfp, 1e-9)
    return conf, prec, rec


def average_precision(prec: np.ndarray, rec: np.ndarray) -> float:
    """All-point interpolated AP (area under the monotone precision envelope)."""
    if len(prec) == 0:
        return 0.0
    mrec = np.concatenate([[0.0], rec, [1.0]]); mpre = np.concatenate([[0.0], prec, [0.0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.flatnonzero(mrec[1:] != mrec[:-1])
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def summarize(m: dict) -> dict:
    n_gt = m["n_gt"]
    aps = {}
    for t in IOU_THRESHOLDS:
        c, p, r = pr_curve(m["recs"][t]["conf"], m["recs"][t]["status"], n_gt)
        aps[t] = average_precision(p, r)
    c, p, r = pr_curve(m["recs"][0.5]["conf"], m["recs"][0.5]["status"], n_gt)
    out = {"n_gt_boxes": n_gt, "AP50": aps[0.5], "AP50_95": float(np.mean(list(aps.values()))),
           "AP_by_iou": {str(k): v for k, v in aps.items()}}
    sel = c >= DEMO_CONF
    if sel.any():
        k = int(np.flatnonzero(sel)[-1])
        out["at_conf_0.35"] = {"precision": float(p[k]), "recall": float(r[k]), "tp": int(round(r[k] * n_gt)),
                               "fp": int(round(k + 1 - round(r[k] * n_gt))), "fn": int(n_gt - round(r[k] * n_gt))}
    else:
        out["at_conf_0.35"] = {"precision": None, "recall": 0.0, "tp": 0, "fp": 0, "fn": n_gt}
    f1 = 2 * p * r / np.maximum(p + r, 1e-9)
    if len(f1):
        b = int(np.argmax(f1))
        out["best_f1"] = {"threshold": float(c[b]), "f1": float(f1[b]), "precision": float(p[b]), "recall": float(r[b])}
    # recall buckets at demo conf
    vis, hgt, mc = (np.asarray(m["gt"][k]) for k in ("vis", "height", "matched_conf"))
    hit = mc >= DEMO_CONF
    def bucket(mask):
        n = int(mask.sum()); return {"count": n, "detected": int(hit[mask].sum()), "recall": float(hit[mask].mean()) if n else None}
    out["recall_by_visibility"] = {"<0.3": bucket(vis < 0.3), "0.3-0.7": bucket((vis >= 0.3) & (vis <= 0.7)), ">0.7": bucket(vis > 0.7)}
    out["recall_by_height_px"] = {"<50": bucket(hgt < 50), "50-100": bucket((hgt >= 50) & (hgt <= 100)), ">100": bucket(hgt > 100)}
    out["_curve"] = (c, p, r)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--conf", type=float, default=0.01)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--nms-iou", type=float, default=0.5, help="NMS IoU, same as runtime.iou_threshold in the demo config")
    ap.add_argument("--use-cache", action="store_true")
    ap.add_argument("--out", default="reports/metrics/detection.json")
    ap.add_argument("--plot", default="docs/pr_curve.png")
    args = ap.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    parts, results, timing = {}, {}, {}
    for seq in SEQS:
        cache = CACHE / f"dets_{seq}.npz"
        if args.use_cache and cache.exists():
            dets = np.load(cache)["dets"]
        else:
            dets, timing[seq] = run_detector(seq, args.conf, args.imgsz, args.nms_iou)
            np.savez_compressed(cache, dets=dets)
        parts[seq] = match_sequence(seq, dets)
        results[seq] = summarize(parts[seq]); results[seq]["gt_stats"] = parts[seq]["gt_stats"]; results[seq]["num_predictions_conf_0.01"] = int(len(dets))
        print(seq, {k: results[seq][k] for k in ("AP50", "AP50_95", "at_conf_0.35")}, flush=True)
    results["overall"] = summarize(merge(list(parts.values())))
    print("overall", {k: results["overall"][k] for k in ("AP50", "AP50_95", "at_conf_0.35", "best_f1")})
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, r in results.items():
        c, p, rc = r["_curve"]
        ax.plot(rc, p, lw=2.5 if name == "overall" else 1.2, ls="-" if name == "overall" else "--", label=f"{name} (AP50 {r['AP50']:.3f})")
        sel = c >= DEMO_CONF
        if sel.any():
            k = int(np.flatnonzero(sel)[-1]); ax.plot(rc[k], p[k], "o", ms=6, color=ax.lines[-1].get_color())
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_xlim(0, 1); ax.set_ylim(0, 1.02); ax.grid(alpha=.3)
    ax.set_title("YOLO11s ONNX person detection, MOT17 (IoU 0.5)\ndots = demo threshold conf 0.35"); ax.legend(loc="lower left", fontsize=8)
    Path(args.plot).parent.mkdir(parents=True, exist_ok=True); fig.tight_layout(); fig.savefig(args.plot, dpi=140)
    for r in results.values():
        r.pop("_curve", None)
    payload = {"detector": "yolo11s.onnx", "imgsz": args.imgsz, "conf_floor": args.conf, "nms_iou": args.nms_iou,
               "ignore_classes": [2, 7, 8, 12], "ignore_iou": IGNORE_IOU, "ap_definition": "all-point interpolation, pooled over sequences for 'overall'",
               "timing_ms_per_frame_detector_incl_preprocess": timing, "results": results}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2))
    print("wrote", args.out, args.plot)


if __name__ == "__main__":
    main()
