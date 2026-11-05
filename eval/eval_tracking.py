"""Tracking evaluation (ByteTrack vs BoT-SORT, ReID off) on MOT17, person class.

The detector (YOLO11s ONNX, conf 0.35, NMS IoU 0.5, imgsz 640, person only) runs ONCE per frame and
the same detections are fed to both trackers, so they differ only in the association logic.
Trackers are Ultralytics' own classes with their stock yaml configs (bytetrack.yaml / botsort.yaml).
Results are written in MOT format, then scored with py-motmetrics.

Usage:
  python -m eval.eval_tracking              # run detector+trackers, write MOT files, score
  python -m eval.eval_tracking --eval-only  # re-score existing MOT files
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from eval.mot_common import SEQS, image_path, iou_matrix, load_detector, load_gt, per_frame, seq_info

TRACKERS = {"bytetrack": "bytetrack.yaml", "botsort": "botsort.yaml"}
OUT_DIR = Path("reports/metrics/mot_results")


def make_tracker(yaml_name: str, frame_rate: float):
    from ultralytics.cfg import IterableSimpleNamespace
    from ultralytics.trackers.track import TRACKER_MAP
    from ultralytics.utils import YAML
    from ultralytics.utils.checks import check_yaml
    cfg = IterableSimpleNamespace(**YAML.load(check_yaml(yaml_name)))
    cfg.device = "cpu"
    # this ultralytics version has no frame_rate arg: max_frames_lost = track_buffer frames
    return TRACKER_MAP[cfg.tracker_type](args=cfg), cfg


def run_sequence(seq: str, model, conf: float, imgsz: int, nms_iou: float) -> dict:
    import cv2
    info = seq_info(seq)
    trackers = {name: make_tracker(y, info["frame_rate"]) for name, y in TRACKERS.items()}
    rows = {n: [] for n in TRACKERS}
    times = {n: [] for n in TRACKERS}
    for fr in range(1, info["length"] + 1):
        img = cv2.imread(str(image_path(seq, fr)))
        res = model.predict(source=img, conf=conf, iou=nms_iou, imgsz=imgsz, classes=[0], verbose=False)[0]
        det = res.boxes.cpu().numpy()
        for name, (trk, _) in trackers.items():
            t0 = time.perf_counter()
            tracks = trk.update(det, img)
            times[name].append((time.perf_counter() - t0) * 1000)
            for x1, y1, x2, y2, tid, score, *_ in tracks:
                rows[name].append((fr, int(tid), x1, y1, x2 - x1, y2 - y1, score))
    for name in TRACKERS:
        d = OUT_DIR / name; d.mkdir(parents=True, exist_ok=True)
        with open(d / f"{seq}.txt", "w") as f:
            for fr, tid, x, y, w, h, s in rows[name]:
                f.write(f"{fr},{tid},{x:.2f},{y:.2f},{w:.2f},{h:.2f},{s:.4f},-1,-1,-1\n")
    return {n: {"mean_ms": float(np.mean(t)), "median_ms": float(np.median(t)), "p95_ms": float(np.percentile(t, 95)), "frames": len(t)} for n, t in times.items()}


def load_mot_results(path: Path) -> dict[int, dict[str, np.ndarray]]:
    raw = np.loadtxt(path, delimiter=",", ndmin=2)
    if raw.size == 0:
        return {}
    block = {"frame": raw[:, 0].astype(int), "id": raw[:, 1].astype(int),
             "xyxy": np.stack([raw[:, 2], raw[:, 3], raw[:, 2] + raw[:, 4], raw[:, 3] + raw[:, 5]], 1)}
    return per_frame(block)


def drop_ignore_matches(pred: dict, gt_valid: dict | None, gt_ign: dict | None):
    """TrackEval-style preprocessing: Hungarian-match predictions to all GT boxes of the frame
    (valid + ignore classes) at IoU>0.5; predictions matched to an ignore-class box are removed."""
    from scipy.optimize import linear_sum_assignment
    px, pid = pred["xyxy"], pred["id"]
    if gt_ign is None or len(gt_ign["xyxy"]) == 0 or len(px) == 0:
        return px, pid, 0
    gv = gt_valid["xyxy"] if gt_valid else np.zeros((0, 4))
    allg = np.concatenate([gv, gt_ign["xyxy"]])
    iou = iou_matrix(px, allg)
    r, c = linear_sum_assignment(-iou)
    drop = np.zeros(len(px), bool)
    for i, j in zip(r, c):
        if iou[i, j] > 0.5 and j >= len(gv):
            drop[i] = True
    return px[~drop], pid[~drop], int(drop.sum())


def score(tracker: str, out_dir: Path = OUT_DIR) -> dict:
    import motmetrics as mm
    accs, names, extra = [], [], {}
    for seq in SEQS:
        gt = load_gt(seq)
        gv, gi = per_frame(gt["valid"]), per_frame(gt["ignore"])
        pr = load_mot_results(out_dir / tracker / f"{seq}.txt")
        acc = mm.MOTAccumulator(auto_id=False)
        pred_ids, pred_ids_raw, dropped = set(), set(), 0
        for fr in range(1, seq_info(seq)["length"] + 1):
            g = gv.get(fr); p = pr.get(fr)
            gids = g["id"] if g else np.zeros(0, int); gx = g["xyxy"] if g else np.zeros((0, 4))
            if p:
                pred_ids_raw.update(p["id"].tolist())
                px, pid, nd = drop_ignore_matches(p, g, gi.get(fr)); dropped += nd
            else:
                px, pid = np.zeros((0, 4)), np.zeros(0, int)
            pred_ids.update(pid.tolist())
            # same as mm.distances.iou_matrix(max_iou=0.5) (1-IoU, NaN above 0.5), computed locally because
            # py-motmetrics 1.4.0 calls np.asfarray, which NumPy 2 removed
            iou = iou_matrix(gx, px)
            dist = np.where(iou >= 0.5, 1.0 - iou, np.nan)
            acc.update(gids, pid, dist, frameid=fr)
        accs.append(acc); names.append(seq)
        extra[seq] = {"pred_identities": len(pred_ids), "pred_identities_before_ignore_filter": len(pred_ids_raw), "pred_boxes_dropped_in_ignore_regions": dropped}
    mh = mm.metrics.create()
    cols = ["mota", "idf1", "num_switches", "num_false_positives", "num_misses", "mostly_tracked", "mostly_lost",
            "partially_tracked", "num_unique_objects", "num_fragmentations", "recall", "precision", "num_objects", "num_matches"]
    summ = mh.compute_many(accs, names=names, metrics=cols, generate_overall=True)
    out = {}
    for name in summ.index:
        row = {k: (float(v) if k in ("mota", "idf1", "recall", "precision") else int(v)) for k, v in summ.loc[name].items()}
        row["gt_identities"] = row.pop("num_unique_objects")
        if name in extra:
            row.update(extra[name])
        out[name] = row
    out["OVERALL"]["pred_identities"] = sum(extra[s]["pred_identities"] for s in SEQS)
    return out


ABLATION = {  # name -> (min detector conf given to the tracker, tracker yaml)
    "a_conf0.35_default": (0.35, "bytetrack.yaml"),
    "b_conf0.10_default": (0.10, "bytetrack.yaml"),
    "c_conf0.10_buffer60": (0.10, "configs/bytetrack_buffer60.yaml"),
}
ABL_DIR = Path("reports/metrics/mot_results_conf_ablation")


def vis_recall_from_tracks(tracker: str, out_dir: Path) -> dict:
    """Recall by GT visibility computed from TRACKER OUTPUT: a GT box is hit if the Hungarian match
    between that frame's valid GT and (ignore-filtered) tracker boxes has IoU>=0.5."""
    from scipy.optimize import linear_sum_assignment
    vis_all, hit_all = [], []
    for seq in SEQS:
        gt = load_gt(seq); gv, gi = per_frame(gt["valid"]), per_frame(gt["ignore"])
        pr = load_mot_results(out_dir / tracker / f"{seq}.txt")
        for fr, g in gv.items():
            hit = np.zeros(len(g["id"]), bool)
            p = pr.get(fr)
            if p:
                px, _, _ = drop_ignore_matches(p, g, gi.get(fr))
                if len(px):
                    iou = iou_matrix(g["xyxy"], px); r, c = linear_sum_assignment(-iou)
                    for i, j in zip(r, c):
                        hit[i] = iou[i, j] >= 0.5
            vis_all.extend(g["vis"].tolist()); hit_all.extend(hit.tolist())
    vis, hit = np.asarray(vis_all), np.asarray(hit_all)
    def b(m):
        n = int(m.sum()); return {"count": n, "hit": int(hit[m].sum()), "recall": float(hit[m].mean()) if n else None}
    return {"<0.3": b(vis < 0.3), "0.3-0.7": b((vis >= 0.3) & (vis <= 0.7)), ">0.7": b(vis > 0.7)}


def run_ablation(args) -> None:
    import cv2
    out = {"detector": "yolo11s.onnx", "imgsz": args.imgsz, "nms_iou": args.nms_iou, "detector_conf_floor": 0.10,
           "note": "detector runs once per frame at conf 0.10; config a feeds the tracker only boxes with conf>=0.35 (identical to running the detector at 0.35, since NMS is applied independently of the score floor order)",
           "configs": {}}
    model = load_detector()
    timing = {n: {} for n in ABLATION}
    for seq in SEQS:
        info = seq_info(seq)
        trk = {n: make_tracker(y, info["frame_rate"])[0] for n, (_, y) in ABLATION.items()}
        rows = {n: [] for n in ABLATION}; times = {n: [] for n in ABLATION}
        for fr in range(1, info["length"] + 1):
            img = cv2.imread(str(image_path(seq, fr)))
            res = model.predict(source=img, conf=0.10, iou=args.nms_iou, imgsz=args.imgsz, classes=[0], verbose=False)[0]
            for n, (cmin, _) in ABLATION.items():
                b = res.boxes.cpu().numpy()
                det = b[b.conf >= cmin] if len(b) else b
                t0 = time.perf_counter(); tracks = trk[n].update(det, img); times[n].append((time.perf_counter() - t0) * 1000)
                for x1, y1, x2, y2, tid, sc, *_ in tracks:
                    rows[n].append((fr, int(tid), x1, y1, x2 - x1, y2 - y1, sc))
        for n in ABLATION:
            d = ABL_DIR / n; d.mkdir(parents=True, exist_ok=True)
            with open(d / f"{seq}.txt", "w") as f:
                for fr, tid, x, y, w, h, sc in rows[n]:
                    f.write(f"{fr},{tid},{x:.2f},{y:.2f},{w:.2f},{h:.2f},{sc:.4f},-1,-1,-1\n")
            timing[n][seq] = float(np.mean(times[n]))
        print(seq, "done", flush=True)
    for n, (cmin, y) in ABLATION.items():
        sc = score(n, ABL_DIR)
        out["configs"][n] = {"detector_conf_to_tracker": cmin, "tracker_yaml": y, "scores": sc,
                             "recall_by_visibility_from_tracker_output": vis_recall_from_tracks(n, ABL_DIR),
                             "tracker_ms_per_frame_mean_by_seq": timing[n]}
        o = sc["OVERALL"]; print(n, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in o.items()})
    Path(args.out).write_text(json.dumps(out, indent=2)); print("wrote", args.out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--nms-iou", type=float, default=0.5)
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--conf-ablation", action="store_true", help="ByteTrack detector-conf ablation (a/b/c); writes reports/metrics/tracking_conf_ablation.json")
    ap.add_argument("--out", default="reports/metrics/tracking.json")
    args = ap.parse_args()
    if args.conf_ablation:
        args.out = "reports/metrics/tracking_conf_ablation.json" if args.out == "reports/metrics/tracking.json" else args.out
        return run_ablation(args)
    timing_path = Path("reports/metrics/tracking_timing.json")
    if args.eval_only:
        timing = json.loads(timing_path.read_text())
    else:
        model = load_detector()
        timing = {}
        for seq in SEQS:
            timing[seq] = run_sequence(seq, model, args.conf, args.imgsz, args.nms_iou)
            print(seq, {n: round(v["mean_ms"], 2) for n, v in timing[seq].items()}, flush=True)
        timing_path.parent.mkdir(parents=True, exist_ok=True); timing_path.write_text(json.dumps(timing, indent=2))
    result = {"detector": "yolo11s.onnx", "conf": args.conf, "imgsz": args.imgsz, "nms_iou": args.nms_iou,
              "trackers": {}, "note": "tracker timing = Ultralytics tracker.update() only (excludes detector); botsort includes GMC sparseOptFlow on the full 1080p frame"}
    for name in TRACKERS:
        sc = score(name)
        tm = {s: timing[s][name] for s in SEQS}
        frames = sum(tm[s]["frames"] for s in SEQS)
        sc["OVERALL"]["tracker_ms_per_frame_mean"] = sum(tm[s]["mean_ms"] * tm[s]["frames"] for s in SEQS) / frames
        result["trackers"][name] = {"scores": sc, "timing": tm}
        o = sc["OVERALL"]
        print(name, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in o.items()})
    Path(args.out).write_text(json.dumps(result, indent=2))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
