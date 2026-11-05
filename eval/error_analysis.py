"""Data-backed failure cases -> docs/failures/*.jpg and reports/metrics/error_analysis.json.

  python -m eval.error_analysis            # all cases (needs MOT17 data, reports/metrics/mot_results/bytetrack,
                                           #  alerts/analysis_4332 + alerts/analysis_61 from infer_video.py runs)

Every case picks its example programmatically (the rule is stored in the JSON next to the numbers).
Colours: GT green, predictions red (cases a-c); predictions only for the demo-clip cases (d, e).
"""
from __future__ import annotations

import glob
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from eval.eval_tracking import drop_ignore_matches, load_mot_results
from eval.mot_common import SEQS, image_path, iou_matrix, load_gt, per_frame, seq_info

OUT = Path("docs/failures")
GREEN, RED, YELLOW, BLUE = (0, 200, 0), (0, 0, 230), (0, 215, 255), (230, 120, 0)
SPLIT_FRAMES = {s: list(range(1, seq_info(s)["length"] + 1, 5)) for s in SEQS}  # same every-5th-frame subset as speed benchmark
_MODELS: dict = {}


def model(path: str):
    if path not in _MODELS:
        from ultralytics import YOLO
        _MODELS[path] = YOLO(path, task="detect")
    return _MODELS[path]


def detect_persons(img, weights: str, imgsz: int, conf: float = 0.35) -> np.ndarray:
    r = model(weights).predict(img, conf=conf, iou=0.5, imgsz=imgsz, classes=[0], verbose=False)[0]
    return np.column_stack([r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()]) if len(r.boxes) else np.zeros((0, 5))


def box(img, xyxy, color, label=None, thick=2, scale=1.0, off=(0, 0)):
    x1, y1, x2, y2 = [(v - o) * scale for v, o in zip(xyxy[:4], (off[0], off[1], off[0], off[1]))]
    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, thick)
    if label:
        cv2.putText(img, label, (int(x1) + 2, max(12, int(y1) - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def banner(img, text):
    cv2.rectangle(img, (0, 0), (img.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(img, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def crop_view(img, win, scale, gts=(), preds=(), gt_label=None, pred_label=None, title=""):
    x1, y1, x2, y2 = win
    view = cv2.resize(img[y1:y2, x1:x2], None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    for i, g in enumerate(gts):
        box(view, g, GREEN, gt_label(i) if gt_label else None, 2, scale, (x1, y1))
    for i, p in enumerate(preds):
        box(view, p, RED, pred_label(i) if pred_label else None, 2, scale, (x1, y1))
    return banner(view, title)


def best_window(points: np.ndarray, w: int, h: int, W: int, H: int) -> tuple[int, int, int, int]:
    """Window (w x h) containing the most of the given box centres."""
    best, arg = -1, (0, 0)
    cx, cy = points[:, 0], points[:, 1]
    for x in range(0, W - w + 1, 40):
        for y in range(0, H - h + 1, 40):
            n = int(((cx >= x) & (cx < x + w) & (cy >= y) & (cy < y + h)).sum())
            if n > best:
                best, arg = n, (x, y)
    inside = (cx >= arg[0]) & (cx < arg[0] + w) & (cy >= arg[1]) & (cy < arg[1] + h)
    x = int(np.clip(cx[inside].mean() - w / 2, 0, W - w)); y = int(np.clip(cy[inside].mean() - h / 2, 0, H - h))  # recentre on the group
    return x, y, x + w, y + h


def hits(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """GT box hit = Hungarian match with IoU >= 0.5 (same rule as the tracker recall-by-visibility)."""
    from scipy.optimize import linear_sum_assignment
    h = np.zeros(len(gt), bool)
    if len(gt) and len(pred):
        iou = iou_matrix(gt, pred[:, :4]); r, c = linear_sum_assignment(-iou)
        for i, j in zip(r, c):
            h[i] = iou[i, j] >= 0.5
    return h


# --------------------------------------------------------------------------------------------- a) small people
def _pair_image(seq: str, fr: int, fname: str, hmax: int, w640: str, w960: str) -> dict:
    """Side-by-side crop (640 | 960) around the densest group of GT persons with height <= hmax."""
    img = cv2.imread(str(image_path(seq, fr))); H, W = img.shape[:2]
    gt = per_frame(load_gt(seq)["valid"])[fr]["xyxy"]; hgt = gt[:, 3] - gt[:, 1]
    sm = gt[hgt <= hmax]
    win = best_window(np.column_stack([(sm[:, 0] + sm[:, 2]) / 2, (sm[:, 1] + sm[:, 3]) / 2]), 480, 270, W, H)
    cxs, cys = (gt[:, 0] + gt[:, 2]) / 2, (gt[:, 1] + gt[:, 3]) / 2
    inwin = (cxs >= win[0]) & (cxs < win[2]) & (cys >= win[1]) & (cys < win[3])
    panels, stats = [], {"sequence": seq, "frame": fr, "crop_xyxy": list(win), "hmax_px": hmax}
    for sz, w in ((640, w640), (960, w960)):
        pred = detect_persons(img, w, sz); h_all = hits(gt, pred)
        stats[str(sz)] = {"gt_lt50_in_frame": int((hgt < 50).sum()), "lt50_hit": int((h_all & (hgt < 50)).sum()),
                          "gt_50_100_in_frame": int(((hgt >= 50) & (hgt <= 100)).sum()), "50_100_hit": int((h_all & (hgt >= 50) & (hgt <= 100)).sum()),
                          "gt_all_in_frame": int(len(gt)), "all_hit": int(h_all.sum()), "preds": int(len(pred)),
                          "gt_le_hmax_in_crop": int(((hgt <= hmax) & inwin).sum()), "le_hmax_in_crop_hit": int((h_all & (hgt <= hmax) & inwin).sum())}
        st = stats[str(sz)]
        panels.append(crop_view(img, win, 2.0, gt[inwin], pred, title=f"imgsz {sz}: GT<={hmax}px in crop hit {st['le_hmax_in_crop_hit']}/{st['gt_le_hmax_in_crop']} (green GT, red pred>=0.35)"))
    cv2.imwrite(str(OUT / fname), np.hstack(panels), [cv2.IMWRITE_JPEG_QUALITY, 90])
    return stats


def case_a(res: dict) -> None:
    # Pass 1: for every eval frame count hits on GT persons < 100 px at 640 and at 960 (same Hungarian-IoU0.5 hit rule).
    w640, w960 = "yolo11s.onnx", "data/bench_models/yolo11s_960.onnx"
    per, tot = [], {640: {"lt50": [0, 0], "50-100": [0, 0]}, 960: {"lt50": [0, 0], "50-100": [0, 0]}}
    for seq in SEQS:
        gv = per_frame(load_gt(seq)["valid"])
        for fr in SPLIT_FRAMES[seq]:
            g = gv.get(fr)
            if g is None:
                continue
            gt = g["xyxy"]; hgt = gt[:, 3] - gt[:, 1]
            img = cv2.imread(str(image_path(seq, fr)))
            h = {sz: hits(gt, detect_persons(img, w, sz)) for sz, w in ((640, w640), (960, w960))}
            for sz in (640, 960):
                tot[sz]["lt50"][0] += int((h[sz] & (hgt < 50)).sum()); tot[sz]["lt50"][1] += int((hgt < 50).sum())
                tot[sz]["50-100"][0] += int((h[sz] & (hgt >= 50) & (hgt <= 100)).sum()); tot[sz]["50-100"][1] += int(((hgt >= 50) & (hgt <= 100)).sum())
            m = hgt <= 100
            per.append((int((h[960] & m).sum()) - int((h[640] & m).sum()), int((hgt < 50).sum()), seq, fr))
    gain, n_small, seq, fr = max(per)
    stats = _pair_image(seq, fr, "a_small_people_640_vs_960.jpg", 100, w640, w960)
    # second image: the frame with the most GT < 50 px (where 960 still finds (almost) nothing)
    n50, seq50, fr50 = max((n_lt50, sq, f) for _, n_lt50, sq, f in per)
    stats50 = _pair_image(seq50, fr50, "a2_tiny_people_640_vs_960.jpg", 49, w640, w960)
    sp = json.load(open("reports/metrics/speed.json"))["variants"]
    res["a_small_people"] = {
        "selection_rule": "eval frame (every 5th) with the largest gain in hits on GT persons <= 100 px going from imgsz 640 to 960 (ties: more GT < 50 px)",
        "sequence": seq, "frame": fr, "hit_gain_le100px": gain, "this_frame": stats,
        "tiny_people_example": {"selection_rule": "eval frame with the most GT persons < 50 px", "sequence": seq50, "frame": fr50, "this_frame": stats50},
        "all_eval_frames_hits_over_gt": {str(sz): {k: {"hit": v[0], "gt": v[1], "recall": v[0] / v[1]} for k, v in d.items()} for sz, d in tot.items()},
        "benchmark_recall_at_conf_0.35_h<50px": {k: sp[k]["accuracy"]["recall_by_height_px"]["<50"] for k in ("fp32_onnx_cpu", "s960_onnx_cpu")},
        "benchmark_recall_50_100px": {k: sp[k]["accuracy"]["recall_by_height_px"]["50-100"]["recall"] for k in ("fp32_onnx_cpu", "s960_onnx_cpu")},
        "benchmark_latency_p50_ms": {k: sp[k]["latency"]["model_only"]["p50_ms"] for k in ("fp32_onnx_cpu", "s960_onnx_cpu")}}


# --------------------------------------------------------------------------------------------- b) occlusion
def case_b(res: dict) -> None:
    best = (-1, "", 0)
    for seq in SEQS:
        gv = per_frame(load_gt(seq)["valid"])
        for fr in SPLIT_FRAMES[seq]:
            g = gv.get(fr)
            if g is not None:
                n = int((g["vis"] < 0.3).sum())
                best = max(best, (n, seq, fr))
    n_occ, seq, fr = best
    img = cv2.imread(str(image_path(seq, fr))); H, W = img.shape[:2]
    g = per_frame(load_gt(seq)["valid"])[fr]
    pred = detect_persons(img, "yolo11s.onnx", 640)
    h = hits(g["xyxy"], pred)
    occ = g["vis"] < 0.3
    c = (g["xyxy"][occ][:, :2] + g["xyxy"][occ][:, 2:]) / 2
    win = best_window(c, 560, 315, W, H)
    cx, cy = (g["xyxy"][:, 0] + g["xyxy"][:, 2]) / 2, (g["xyxy"][:, 1] + g["xyxy"][:, 3]) / 2
    inw = (cx >= win[0]) & (cx < win[2]) & (cy >= win[1]) & (cy < win[3])
    view = cv2.resize(img[win[1]:win[3], win[0]:win[2]], None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    for i in np.flatnonzero(inw):
        col = YELLOW if g["vis"][i] < 0.3 else GREEN
        box(view, g["xyxy"][i], col, f"v{g['vis'][i]:.2f}" if g["vis"][i] < 0.3 else None, 2, 2.0, win[:2])
    for p in pred:
        box(view, p, RED, None, 2, 2.0, win[:2])
    banner(view, f"{seq} frame {fr}: GT green (visibility>=0.3) / yellow (<0.3, label=visibility), pred red conf>=0.35")
    cv2.imwrite(str(OUT / "b_occlusion_crowd.jpg"), view, [cv2.IMWRITE_JPEG_QUALITY, 90])
    det = json.load(open("reports/metrics/detection.json"))["results"]["overall"]["recall_by_visibility"]
    tr = json.load(open("reports/metrics/tracking_conf_ablation.json"))["configs"]
    res["b_occlusion"] = {
        "selection_rule": "eval frame (every 5th) with the most GT persons of visibility < 0.3", "sequence": seq, "frame": fr, "crop_xyxy": list(win),
        "this_frame": {"gt": int(len(g["vis"])), "gt_vis_lt_0.3": int(occ.sum()), "gt_vis_lt_0.3_hit": int((h & occ).sum()),
                       "gt_vis_ge_0.3": int((~occ).sum()), "gt_vis_ge_0.3_hit": int((h & ~occ).sum())},
        "detector_recall_by_visibility_conf_0.35_all_frames": det,
        "tracker_output_recall_by_visibility": {k: v["recall_by_visibility_from_tracker_output"] for k, v in tr.items()}}


# --------------------------------------------------------------------------------------------- c) ID switch
def switches(tracker_dir: str) -> list[dict]:
    import motmetrics as mm
    out = []
    for seq in SEQS:
        gt = load_gt(seq); gv, gi = per_frame(gt["valid"]), per_frame(gt["ignore"])
        pr = load_mot_results(Path(tracker_dir) / f"{seq}.txt")
        acc = mm.MOTAccumulator(auto_id=False)
        for fr in range(1, seq_info(seq)["length"] + 1):
            g, p = gv.get(fr), pr.get(fr)
            gids = g["id"] if g else np.zeros(0, int); gx = g["xyxy"] if g else np.zeros((0, 4))
            px, pid = (drop_ignore_matches(p, g, gi.get(fr))[:2]) if p else (np.zeros((0, 4)), np.zeros(0, int))
            iou = iou_matrix(gx, px)
            acc.update(gids, pid, np.where(iou >= 0.5, 1.0 - iou, np.nan), frameid=fr)
        ev = acc.mot_events
        sw = ev[ev["Type"] == "SWITCH"]
        for (fr, _), row in sw.iterrows():
            out.append({"seq": seq, "frame": int(fr), "gt_id": int(row["OId"]), "new_hyp": int(row["HId"])})
        out_matches = ev[ev["Type"].isin(["MATCH", "SWITCH"])].reset_index()
        for s in [o for o in out if o["seq"] == seq]:
            m = out_matches[out_matches["OId"] == s["gt_id"]]
            m = m[m["FrameId"] <= s["frame"]]
            prev = m[m["FrameId"] < s["frame"]]
            s["prev_hyp"] = int(prev.iloc[-1]["HId"]) if len(prev) else None
            s["last_prev_frame"] = int(prev.iloc[-1]["FrameId"]) if len(prev) else None
            allm = out_matches[out_matches["OId"] == s["gt_id"]]
            s["frames_matched_before_with_prev"] = int((allm[(allm["FrameId"] < s["frame"]) & (allm["HId"] == s["prev_hyp"])]).shape[0])
            s["frames_matched_after_with_new"] = int((allm[(allm["FrameId"] >= s["frame"]) & (allm["HId"] == s["new_hyp"])]).shape[0])
    return out


def case_c(res: dict) -> None:
    sw = switches("reports/metrics/mot_results/bytetrack")
    reported = json.load(open("reports/metrics/tracking.json"))
    reported = reported["trackers"]["bytetrack"]["scores"]["OVERALL"]["num_switches"]
    cands = []
    for s in sw:
        if s["prev_hyp"] is None or s["frames_matched_before_with_prev"] < 20 or s["frames_matched_after_with_new"] < 20:
            continue
        g = per_frame(load_gt(s["seq"])["valid"])
        gap = s["frame"] - s["last_prev_frame"]
        cur = g.get(s["frame"]); prevf = g.get(s["last_prev_frame"])
        if cur is None or s["gt_id"] not in cur["id"]:
            continue
        me = cur["xyxy"][cur["id"] == s["gt_id"]][0:1]
        others = cur["xyxy"][cur["id"] != s["gt_id"]]
        ov = float(iou_matrix(me, others).max()) if len(others) else 0.0
        s.update(gap_frames=int(gap), max_overlap_with_other_gt=ov,
                 vis_at_switch=float(cur["vis"][cur["id"] == s["gt_id"]][0]))
        cands.append(s)
    # rule: overlap with another GT person highest; ties -> longer tracking on both sides
    cands.sort(key=lambda s: (-s["max_overlap_with_other_gt"], -(s["frames_matched_before_with_prev"] + s["frames_matched_after_with_new"])))
    s = cands[0]
    seq, f0 = s["seq"], s["frame"]
    gv = per_frame(load_gt(seq)["valid"]); gi = per_frame(load_gt(seq)["ignore"]); pr = load_mot_results(Path("reports/metrics/mot_results/bytetrack") / f"{seq}.txt")
    frames = sorted({s["last_prev_frame"], max(1, f0 - 1), f0, min(seq_info(seq)["length"], f0 + 10)})
    g0 = gv[f0]; me = g0["xyxy"][g0["id"] == s["gt_id"]][0]
    ov = g0["xyxy"][g0["id"] != s["gt_id"]]; j = int(np.argmax(iou_matrix(me[None], ov)[0]))
    union = np.array([min(me[0], ov[j][0]), min(me[1], ov[j][1]), max(me[2], ov[j][2]), max(me[3], ov[j][3])])
    pad = 0.25 * max(union[2] - union[0], union[3] - union[1])
    info = seq_info(seq)
    win = (int(max(0, union[0] - pad)), int(max(0, union[1] - pad)), int(min(info["width"], union[2] + pad)), int(min(info["height"], union[3] + pad)))
    sc = 400 / (win[3] - win[1])
    panels = []
    for fr in frames:
        img = cv2.imread(str(image_path(seq, fr))); g = gv.get(fr); p = pr.get(fr)
        gx = g["xyxy"] if g else np.zeros((0, 4)); gids = g["id"] if g else np.zeros(0, int)
        px, pid = (drop_ignore_matches(p, g, gi.get(fr))[:2]) if p else (np.zeros((0, 4)), np.zeros(0, int))
        keep = lambda b: ((b[:, 0] + b[:, 2]) / 2 >= win[0]) & ((b[:, 0] + b[:, 2]) / 2 < win[2]) & ((b[:, 1] + b[:, 3]) / 2 >= win[1]) & ((b[:, 1] + b[:, 3]) / 2 < win[3])
        # draw only the tracked GT person, the GT person it overlaps at the switch frame, and predictions touching either
        focus_ids = {s["gt_id"], int(g0["id"][g0["id"] != s["gt_id"]][j])}
        gk = np.array([i in focus_ids for i in gids], bool) if len(gids) else np.zeros(0, bool)
        pk = (iou_matrix(px, gx[gk]).max(1) >= 0.3) if (len(px) and gk.any()) else np.zeros(len(px), bool)
        panels.append(crop_view(img, win, sc, gx[gk], px[pk], gt_label=lambda i, ids=gids[gk]: f"GT{ids[i]}", pred_label=lambda i, ids=pid[pk]: f"T{ids[i]}",
                                title=(f"f{fr}: last match GT{s['gt_id']}=T{s['prev_hyp']}" if fr == s["last_prev_frame"] else f"f{fr}: SWITCH GT{s['gt_id']}=T{s['new_hyp']}" if fr == f0 else f"f{fr}: {fr - f0:+d} frames vs switch")))
    cv2.imwrite(str(OUT / "c_id_switch.jpg"), np.hstack(panels), [cv2.IMWRITE_JPEG_QUALITY, 90])
    res["c_id_switch"] = {"selection_rule": "among ByteTrack (conf 0.35) ID switches with >=20 matched frames before (old ID) and after (new ID), the one whose GT box overlaps another GT person the most at the switch frame",
                          "total_switches_recomputed": len(sw), "total_switches_in_tracking_json": reported, "chosen": s, "frames_shown": frames,
                          "crop_xyxy": list(win)}


# --------------------------------------------------------------------------------------------- d) rainy night clip
def load_preds(run: str) -> tuple[list[dict], str]:
    f = glob.glob(f"alerts/{run}/*/predictions.jsonl")[0]
    return [json.loads(l) for l in open(f)], f


def case_d(res: dict) -> None:
    rows, _ = load_preds("analysis_4332")
    tracks: dict[int, dict] = defaultdict(lambda: {"frames": [], "classes": Counter()})
    for r in rows:
        for d in r["detections"]:
            if d["track_id"] is not None:
                t = tracks[d["track_id"]]; t["frames"].append(r["frame_idx"]); t["classes"][d["class_name"]] += 1
    lens = np.array([len(t["frames"]) for t in tracks.values()])
    short = {k for k, t in tracks.items() if len(t["frames"]) < 15}
    flips = {k: t for k, t in tracks.items() if len(t["classes"]) > 1}
    # flicker = track with gaps (frames missing inside its life)
    gappy = {k for k, t in tracks.items() if (t["frames"][-1] - t["frames"][0] + 1) > len(t["frames"])}
    # flip example: track with the largest minority-class frame count
    ex = max(flips, key=lambda k: sorted(flips[k]["classes"].values())[-2]) if flips else None
    cap = cv2.VideoCapture("data/videos/mixkit-4332.mp4")

    def frame_at(i):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i - 1); ok, im = cap.read(); return im

    def preds_at(i):
        return next(r for r in rows if r["frame_idx"] == i)["detections"]
    out = {"tracks_total": len(tracks), "frames_in_clip": len(rows), "tracks_shorter_than_15_frames": len(short),
           "median_track_length_frames": float(np.median(lens)), "tracks_with_gaps": len(gappy),
           "tracks_with_more_than_one_class": len(flips),
           "class_flip_breakdown": Counter("/".join(sorted(t["classes"])) for t in flips.values()),
           "boxes_total": int(lens.sum()), "boxes_in_short_tracks": int(sum(len(tracks[k]["frames"]) for k in short))}
    out["class_flip_breakdown"] = dict(out["class_flip_breakdown"])
    if ex is not None:
        t = flips[ex]; cls = [c for c, _ in t["classes"].most_common(2)]
        panels = []
        for c in cls:  # one frame per class: the middle frame of that class' appearances
            fs = [r["frame_idx"] for r in rows for d in r["detections"] if d["track_id"] == ex and d["class_name"] == c]
            fr = fs[len(fs) // 2]; img = frame_at(fr)
            d0 = next(d for d in preds_at(fr) if d["track_id"] == ex)
            w, h = d0["x2"] - d0["x1"], d0["y2"] - d0["y1"]
            win = (int(max(0, d0["x1"] - w)), int(max(0, d0["y1"] - h * 0.5)), int(min(img.shape[1], d0["x2"] + w)), int(min(img.shape[0], d0["y2"] + h * 0.5)))
            sc = 420 / (win[3] - win[1])
            others = [d for d in preds_at(fr) if d["track_id"] != ex]
            view = crop_view(img, win, sc, (), [[d0["x1"], d0["y1"], d0["x2"], d0["y2"]]], pred_label=lambda i, d0=d0: f"#{ex} {d0['class_name']} {d0['confidence']:.2f}", title=f"track #{ex}, frame {fr}: {c}")
            panels.append(view)
        hh = min(p.shape[0] for p in panels)
        cv2.imwrite(str(OUT / "d1_class_flip.jpg"), np.hstack([p[:hh] for p in panels]), [cv2.IMWRITE_JPEG_QUALITY, 90])
        out["flip_example"] = {"selection_rule": "track with >1 class whose second most frequent class has the most frames", "track_id": int(ex),
                               "classes": dict(t["classes"]), "first_frame": t["frames"][0], "last_frame": t["frames"][-1]}
    # short-lived tracks on a busy frame (the frame with the most distinct short tracks)
    cnt = Counter()
    for r in rows:
        cnt[r["frame_idx"]] = sum(1 for d in r["detections"] if d["track_id"] in short)
    fr = cnt.most_common(1)[0][0]; img = frame_at(fr)
    for d in preds_at(fr):
        isshort = d["track_id"] in short
        box(img, [d["x1"], d["y1"], d["x2"], d["y2"]], YELLOW if isshort else RED, f"#{d['track_id']} {d['class_name']}", 2)
    banner(img, f"frame {fr}: yellow = tracks that live < 15 frames in the whole clip, red = longer tracks")
    cv2.imwrite(str(OUT / "d2_short_tracks.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    out["short_tracks_example_frame"] = {"frame": fr, "short_tracks_on_frame": cnt[fr]}
    # umbrellas: unfiltered detector, every 10th frame
    umb_frames, umb_total, best = 0, 0, (0, 1)
    sampled = list(range(1, len(rows) + 1, 10))
    for i in sampled:
        r = model("yolo11s.onnx").predict(frame_at(i), conf=0.35, iou=0.5, imgsz=640, verbose=False)[0]
        names = [r.names[int(c)] for c in r.boxes.cls.cpu().numpy()]
        n = names.count("umbrella"); umb_total += n; umb_frames += n > 0
        best = max(best, (n, i))
    out["umbrella_unfiltered_detector"] = {"sampled_frames": len(sampled), "frames_with_umbrella_box": umb_frames, "umbrella_boxes": umb_total,
                                           "in_demo_class_filter": False, "frame_with_most": best[1]}
    img = frame_at(best[1])
    r = model("yolo11s.onnx").predict(img, conf=0.35, iou=0.5, imgsz=640, verbose=False)[0]
    for b, c, cf in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy()):
        nm = r.names[int(c)]
        if nm in ("umbrella", "person"):
            box(img, b, BLUE if nm == "umbrella" else RED, f"{nm} {cf:.2f}", 2)
    banner(img, f"frame {best[1]}: unfiltered YOLO11s conf>=0.35, umbrella blue, person red")
    cv2.imwrite(str(OUT / "d3_umbrellas.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 3.2)); ax.hist(np.minimum(lens, 120), bins=40, color="#4477aa")
    ax.set_xlabel("frames a track ID is visible (capped at 120)"); ax.set_ylabel("tracks"); ax.set_title(f"clip 4332: {len(tracks)} track IDs in {len(rows)} frames")
    fig.tight_layout(); fig.savefig(OUT / "d4_track_length_hist.png", dpi=130)
    res["d_night_clip"] = out


# --------------------------------------------------------------------------------------------- e) collision rule
def case_e(res: dict) -> None:
    from src.infer.runtime import VEHICLE_CLASSES, _bbox_iou
    thr = 0.35  # configs/infer_incidents_pretrained.yaml thresholds.overlap_iou_threshold
    summary = {"overlap_iou_threshold": thr, "rule_active_in_default_demo_config": False,
               "rule_source": "src/infer/runtime.py _update_collision_candidates (called only from _update_incident_rules, i.e. mode incident_enhanced)", "clips": {}}
    pairs_all = []
    for clip in ("61", "4332"):
        rows, _ = load_preds(f"analysis_{clip}")
        n_frames, pairs = 0, []
        for r in rows:
            veh = [d for d in r["detections"] if d["class_name"] in VEHICLE_CLASSES and d["track_id"] is not None]
            hit = False
            for i, a in enumerate(veh):
                for b in veh[i + 1:]:
                    iou = _bbox_iou(a, b)
                    if iou >= thr:
                        pairs.append({"clip": clip, "frame": r["frame_idx"], "ids": [a["track_id"], b["track_id"]], "classes": [a["class_name"], b["class_name"]], "iou": iou, "a": a, "b": b}); hit = True
            n_frames += hit
        summary["clips"][clip] = {"frames": len(rows), "frames_with_vehicle_pair_iou_ge_thr": n_frames, "pairs": len(pairs),
                                  "distinct_pairs": len({tuple(sorted(p["ids"])) for p in pairs})}
        pairs_all += pairs
    summary["candidates_top10"] = [{k: v for k, v in p.items() if k not in ("a", "b")} for p in sorted(pairs_all, key=lambda p: -p["iou"])[:10]]
    res["e_collision_rule"] = summary
    # candidate sheet for manual inspection (the final pick is stored in docs by pick_e)
    (OUT / "_e_candidates.json").write_text(json.dumps([{k: v for k, v in p.items() if k not in ("a", "b")} for p in pairs_all], indent=1))


def render_e(res: dict, picks: list[tuple[str, int, list[int]]]) -> None:
    from src.infer.runtime import _bbox_iou
    panels, chosen = [], []
    for clip, frame, ids in picks:
        rows, _ = load_preds(f"analysis_{clip}")
        r = next(r for r in rows if r["frame_idx"] == frame)
        cap = cv2.VideoCapture(f"data/videos/mixkit-{clip}.mp4"); cap.set(cv2.CAP_PROP_POS_FRAMES, frame - 1); _, img = cap.read()
        a, b = [next(d for d in r["detections"] if d["track_id"] == i) for i in ids]
        iou = _bbox_iou(a, b)
        x1, y1 = min(a["x1"], b["x1"]), min(a["y1"], b["y1"]); x2, y2 = max(a["x2"], b["x2"]), max(a["y2"], b["y2"])
        pad = 0.25 * max(x2 - x1, y2 - y1)
        win = (int(max(0, x1 - pad)), int(max(0, y1 - pad)), int(min(img.shape[1], x2 + pad)), int(min(img.shape[0], y2 + pad)))
        sc = 420 / (win[3] - win[1])
        view = cv2.resize(img[win[1]:win[3], win[0]:win[2]], None, fx=sc, fy=sc, interpolation=cv2.INTER_CUBIC)
        for k, (d, col) in enumerate(((a, RED), (b, YELLOW))):
            box(view, [d["x1"], d["y1"], d["x2"], d["y2"]], col, None, 3, sc, win[:2])
            cv2.putText(view, f"#{d['track_id']} {d['class_name']}", (8, 52 + 22 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)
        panels.append(banner(view, f"clip {clip} frame {frame}: box IoU {iou:.2f} (rule >= 0.35)"))
        chosen.append({"clip": clip, "frame": frame, "track_ids": ids, "iou": iou, "classes": [a["class_name"], b["class_name"]]})
    cv2.imwrite(str(OUT / "e_collision_iou.jpg"), np.hstack(panels), [cv2.IMWRITE_JPEG_QUALITY, 90])
    res.setdefault("e_collision_rule", {})["examples_in_image"] = chosen


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="a,b,c,d,e")
    ap.add_argument("--e-pick", default=None, help="comma-separated clip:frame:idA:idB for the collision montage")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    path = Path("reports/metrics/error_analysis.json")
    res = json.loads(path.read_text()) if path.exists() else {}
    for c in [c for c in args.cases.split(",") if c]:
        {"a": case_a, "b": case_b, "c": case_c, "d": case_d, "e": case_e}[c](res)
        print("done", c, flush=True)
    if args.e_pick:
        render_e(res, [(c, int(f), [int(i), int(j)]) for c, f, i, j in (x.split(":") for x in args.e_pick.split(","))])
    path.write_text(json.dumps(res, indent=2, default=float))


if __name__ == "__main__":
    main()
