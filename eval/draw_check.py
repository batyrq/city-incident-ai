"""Overlay GT (green valid, yellow ignore regions) and predictions (red) on random frames to verify
coordinate/frame-index alignment. Usage: python -m eval.draw_check [--seed 0]"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np

from eval.eval_tracking import OUT_DIR, load_mot_results
from eval.mot_common import SEQS, image_path, load_gt, per_frame, seq_info


def box(img, b, color, label=None, th=2):
    x1, y1, x2, y2 = [int(round(v)) for v in b]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, th)
    if label:
        cv2.putText(img, label, (x1, max(12, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="docs/eval_check")
    args = ap.parse_args()
    rng = random.Random(args.seed)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    for seq in SEQS:
        gt = load_gt(seq); gv, gi = per_frame(gt["valid"]), per_frame(gt["ignore"])
        dets = per_frame({"frame": (d := np.load(f"data/mot/cache/dets_{seq}.npz")["dets"])[:, 0].astype(int), "xyxy": d[:, 1:5], "conf": d[:, 5]})
        trk = load_mot_results(OUT_DIR / "bytetrack" / f"{seq}.txt")
        for fr in sorted(rng.sample(range(1, seq_info(seq)["length"] + 1), 3)):
            for kind in ("det", "track"):
                img = cv2.imread(str(image_path(seq, fr)))
                for b in (gi.get(fr) or {"xyxy": []})["xyxy"]:
                    box(img, b, (0, 255, 255), th=1)
                for b in (gv.get(fr) or {"xyxy": []})["xyxy"]:
                    box(img, b, (0, 255, 0))
                if kind == "det" and fr in dets:
                    for b, c in zip(dets[fr]["xyxy"], dets[fr]["conf"]):
                        if c >= 0.35:
                            box(img, b, (0, 0, 255))
                if kind == "track" and fr in trk:
                    for b, i in zip(trk[fr]["xyxy"], trk[fr]["id"]):
                        box(img, b, (0, 0, 255), label=str(i))
                n_gt = len(gv.get(fr, {"xyxy": []})["xyxy"])
                cv2.putText(img, f"{seq} frame {fr} {kind}  GT valid={n_gt} (green) pred (red) ignore (yellow)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                cv2.imwrite(f"{args.out}/{seq}_f{fr:04d}_{kind}.jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    print("wrote", args.out)


if __name__ == "__main__":
    main()
