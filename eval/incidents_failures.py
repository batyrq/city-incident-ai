"""Pick the most instructive test-set mistakes of the selected approach per class: the 3 most confident false positives
(at most 2 from one negative group, so hard negatives and street negatives both show up) and the 3 lowest-scored false negatives.

  python -m eval.incidents_failures    -> docs/incident_failures/<class>_{FP,FN}<i>_s<score>.jpg + docs/incident_failures/failures.json
The one-line reasons are written by hand in docs/incident_failures/README.md after looking at each image.
"""
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs/incident_failures"
LOCAL = ROOT / "data/incidents/failures_unlicensed"
APPROACH_FILE = {"yolo": "C_yolo11n_dfire", "probe": "C_clip_probe", "clip_zeroshot": "B_clip_zeroshot"}


def main() -> None:
    cfg = json.loads((ROOT / "configs/incident_detectors.json").read_text())["classes"]
    shutil.rmtree(OUT, ignore_errors=True)
    OUT.mkdir(parents=True)
    meta = {}
    src_of = {r["path"]: r["source"] for r in csv.DictReader(open(ROOT / "data/incidents/manifest.csv"))}
    for task, spec in cfg.items():
        rows = [r for r in csv.DictReader(open(ROOT / f"data/incidents/scores/{APPROACH_FILE[spec['approach']]}_{task}.csv")) if r["split"] == "test"]
        thr = spec["threshold"]
        fps = sorted([r for r in rows if r["label"] == "0" and float(r["score"]) >= thr], key=lambda r: -float(r["score"]))
        fns = sorted([r for r in rows if r["label"] == "1" and float(r["score"]) < thr], key=lambda r: float(r["score"]))
        pick_fp, per_group = [], {}
        for r in fps:
            if per_group.get(r["group"], 0) < 2 and len(pick_fp) < 3:
                pick_fp.append(r)
                per_group[r["group"]] = per_group.get(r["group"], 0) + 1
        meta[task] = {"approach": spec["approach"], "threshold": thr, "n_fp": len(fps), "n_fn": len(fns), "items": []}
        for kind, items in (("FP", pick_fp), ("FN", fns[:3])):
            for i, r in enumerate(items, 1):
                im = Image.open(ROOT / r["path"]).convert("RGB")
                s = 480 / im.width
                im = im.resize((480, round(im.height * s)))
                d = ImageDraw.Draw(im)
                label = f"{task} {kind}: score {float(r['score']):.3f} (threshold {thr:.3f})"
                d.rectangle([0, 0, im.width, 18], fill=(0, 0, 0))
                d.text((4, 3), label, fill=(255, 255, 255))
                name = f"{task}_{kind}{i}_s{float(r['score']):.2f}.jpg"
                local_only = r["path"] and any(k in src_of.get(r["path"], "") for k in ("zinnia82", "hiennguyen9874"))  # these two datasets state no licence: keep their images out of git
                (LOCAL if local_only else OUT).mkdir(parents=True, exist_ok=True)
                im.save((LOCAL if local_only else OUT) / name, quality=88)
                meta[task]["items"].append({"file": name, "kind": kind, "score": float(r["score"]), "group": r["group"], "source": src_of.get(r["path"], ""), "path": r["path"], "committed": not local_only})
    (OUT / "failures.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps({t: {k: v for k, v in m.items() if k != "items"} for t, m in meta.items()}))


if __name__ == "__main__":
    main()
