"""Sanity check for approach A on fire/smoke: does YOLO-World (and YOLOE, if it loads) score fire at all, for several prompt wordings?
Runs on 60 val positives + 120 val negatives (CPU), writes reports/metrics/incident_prompt_probe.json."""
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from ultralytics import YOLOWorld

ROOT = Path(__file__).resolve().parent.parent
out = {}
for task in ("fire", "smoke"):
    rows = [x for x in csv.DictReader(open(ROOT / f"data/incidents/scores/A_yolo_world_{task}.csv")) if x["split"] == "val"]
    sel = [x for x in rows if x["label"] == "1"][:60] + [x for x in rows if x["label"] == "0"][:120]
    y = np.array([int(x["label"]) for x in sel])
    variants = [["fire"], ["flame"], ["a fire"], ["burning fire", "fire flames"], ["wildfire"]] if task == "fire" else [["smoke"], ["a plume of smoke"], ["smoke cloud"], ["wildfire smoke"]]
    out[task] = []
    for prompts in variants:
        m = YOLOWorld(str(ROOT / "yolov8s-world.pt")); m.set_classes(prompts)
        s = []
        for x in sel:
            r = m.predict(str(ROOT / x["path"]), conf=0.0001, imgsz=640, device="cpu", verbose=False)[0]
            s.append(float(r.boxes.conf.max()) if len(r.boxes) else 0.0)
        s = np.array(s)
        out[task].append({"prompts": prompts, "roc_auc": round(float(roc_auc_score(y, s)), 3), "positives_with_any_box": int((s[y == 1] > 0).sum()), "n_pos": int((y == 1).sum()),
                          "negatives_with_any_box": int((s[y == 0] > 0).sum()), "n_neg": int((y == 0).sum()), "max_score_on_positives": round(float(s[y == 1].max()), 4), "conf_floor": 0.0001})
(ROOT / "reports/metrics/incident_prompt_probe.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=1))
