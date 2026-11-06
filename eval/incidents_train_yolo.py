"""Fine-tune YOLO11n on the D-Fire train split (minus a 10 % holdout) on Apple MPS, time-boxed.

  python -m eval.incidents_train_yolo [hours]     -> runs/incidents/dfire_yolo11n/weights/best.pt + reports/metrics/incident_yolo_train.json
"""
import json
import sys
import time
from pathlib import Path

from ultralytics import YOLO

hours = float(sys.argv[1]) if len(sys.argv) > 1 else 0.75
t0 = time.time()
m = YOLO("yolo11n.pt")
r = m.train(data="data/incidents/dfire_yolo/data.yaml", imgsz=640, batch=16, workers=4, device="mps", time=hours, epochs=30,
            project=str(Path("runs/incidents").resolve()), name="dfire_yolo11n", exist_ok=True, patience=100, verbose=False, plots=False, seed=0)
res = Path(r.save_dir)
import csv
rows = list(csv.DictReader(open(res / "results.csv")))
rows = [{k.strip(): v for k, v in x.items()} for x in rows]
Path("reports/metrics/incident_yolo_train.json").write_text(json.dumps({
    "model": "yolo11n.pt", "imgsz": 640, "batch": 16, "device": "mps", "time_budget_hours": hours, "epochs_completed": len(rows),
    "wall_minutes": round((time.time() - t0) / 60, 1), "last_epoch": rows[-1], "weights": str(res / "weights/best.pt")}, indent=2))
print("done", len(rows), "epochs")
