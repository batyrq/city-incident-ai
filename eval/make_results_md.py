"""Render markdown result tables from reports/metrics/{detection,tracking}.json (prints to stdout)."""
import json
from pathlib import Path

det = json.loads(Path("reports/metrics/detection.json").read_text())["results"]
trk = json.loads(Path("reports/metrics/tracking.json").read_text())
pct = lambda v: "n/a" if v is None else f"{100 * v:.1f}%"
seqs = ["MOT17-02", "MOT17-04", "MOT17-09", "overall"]

print("### Detection (person, IoU 0.5 unless noted)\n")
print("| Sequence | GT boxes | AP50 | AP50:95 | P @0.35 | R @0.35 | best-F1 thr | best F1 (P / R) |")
print("|---|---:|---:|---:|---:|---:|---:|---|")
for s in seqs:
    r = det[s]; a = r["at_conf_0.35"]; b = r["best_f1"]
    print(f"| {s} | {r['n_gt_boxes']} | {r['AP50']:.3f} | {r['AP50_95']:.3f} | {pct(a['precision'])} | {pct(a['recall'])} | {b['threshold']:.3f} | {b['f1']:.3f} ({pct(b['precision'])} / {pct(b['recall'])}) |")
print("\nRecall at conf 0.35 by GT visibility and box height (all three sequences pooled):\n")
o = det["overall"]
print("| Bucket | GT boxes | Detected | Recall |\n|---|---:|---:|---:|")
for k, v in o["recall_by_visibility"].items():
    print(f"| visibility {k} | {v['count']} | {v['detected']} | {pct(v['recall'])} |")
for k, v in o["recall_by_height_px"].items():
    print(f"| height {k} px | {v['count']} | {v['detected']} | {pct(v['recall'])} |")
print("\n### Tracking (person, conf 0.35, same detections for both trackers)\n")
print("| Tracker | Sequence | MOTA | IDF1 | IDsw | FP | FN | MT | ML | Pred IDs / GT IDs |")
print("|---|---|---:|---:|---:|---:|---:|---:|---:|---|")
for name, label in (("bytetrack", "ByteTrack"), ("botsort", "BoT-SORT (no ReID)")):
    sc = trk["trackers"][name]["scores"]
    for s in ["MOT17-02", "MOT17-04", "MOT17-09", "OVERALL"]:
        r = sc[s]
        print(f"| {label} | {s} | {pct(r['mota'])} | {pct(r['idf1'])} | {r['num_switches']} | {r['num_false_positives']} | {r['num_misses']} | {r['mostly_tracked']} | {r['mostly_lost']} | {r['pred_identities']} / {r['gt_identities']} |")
print("\nTracker-only time per frame (mean, ms; excludes the detector):\n")
print("| Tracker | " + " | ".join(["MOT17-02", "MOT17-04", "MOT17-09"]) + " | overall |\n|---|---:|---:|---:|---:|")
for name, label in (("bytetrack", "ByteTrack"), ("botsort", "BoT-SORT")):
    t = trk["trackers"][name]["timing"]
    print(f"| {label} | " + " | ".join(f"{t[s]['mean_ms']:.2f}" for s in ["MOT17-02", "MOT17-04", "MOT17-09"]) + f" | {trk['trackers'][name]['scores']['OVERALL']['tracker_ms_per_frame_mean']:.2f} |")
