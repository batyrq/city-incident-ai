"""Render the incident-detection docs from the JSON files (no number is typed by hand):
  data/incidents/SOURCES.md + docs/INCIDENT_DATA.md (datasets, licences, counts) and the README block between the incidents markers.
  python -m eval.incidents_docs
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
M = ROOT / "reports/metrics"
J = lambda n: json.loads((M / n).read_text())
inc, lat, vid, dat, aud, ytr, prm = (J(x) for x in ("incidents.json", "incident_latency.json", "incident_videos.json", "incident_data.json", "incident_crash_audit.json", "incident_yolo_train.json", "incident_prompt_probe.json"))
dcfg = json.loads((ROOT / "configs/incident_detectors.json").read_text())["classes"]
pct = lambda v: "–" if v is None else f"{v * 100:.1f}%"
C = ["fire", "smoke", "flood", "crash"]
NM = {"A": "A YOLO-World zero-shot", "B": "B CLIP zero-shot", "D": "D ready community weights", "C": "C trained by me"}
DN = {"fire": "rabahdev/fire-smoke-yolov8n", "smoke": "rabahdev/fire-smoke-yolov8n", "flood": "prithivMLmods/Flood-Image-Detection", "crash": "Enos-123/traffic-accident-detection-yolo11x"}
CN = {"fire": "YOLO11n fine-tuned on D-Fire", "smoke": "YOLO11n fine-tuned on D-Fire", "flood": "linear probe on CLIP", "crash": "linear probe on CLIP"}

# ---------------------------------------------------------------- data docs
ic = dat["image_counts"]
cnt = lambda t, s: ", ".join(f"{k.replace('neg_', 'neg ')} {v}" for k, v in ic[f"{t}/{s}"].items())
src = f"""# Incident datasets

Built by `python -m eval.incidents_data` (raw downloads: `python -m eval.incidents_download`, `python -m eval.incidents_clips`). Images are stored resized to at most {dat['max_side_px']} px.
Splits never share a source clip; test items are dropped when a perceptual hash (dHash, Hamming <= 4) matches a training image
(removed from eval: {", ".join(f"{k} {v}" for k, v in dat['near_duplicates_removed_from_eval'].items())}).

| Class | Positives from | Licence | Negatives from |
|---|---|---|---|
| fire, smoke | D-Fire via `hanvithSai/gavin-dfire-raw` (mirror of gaiasd/DFireDataset): official train {dat['dfire_official']['train']['images']} imgs (holdout {dat['dfire_holdout_val']} for validation, {dat['dfire_train_used_for_yolo']} for YOLO training), official test {dat['dfire_official']['test']['images']} imgs | CC BY 4.0; some images come from third-party sources (HPWREN, FESB, FireNet) under their own terms | D-Fire images with no label ("other"), MOT17-02/04/09 and demo clips a-d ("street"), Mixkit sunsets, car lights, neon, steam, ambulances ("hard") |
| flood | `zinnia82/flood-binary-hnm-benchmark` (street-level flood vs non-flood; its `val` is my test, its `train` is split 80/20 into train/val) | **no licence stated** (images not redistributed here) | its non-flood categories: street_clear (street), river, swimming_pool, park_walkway (hard), the rest (other), plus Mixkit rain, traffic, rivers, fountains |
| crash | `justjuu/traffic-accident-cctv-object-detection` accident frames (CCTV/dashcam) | CC0 | `hiennguyen9874/traffic-accident-detection` non-accident photos (**no licence stated**, not redistributed) as street, Mixkit car lights, traffic jams, ambulances, hazard lights, rain as hard |

Videos for the video-level check: Mixkit (Stock Video Free License) and Wikimedia Commons (licence per clip in `demo_videos/incidents/README.md`).

## Counts (written by the script)

| Task / split | Counts |
|---|---|
""" + "\n".join(f"| {t}/{s} | {cnt(t, s)} |" for t in C for s in ("train", "val", "test") if f"{t}/{s}" in ic) + "\n"
(ROOT / "docs/INCIDENT_DATA.md").write_text(src)
(ROOT / "data/incidents").mkdir(exist_ok=True)
(ROOT / "data/incidents/SOURCES.md").write_text(src)

# ---------------------------------------------------------------- README block
L = ["<!-- incidents:start -->", "",
     "Test-split results (image level). **No training:** A open-vocabulary YOLO-World, B CLIP ViT-B/16 zero-shot, D ready-made community weights. **Trained by me:** C "
     f"(fire/smoke: YOLO11n fine-tuned on D-Fire, {ytr['epochs_completed']} epochs in {ytr['wall_minutes']} min on MPS; flood/crash: logistic regression on frozen CLIP embeddings). "
     "Thresholds come from the validation split. Default per class (bold) = best validation F1, but a no-training approach wins when within 0.03 F1.", "",
     "| Class | Approach | Precision | Recall | F1 | ROC-AUC | FPR street | FPR hard neg. | ms/img |", "|---|---|---|---|---|---|---|---|---|"]
for c in C:
    for ap in ("A", "B", "D", "C"):
        m = inc[c].get(ap)
        if not m:
            continue
        nm = {"A": NM["A"], "B": NM["B"], "D": f"D `{DN[c]}`", "C": f"C {CN[c]}"}[ap]
        b = "**" if dcfg[c]["letter"] == ap else ""
        L.append(f"| {c} | {b}{nm}{b} | {pct(m['precision'])} | {pct(m['recall'])} | {m['f1']:.3f} | {m['roc_auc']:.3f} | {pct(m['fpr_street'])} | {pct(m['fpr_hard'])} | {lat['ms'][c][ap]:.0f} |")
vs = vid["summary"]
n = inc["fire"]["B"]
L += ["", f"Test sets: {n['n_pos']} positives per class (crash {inc['crash']['B']['n_pos']} after de-duplication) and {n['n_neg_street'] + n['n_neg_hard'] + n['n_neg_background']} negatives for fire/smoke; counts for every split in [docs/INCIDENT_DATA.md](docs/INCIDENT_DATA.md). "
      f"Hard negatives: sunsets, car lights, neon, steam, rain, rivers, ambulances. Failure images with reasons: [docs/incident_failures/](docs/incident_failures/README.md); confusion matrices: `docs/incident_confusion/`.",
      f"Prompt check for A: YOLO-World never separated fire (best ROC-AUC {max(x['roc_auc'] for x in prm['fire']):.2f} over {len(prm['fire'])} wordings) and smoke only weakly ({max(x['roc_auc'] for x in prm['smoke']):.2f}).", "",
      f"**Video level** (`reports/metrics/incident_videos.json`, {vs['incident_clips']} real incident clips, {vs['normal_clips']} normal clips = {vs['normal_seconds_scored']:.0f} s): {vs['detected']} of {vs['incident_clips']} incident clips fired the expected class, "
      f"but {vs['detected_before_incident_started']} of those fired *before* the incident started (the two crash clips). {vs['false_alarms_total']} false alarms on normal clips "
      + "(" + "; ".join(f"{k.replace('demo_videos/', '')}: " + ", ".join(f"{c} at {v[0]:.1f} s" for c, v in r["false_alarms"].items()) for k, r in vid["normal_clips"].items() if r["false_alarms"]) + "). "
      f"Fire clips also fire `smoke` and vice versa, which is plausible: they show both.", "",
      "**The crash detector does not work on real video.** " + f"The probe scores {aud['clips']['control c4 dashcam highway, no collision seen']['mean_C']:.2f} on average on a dashcam highway clip and {aud['clips']['control c5 street CCTV, no collision seen']['mean_C']:.2f} on a street CCTV clip where I saw no collision, "
      f"against {aud['clips']['crash_dashcam_highway (crash 2-7 s)']['mean_C']:.2f} on a dashcam crash clip: it learned camera style (positives are dashcam/CCTV frames, negatives mostly ordinary photos). "
      "The high image-level crash F1 is therefore a dataset artefact. Treat `crash_candidate` as experimental. Details: `reports/metrics/incident_crash_audit.json`.", "",
      "Ready-made weights (D): `rabahdev/fire-smoke-yolov8n` (AGPL-3.0, trained on the D-Fire official train split, so D-Fire test images were not in its training; my validation holdout was, so its validation threshold is optimistic and D was excluded from the default choice for fire/smoke), "
      "`prithivMLmods/Flood-Image-Detection` (Apache-2.0, SigLIP2 fine-tuned on TheNetherWatcher/DisasterClassification), `Enos-123/traffic-accident-detection-yolo11x` (MIT, trained on a Roboflow accident dataset that may overlap the CCTV accident frames I test on, so its crash numbers may be optimistic). D is evaluated but not wired into the pipeline.", "",
      "<!-- incidents:end -->"]
block = "\n".join(L)
r = (ROOT / "README.md").read_text()
r = re.sub(r"<!-- incidents:start -->.*?<!-- incidents:end -->", lambda _: block, r, flags=re.S)
(ROOT / "README.md").write_text(r)
print("ok")
