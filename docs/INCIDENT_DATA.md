# Incident datasets

Built by `python -m eval.incidents_data` (raw downloads: `python -m eval.incidents_download`, `python -m eval.incidents_clips`). Images are stored resized to at most 640 px.
Splits never share a source clip; test items are dropped when a perceptual hash (dHash, Hamming <= 4) matches a training image
(removed from eval: fire:positive 149, fire:negative 469, smoke:positive 197, smoke:negative 519, crash:positive 157, flood:positive 7, flood:negative 27).

| Class | Positives from | Licence | Negatives from |
|---|---|---|---|
| fire, smoke | D-Fire via `hanvithSai/gavin-dfire-raw` (mirror of gaiasd/DFireDataset): official train 17221 imgs (holdout 1685 for validation, 15536 for YOLO training), official test 4306 imgs | CC BY 4.0; some images come from third-party sources (HPWREN, FESB, FireNet) under their own terms | D-Fire images with no label ("other"), MOT17-02/04/09 and demo clips a-d ("street"), Mixkit sunsets, car lights, neon, steam, ambulances ("hard") |
| flood | `zinnia82/flood-binary-hnm-benchmark` (street-level flood vs non-flood; its `val` is my test, its `train` is split 80/20 into train/val) | **no licence stated** (images not redistributed here) | its non-flood categories: street_clear (street), river, swimming_pool, park_walkway (hard), the rest (other), plus Mixkit rain, traffic, rivers, fountains |
| crash | `justjuu/traffic-accident-cctv-object-detection` accident frames (CCTV/dashcam) | CC0 | `hiennguyen9874/traffic-accident-detection` non-accident photos (**no licence stated**, not redistributed) as street, Mixkit car lights, traffic jams, ambulances, hazard lights, rain as hard |

Videos for the video-level check: Mixkit (Stock Video Free License) and Wikimedia Commons (licence per clip in `demo_videos/incidents/README.md`).

## Counts (written by the script)

| Task / split | Counts |
|---|---|
| fire/train | neg street 80, neg hard 240 |
| fire/val | neg street 40, neg hard 96, pos 200, neg background 264 |
| fire/test | neg street 160, neg hard 120, pos 300, neg background 320 |
| smoke/train | neg street 80, neg hard 240 |
| smoke/val | neg street 40, neg hard 96, pos 200, neg background 264 |
| smoke/test | neg street 160, neg hard 120, pos 300, neg background 320 |
| flood/train | neg street 458, neg hard 727, pos 1033, neg background 620 |
| flood/val | neg street 100, neg hard 183, pos 200, neg background 117 |
| flood/test | neg street 233, neg hard 233, pos 300, neg background 134 |
| crash/train | neg street 408, neg hard 180, pos 1500 |
| crash/val | neg street 172, neg hard 96, pos 200 |
| crash/test | neg street 356, neg hard 108, pos 262 |
