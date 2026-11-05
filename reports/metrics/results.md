### Detection (person, IoU 0.5 unless noted)

| Sequence | GT boxes | AP50 | AP50:95 | P @0.35 | R @0.35 | best-F1 thr | best F1 (P / R) |
|---|---:|---:|---:|---:|---:|---:|---|
| MOT17-02 | 18581 | 0.427 | 0.225 | 92.2% | 25.2% | 0.082 | 0.502 (65.3% / 40.8%) |
| MOT17-04 | 47557 | 0.594 | 0.328 | 93.9% | 31.8% | 0.031 | 0.688 (80.2% / 60.2%) |
| MOT17-09 | 5325 | 0.715 | 0.427 | 92.0% | 69.4% | 0.370 | 0.792 (92.7% / 69.1%) |
| overall | 71463 | 0.552 | 0.303 | 93.3% | 32.9% | 0.055 | 0.628 (74.5% / 54.2%) |

Recall at conf 0.35 by GT visibility and box height (all three sequences pooled):

| Bucket | GT boxes | Detected | Recall |
|---|---:|---:|---:|
| visibility <0.3 | 22627 | 831 | 3.7% |
| visibility 0.3-0.7 | 18826 | 3226 | 17.1% |
| visibility >0.7 | 30010 | 19442 | 64.8% |
| height <50 px | 1276 | 0 | 0.0% |
| height 50-100 px | 8149 | 595 | 7.3% |
| height >100 px | 62038 | 22904 | 36.9% |

### Tracking (person, conf 0.35, same detections for both trackers)

| Tracker | Sequence | MOTA | IDF1 | IDsw | FP | FN | MT | ML | Pred IDs / GT IDs |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| ByteTrack | MOT17-02 | 22.9% | 32.3% | 37 | 322 | 13976 | 8 | 39 | 84 / 62 |
| ByteTrack | MOT17-04 | 29.0% | 42.2% | 49 | 1033 | 32661 | 8 | 45 | 99 / 83 |
| ByteTrack | MOT17-09 | 61.6% | 56.7% | 44 | 295 | 1706 | 9 | 4 | 54 / 26 |
| ByteTrack | OVERALL | 29.9% | 41.2% | 130 | 1650 | 48343 | 25 | 88 | 237 / 171 |
| BoT-SORT (no ReID) | MOT17-02 | 22.8% | 31.1% | 41 | 330 | 13979 | 8 | 39 | 85 / 62 |
| BoT-SORT (no ReID) | MOT17-04 | 29.4% | 43.1% | 46 | 961 | 32588 | 9 | 45 | 98 / 83 |
| BoT-SORT (no ReID) | MOT17-09 | 63.2% | 58.9% | 42 | 254 | 1661 | 10 | 4 | 52 / 26 |
| BoT-SORT (no ReID) | OVERALL | 30.2% | 41.7% | 129 | 1545 | 48228 | 27 | 88 | 235 / 171 |

Tracker-only time per frame (mean, ms; excludes the detector):

| Tracker | MOT17-02 | MOT17-04 | MOT17-09 | overall |
|---|---:|---:|---:|---:|
| ByteTrack | 0.99 | 1.45 | 0.93 | 1.20 |
| BoT-SORT | 9.16 | 10.64 | 9.65 | 9.99 |
