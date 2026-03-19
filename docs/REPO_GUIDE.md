# Repo Guide

## Main MVP Path

- `infer_video.py`
- `infer_rtsp.py`
- `src/infer/runtime.py`
- `configs/infer_pretrained_yolo11s.yaml`
- `configs/infer_open_vocab.yaml`
- `configs/infer_incidents_pretrained.yaml`
- `configs/inference_rtsp.yaml`

These files drive the pretrained-first demo.

## Output Locations

- `alerts/` contains annotated videos and per-event clips.
- `reports/` contains summary JSON, event JSON, timeline JSON, and artifacts JSON.

## Configs That Matter

- `configs/infer_pretrained_yolo11s.yaml`
- `configs/infer_open_vocab.yaml`
- `configs/inference_rtsp.yaml`

Legacy training configs still exist, but they are not the primary demo path.

## What To Open First

1. `README.md`
2. `docs/DEMO_GUIDE.md`
3. `docs/PROJECT_SUMMARY.md`
4. `docs/FRONTEND_INTEGRATION.md`

## What Can Usually Be Ignored For the Demo

- dataset bootstrap scripts,
- detection training configs,
- anomaly and segmentation branches,
- benchmark matrix scripts.

## Legacy / Optional Areas

- `src/train/`
- `scripts/setup_data.py`
- `scripts/verify_data.py`
- `train.py`
- `benchmark.py`

These are still present for completeness, but they are not required for the pretrained-first demo flow.
