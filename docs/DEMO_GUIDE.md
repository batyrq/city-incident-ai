# Demo Guide

This is the shortest possible path to a working demo.

## 1. Install

```bash
bash scripts/bootstrap_env.sh
```

or on Windows:

```powershell
.\scripts\bootstrap_env.ps1
```

## 2. Run a pretrained demo

```bash
py -3 infer_video.py --config configs/infer_pretrained_yolo11s.yaml --source demo.mp4
```

## 3. Run open-vocabulary demo

```bash
py -3 infer_video.py --config configs/infer_open_vocab.yaml --source demo.mp4
```

## 4. Run incident-enhanced demo

```bash
py -3 infer_video.py --config configs/infer_incidents_pretrained.yaml --source demo.mp4
```

## 5. What to look at

- `reports/<run_name>_summary.json`
- `reports/<run_name>_events.json`
- `reports/<run_name>_timeline.json`
- `reports/<run_name>_artifacts.json`
- `alerts/<run_name>/<date>/annotated.mp4`

## 6. If something is missing

- Missing source: the run writes a skipped summary.
- Missing open-vocabulary model: the run may fall back to pretrained YOLO11s.
- Missing OpenCV or Ultralytics: the summary explains the skip reason.
