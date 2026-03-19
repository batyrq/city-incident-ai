# Project Summary

## 1. Project Goal

`city-incident-ai` is a local pretrained-first urban safety demo for video and optional RTSP inference. The goal is to produce a practical hackathon MVP that can annotate common urban scenes, attach event heuristics, and emit machine-readable outputs for a future frontend.

## 2. Why Pretrained-First

The project originally had a fine-tune-first scaffold. For the MVP we pivoted to pretrained-first because that is the fastest path to a working demo without depending on local datasets or long training cycles.

## 3. What Is Already Implemented

- pretrained YOLO11s video inference,
- optional open-vocabulary detection via YOLO-World / YOLOE,
- Ultralytics tracking,
- rule-based event detection,
- standard output artifacts for demo and frontend use,
- backward-compatible training code that is no longer the main path.

## 4. Current Modes

- `pretrained_yolo11s`: stable baseline for people and vehicles,
- `open_vocab`: experimental mode for prompts like `fire`, `smoke`, `flood`, and `water pooling`.
- `incident_enhanced`: experimental extension for pothole, road damage, fire, smoke, and accident candidates.

## 5. What Works Today

- local video inference,
- annotated MP4 generation,
- event summaries and timelines,
- skipped reports when source or dependencies are missing,
- optional RTSP path with lightweight support.
- incident-enhanced candidate outputs with prompt-based and heuristic-assisted labeling.

## 6. What Is Experimental

- open-vocabulary detection quality,
- incident candidate detection for potholes, road damage, fire, smoke, and accident-like scenes,
- RTSP availability and stream stability,
- advanced event logic beyond the basic MVP heuristics.

## 7. What Was Intentionally Postponed

- fine-tuning workflows,
- large benchmark matrices,
- segmentation training,
- anomaly training,
- SAM2 integration,
- a full backend framework.

## 8. Repository Structure

- `infer_video.py`, `infer_rtsp.py`: main demo entry points,
- `src/infer/runtime.py`: inference runtime and event logic,
- `configs/`: demo configs and legacy training configs,
- `alerts/`: per-run annotated outputs,
- `reports/`: JSON summaries and handoff artifacts,
- `docs/`: human-readable project and frontend notes,
- `src/train/`, `scripts/`, `datasets/`: legacy and optional paths kept for completeness.

## 9. Main Scripts

- `infer_video.py`: run a pretrained-first demo on a local video or folder of videos,
- `infer_rtsp.py`: run a lightweight RTSP stream demo,
- `scripts/setup_data.py`: legacy dataset bootstrap path,
- `train.py`: legacy training path, not required for MVP demo.

## 10. JSON Artifacts

The demo produces these machine-readable files:

- `reports/<run_name>_summary.json`,
- `reports/<run_name>_events.json`,
- `reports/<run_name>_timeline.json`,
- `reports/<run_name>_artifacts.json`.

These are designed to be easy to read from a future backend or frontend.

## 11. What Frontend Will Need

The frontend primarily needs:

- the summary JSON for run metadata,
- the events JSON for event cards,
- the timeline JSON for charts,
- the artifacts JSON for file paths,
- the annotated MP4 for playback.

See [`docs/FRONTEND_INTEGRATION.md`](docs/FRONTEND_INTEGRATION.md) for the recommended API contract.

## 12. Next Steps After Hackathon

- wrap the inference flow in a small API service,
- add storage for run history,
- improve prompt handling for open-vocabulary mode,
- add more robust event post-processing,
- optionally connect real datasets back into training if a later project needs it.
