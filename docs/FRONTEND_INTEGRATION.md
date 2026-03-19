# Frontend Integration

This document is a draft contract for the future frontend or lightweight backend layer.

## What the Frontend Receives

Each inference run produces:

- annotated video,
- summary JSON,
- event JSON,
- timeline JSON,
- artifacts JSON,
- optional per-event thumbnails and clips.

Recommended primary file:

- `reports/<run_name>_summary.json`

Supporting files:

- `reports/<run_name>_events.json`
- `reports/<run_name>_timeline.json`
- `reports/<run_name>_artifacts.json`

## UI Fields That Matter

The UI will usually display:

- `run_name`
- `mode`
- `model_type`
- `source_path`
- `summary_status`
- `frames_processed`
- `detections_total`
- `tracked_objects_total`
- `classes_seen`
- `events_detected`
- `output_video_path`
- `alerts_dir`
- `skipped_reason`

For event cards:

- `event_id`
- `event_type`
- `start_frame`
- `end_frame`
- `start_time_sec`
- `end_time_sec`
- `confidence_summary`
- `related_track_ids`
- `thumbnail_path`
- `clip_path`

## Recommended UI Flow

1. Choose a video or RTSP source.
2. Choose mode:
   - `pretrained_yolo11s`
   - `open_vocab`
   - `incident_enhanced`
3. Optionally edit prompts for open-vocabulary mode.
4. Start inference.
5. Show annotated video.
6. Render event cards and a timeline.
7. Link to the raw JSON artifacts for debugging or downloads.

## Draft API Contract

The project does not need a full backend framework right now, but the following contract is a good target.

### `POST /api/infer/video`

Request:

```json
{
  "source_path": "demo.mp4",
  "mode": "pretrained_yolo11s",
  "model_type": "yolo11s",
  "run_name": "demo_run",
  "prompts": ["fire", "smoke"],
  "report_path": "reports/demo_run_summary.json",
  "alerts_dir": "alerts"
}
```

Response:

```json
{
  "status": "completed",
  "run_name": "demo_run",
  "summary_path": "reports/demo_run_summary.json",
  "events_path": "reports/demo_run_events.json",
  "timeline_path": "reports/demo_run_timeline.json",
  "artifacts_path": "reports/demo_run_artifacts.json",
  "output_video_path": "alerts/demo_run/2026-03-19/annotated.mp4"
}
```

Error shape:

```json
{
  "status": "skipped",
  "error": {
    "code": "source_missing",
    "message": "Source missing"
  }
}
```

### `GET /api/runs/{run_name}`

Response should return the summary JSON.

### `GET /api/runs/{run_name}/events`

Response should return the event JSON list.

### `GET /api/runs/{run_name}/artifacts`

Response should return the artifacts JSON object.

### `GET /api/health`

Response:

```json
{
  "status": "ok"
}
```

## Example Event Payload

```json
{
  "event_id": "fire_12ab34cd56",
  "event_type": "fire",
  "start_frame": 18,
  "end_frame": 18,
  "start_time_sec": 0.72,
  "end_time_sec": 0.72,
  "confidence_summary": {
    "count": 2,
    "max": 0.84,
    "mean": 0.79
  },
  "related_track_ids": [],
  "thumbnail_path": "alerts/demo_run/2026-03-19/fire_12ab34cd56/thumbnail.jpg",
  "clip_path": "alerts/demo_run/2026-03-19/fire_12ab34cd56/clip_annotated.mp4"
}
```

## Example Summary Payload

```json
{
  "status": "completed",
  "summary_status": "completed",
  "mode": "pretrained_yolo11s",
  "model_type": "yolo11s",
  "run_name": "demo_run",
  "source_path": "demo.mp4",
  "frames_processed": 80,
  "detections_total": 320,
  "tracked_objects_total": 12,
  "classes_seen": ["person", "car"],
  "events_detected": {
    "crowding": 1
  },
  "output_video_path": "alerts/demo_run/2026-03-19/annotated.mp4",
  "skipped_reason": null
}
```

## Notes for Frontend Developers

- Treat the summary JSON as the canonical run record.
- Use `events.json` for cards and detail drawers.
- Use `timeline.json` for charts or scrubbers.
- Use `artifacts.json` to resolve file paths without guessing.
- If `summary_status` is `skipped`, display the reason instead of an error state.
