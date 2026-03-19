from __future__ import annotations

import csv
import json
import math
import time
import timeit
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.utils.env import has_package
from src.utils.io import ensure_dir, write_json

try:
    import cv2
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    cv2 = None


STANDARD_URBAN_CLASSES = {"person", "car", "bus", "truck", "motorcycle"}
FIRE_PROMPT_ALIASES = {"fire", "smoke"}
FLOOD_PROMPT_ALIASES = {"flood", "water pooling", "water_pooling", "waterpooling", "flood_water"}
VEHICLE_CLASSES = {"car", "bus", "truck", "motorcycle"}
POTHOLE_PROMPT_ALIASES = {"pothole", "potholes"}
ROAD_DAMAGE_PROMPT_ALIASES = {"road damage", "damaged road", "damaged asphalt", "road crack", "broken road surface"}
INCIDENT_FIRE_PROMPT_ALIASES = {"fire"}
INCIDENT_SMOKE_PROMPT_ALIASES = {"smoke"}
ACCIDENT_PROMPT_ALIASES = {"accident", "car crash", "collision", "wrecked vehicle", "overturned car"}


@dataclass
class DemoSpec:
    mode: str
    model_type: str
    weights: str
    prompts: list[str] = field(default_factory=list)
    standard_weights: str = "yolo11s.onnx"
    incident_weights: str | None = None
    incident_model_type: str | None = None
    incident_prompts: list[str] = field(default_factory=list)
    thresholds: dict[str, Any] = field(default_factory=dict)
    report_path: str = "reports/pretrained_infer_summary.json"
    run_name: str = "pretrained_demo"
    alerts_root: str = "alerts"
    tracker_config: str = "bytetrack.yaml"
    fallback_reason: str | None = None
    open_vocab_enabled: bool = False
    incident_enabled: bool = False
    incident_fallback_reason: str | None = None


@dataclass
class DemoState:
    prompt_histories: dict[str, deque] = field(default_factory=lambda: defaultdict(lambda: deque(maxlen=30)))
    cooldown_until: dict[str, float] = field(default_factory=dict)
    track_state: dict[int, dict[str, Any]] = field(default_factory=dict)
    event_counts: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    track_alerts: set[tuple[str, int]] = field(default_factory=set)


def run_inference_stream(source: str, camera_id: str, config: dict[str, Any], model_path: str | None) -> dict[str, Any]:
    """Backward-compatible RTSP entry point."""
    stream_cfg = dict(config)
    stream_cfg.setdefault("run_name", camera_id)
    stream_cfg.setdefault("source_kind", "rtsp")
    stream_cfg.setdefault("report_path", "reports/infer_rtsp_summary.json")
    return _run_video_like_stream(source, stream_cfg, model_path=model_path, camera_id=camera_id)


def run_pretrained_video_demo(source: str, config: dict[str, Any], model_path: str | None = None) -> dict[str, Any]:
    return _run_video_like_stream(source, config, model_path=model_path, camera_id=config.get("camera_id"))


def run_demo_video(source: str, config: dict[str, Any], model_path: str | None = None) -> dict[str, Any]:
    return run_pretrained_video_demo(source, config, model_path=model_path)


def _run_video_like_stream(
    source: str,
    config: dict[str, Any],
    model_path: str | None = None,
    camera_id: str | None = None,
) -> dict[str, Any]:
    started_at = _now_iso()
    report_path = Path(config.get("report_path", "reports/pretrained_infer_summary.json"))
    reports_root = report_path.parent
    alerts_root = Path(config.get("alerts_root", "alerts"))
    run_name = config.get("run_name") or (camera_id or Path(source).stem)
    if cv2 is None:
        result = _build_summary(
            status="skipped",
            reason="OpenCV is not installed",
            spec=_spec_from_config(config, model_path=model_path),
            source=source,
            run_name=run_name,
            started_at=started_at,
            finished_at=_now_iso(),
            report_path=report_path,
            output_dir=None,
            alerts_dir=None,
            output_video_path=None,
            predictions_jsonl=None,
            predictions_csv=None,
            timeline_csv=None,
            timeline_json=None,
            events_json=None,
            artifacts_json=None,
            frames_processed=0,
            detections_total=0,
            tracked_objects_total=0,
            classes_seen=[],
            event_counts={},
            mean_latency_ms=None,
            fps_estimate=None,
        )
        write_json(result, report_path)
        return result
    source_path = Path(source)
    if not source_path.exists() and not source.startswith("rtsp://") and not source.startswith("rtmp://"):
        result = _build_summary(
            status="skipped",
            reason="Source missing",
            spec=_spec_from_config(config, model_path=model_path),
            source=source,
            run_name=run_name,
            started_at=started_at,
            finished_at=_now_iso(),
            report_path=report_path,
            output_dir=None,
            alerts_dir=None,
            output_video_path=None,
            predictions_jsonl=None,
            predictions_csv=None,
            timeline_csv=None,
            timeline_json=None,
            events_json=None,
            artifacts_json=None,
            frames_processed=0,
            detections_total=0,
            tracked_objects_total=0,
            classes_seen=[],
            event_counts={},
            mean_latency_ms=None,
            fps_estimate=None,
        )
        write_json(result, report_path)
        return result

    detector, spec = _load_detector(config, model_path=model_path)
    incident_detector = None
    if spec.incident_enabled:
        incident_detector, incident_reason = _load_incident_detector(spec, config)
        spec.incident_fallback_reason = incident_reason
    if detector is None:
        result = _build_summary(
            status="skipped",
            reason=spec.fallback_reason or "Detector unavailable",
            spec=spec,
            source=source,
            run_name=run_name,
            started_at=started_at,
            finished_at=_now_iso(),
            report_path=report_path,
            output_dir=None,
            alerts_dir=None,
            output_video_path=None,
            predictions_jsonl=None,
            predictions_csv=None,
            timeline_csv=None,
            timeline_json=None,
            events_json=None,
            artifacts_json=None,
            frames_processed=0,
            detections_total=0,
            tracked_objects_total=0,
            classes_seen=[],
            event_counts={},
            mean_latency_ms=None,
            fps_estimate=None,
        )
        write_json(result, report_path)
        return result

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        result = _build_summary(
            status="skipped",
            reason="Failed to open source",
            spec=spec,
            source=source,
            run_name=run_name,
            started_at=started_at,
            finished_at=_now_iso(),
            report_path=report_path,
            output_dir=None,
            alerts_dir=None,
            output_video_path=None,
            predictions_jsonl=None,
            predictions_csv=None,
            timeline_csv=None,
            timeline_json=None,
            events_json=None,
            artifacts_json=None,
            frames_processed=0,
            detections_total=0,
            tracked_objects_total=0,
            classes_seen=[],
            event_counts={},
            mean_latency_ms=None,
            fps_estimate=None,
        )
        write_json(result, report_path)
        return result

    fps = float(cap.get(cv2.CAP_PROP_FPS) or config.get("runtime", {}).get("approx_fps", 25.0))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if width <= 0 or height <= 0:
        ok, first_frame = cap.read()
        if not ok:
            result = _skipped_result(source, run_name, "Empty source", report_path)
            write_json(result, report_path)
            cap.release()
            return result
        height, width = first_frame.shape[:2]
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    date_folder = time.strftime("%Y-%m-%d")
    run_root = ensure_dir(alerts_root / run_name / date_folder)
    annotated_path = run_root / "annotated.mp4"
    writer = cv2.VideoWriter(str(annotated_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    predictions_jsonl = run_root / "predictions.jsonl"
    predictions_csv = run_root / "predictions.csv"
    timeline_csv = run_root / "timeline.csv"
    timeline_json = run_root / "timeline.json"
    events_json = run_root / "events.json"
    artifacts_json = run_root / "artifacts.json"
    summary_events_path = reports_root / f"{run_name}_events.json"
    summary_timeline_path = reports_root / f"{run_name}_timeline.json"
    summary_artifacts_path = reports_root / f"{run_name}_artifacts.json"
    thumbnail_dir = ensure_dir(run_root / "thumbnails")
    state = DemoState()
    prebuffer_frames = max(1, int(round(config.get("runtime", {}).get("clip_seconds_before", 3) * fps)))
    prebuffer: deque[Any] = deque(maxlen=prebuffer_frames)
    frame_idx = 0
    timeline_rows: list[dict[str, Any]] = []
    event_records: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    latencies_ms: list[float] = []
    classes_seen: set[str] = set()
    track_ids_seen: set[int] = set()
    detections_total = 0
    max_frames = int(config.get("runtime", {}).get("max_frames", 0) or 0)

    with predictions_jsonl.open("w", encoding="utf-8") as json_handle, predictions_csv.open("w", encoding="utf-8", newline="") as csv_handle, timeline_csv.open("w", encoding="utf-8", newline="") as timeline_handle:
        pred_writer = csv.DictWriter(csv_handle, fieldnames=["frame_idx", "source", "track_id", "class_name", "confidence", "x1", "y1", "x2", "y2"])
        pred_writer.writeheader()
        timeline_writer = csv.DictWriter(
            timeline_handle,
            fieldnames=[
                "frame_idx",
                "timestamp_sec",
                "events_fired",
                "detections_count",
                "tracked_objects_count",
                "vehicle_count",
                "person_count",
                "incident_detections_count",
                "open_vocab_mode",
            ],
        )
        timeline_writer.writeheader()

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if max_frames and frame_idx >= max_frames:
                break
            frame_idx += 1
            prebuffer.append(frame.copy())
            started = timeit.default_timer()
            result = _infer_frame(detector, frame, spec, config, track=True)
            latencies_ms.append((timeit.default_timer() - started) * 1000.0)
            detections = _extract_detections(result, frame_idx=frame_idx, fps=fps)
            incident_result = None
            incident_detections: list[dict[str, Any]] = []
            if incident_detector is not None:
                incident_result = _infer_frame(incident_detector, frame, spec, config, track=False)
                incident_detections = _extract_detections(incident_result, frame_idx=frame_idx, fps=fps)
            detections_total += len(detections) + len(incident_detections)
            for det in detections:
                classes_seen.add(_normalize(det["class_name"]))
                if det.get("track_id") is not None:
                    track_ids_seen.add(int(det["track_id"]))
            for det in incident_detections:
                classes_seen.add(_normalize(det["class_name"]))

            annotated = result.plot() if hasattr(result, "plot") else frame.copy()
            if incident_detections:
                annotated = _draw_incident_detections(annotated, incident_detections, spec, config)
            writer.write(annotated)

            combined_prediction_rows = [
                {"frame_idx": frame_idx, "source": "standard_tracking", **row} for row in detections
            ]
            combined_prediction_rows.extend(
                {"frame_idx": frame_idx, "source": "open_vocab", **row} for row in incident_detections
            )
            json_handle.write(
                json.dumps(
                    {"frame_idx": frame_idx, "detections": combined_prediction_rows},
                    ensure_ascii=False,
                )
                + "\n"
            )
            prediction_rows.extend(combined_prediction_rows)
            for row in combined_prediction_rows:
                pred_writer.writerow({"frame_idx": frame_idx, **row})

            events, counters, frame_event_records = _update_rules(state, detections, frame, fps, frame_idx, config, run_root, prebuffer)
            event_records.extend(frame_event_records)
            if spec.incident_enabled:
                incident_events, incident_event_records = _update_incident_rules(
                    state,
                    detections,
                    incident_detections,
                    frame,
                    fps,
                    frame_idx,
                    config,
                    run_root,
                    prebuffer,
                    spec,
                    events,
                )
                events.extend(incident_events)
                event_records.extend(incident_event_records)
            timeline_writer.writerow(
                {
                    "frame_idx": frame_idx,
                    "timestamp_sec": round(frame_idx / max(fps, 1e-6), 3),
                    "events_fired": ",".join(events),
                    "detections_count": len(detections) + len(incident_detections),
                    "tracked_objects_count": len({det["track_id"] for det in detections if det.get("track_id") is not None}),
                    "vehicle_count": counters["vehicle_count"],
                    "person_count": counters["person_count"],
                    "incident_detections_count": len(incident_detections),
                    "open_vocab_mode": spec.open_vocab_enabled,
                }
            )
            timeline_rows.append(
                {
                    "frame_idx": frame_idx,
                    "timestamp_sec": round(frame_idx / max(fps, 1e-6), 3),
                    "events_fired": events,
                    "detections_count": len(detections) + len(incident_detections),
                    "tracked_objects_count": len({det["track_id"] for det in detections if det.get("track_id") is not None}),
                    "vehicle_count": counters["vehicle_count"],
                    "person_count": counters["person_count"],
                    "incident_detections_count": len(incident_detections),
                }
            )
            if detections:
                top_det = max(detections, key=lambda item: item["confidence"])
                cv2.imwrite(str(thumbnail_dir / f"frame_{frame_idx:06d}_{top_det['class_name']}.jpg"), frame)

    cap.release()
    writer.release()
    write_json(timeline_rows, timeline_json)
    write_json(event_records, events_json)
    write_json(timeline_rows, summary_timeline_path)
    write_json(event_records, summary_events_path)
    incident_track_ids = sorted(
        {
            int(track_id)
            for record in event_records
            if record.get("source") in {"open_vocab", "fused"}
            for track_id in record.get("related_track_ids", [])
        }
    )
    summary = _build_summary(
        status="completed",
        reason=None,
        spec=spec,
        source=source,
        run_name=run_name,
        started_at=started_at,
        finished_at=_now_iso(),
        report_path=report_path,
        output_dir=run_root,
        alerts_dir=run_root,
        output_video_path=annotated_path,
        predictions_jsonl=predictions_jsonl,
        predictions_csv=predictions_csv,
        timeline_csv=timeline_csv,
        timeline_json=summary_timeline_path,
        events_json=summary_events_path,
        artifacts_json=summary_artifacts_path,
        frames_processed=frame_idx,
        detections_total=detections_total,
        tracked_objects_total=len(track_ids_seen),
        classes_seen=sorted(classes_seen),
        event_counts=dict(state.event_counts),
        mean_latency_ms=round(sum(latencies_ms) / len(latencies_ms), 3) if latencies_ms else None,
        fps_estimate=round(1000.0 / (sum(latencies_ms) / len(latencies_ms)), 3) if latencies_ms else None,
        prompts_used=spec.incident_prompts if spec.incident_enabled else spec.prompts,
        incident_model_type=spec.incident_model_type,
        incident_fallback_reason=spec.incident_fallback_reason,
        events_detected_count=sum(state.event_counts.values()),
        events_detected_types=sorted(state.event_counts.keys()),
        incident_tracks_involved=incident_track_ids,
        stopped_vehicle_candidates=state.event_counts.get("stopped_vehicle", 0),
        collision_candidates=state.event_counts.get("collision_candidate", 0),
    )
    artifacts_payload = {
        "run_name": run_name,
        "mode": spec.mode,
        "model_type": spec.model_type,
        "source_path": source,
        "output_dir": str(run_root),
        "alerts_dir": str(run_root),
        "summary_path": str(report_path),
        "events_path": str(summary_events_path),
        "timeline_path": str(summary_timeline_path),
        "artifacts_path": str(summary_artifacts_path),
        "annotated_video_path": str(annotated_path),
        "predictions_jsonl_path": str(predictions_jsonl),
        "predictions_csv_path": str(predictions_csv),
        "timeline_csv_path": str(timeline_csv),
        "events_detected": dict(state.event_counts),
        "classes_seen": sorted(classes_seen),
        "output_video_path": str(annotated_path),
        "summary_status": "completed",
    }
    write_json(artifacts_payload, artifacts_json)
    write_json(artifacts_payload, summary_artifacts_path)
    write_json(summary, report_path)
    return summary


def _load_detector(config: dict[str, Any], model_path: str | None = None) -> tuple[Any | None, DemoSpec]:
    spec = _spec_from_config(config, model_path=model_path)
    mode = spec.mode
    model_type = spec.model_type
    weights = spec.weights
    prompts = spec.prompts

    if not has_package("ultralytics"):
        spec.fallback_reason = "Ultralytics is not installed"
        return None, spec

    try:
        if model_type in {"yolo11s", "yolo11n", "yolo11m", "yolo"}:
            from ultralytics import YOLO

            return YOLO(weights, task="detect"), spec

        if model_type in {"yolo-world", "yoloworld", "yolov8s-world", "yolov8s-world.pt"}:
            from ultralytics import YOLOWorld

            model = YOLOWorld(weights)
            if prompts:
                model.set_classes(prompts)
            return model, spec

        if model_type in {"yoloe", "yoloe-11s", "yoloe-11s.pt"}:
            from ultralytics import YOLOE

            model = YOLOE(weights, task="detect")
            if prompts:
                model.set_vocab(prompts, prompts)
            return model, spec

        spec.fallback_reason = f"Unsupported model_type '{model_type}'"
        return None, spec
    except Exception as exc:  # pragma: no cover - external model init can fail for env/network reasons
        if mode == "open_vocab" and model_type != "yolo11s":
            try:
                from ultralytics import YOLO

                fallback = YOLO(_default_weights("yolo11s"), task="detect")
                spec.fallback_reason = f"Open-vocabulary model unavailable, falling back to pretrained YOLO11s: {exc}"
                spec.model_type = "yolo11s"
                spec.mode = "pretrained_yolo11s"
                spec.open_vocab_enabled = False
                return fallback, spec
            except Exception as fallback_exc:
                spec.fallback_reason = f"Open-vocabulary and fallback standard models unavailable: {exc}; fallback error: {fallback_exc}"
                return None, spec
        spec.fallback_reason = str(exc)
        return None, spec


def _load_incident_detector(spec: DemoSpec, config: dict[str, Any]) -> tuple[Any | None, str | None]:
    if not spec.incident_enabled:
        return None, None
    if not has_package("ultralytics"):
        return None, "Ultralytics is not installed"

    infer_cfg = config.get("infer", config)
    model_type = (spec.incident_model_type or infer_cfg.get("open_vocab_model_type") or "yolo-world").lower()
    weights = spec.incident_weights or str(infer_cfg.get("open_vocab_model") or infer_cfg.get("incident_model") or _default_weights(model_type))
    prompts = spec.incident_prompts or spec.prompts
    try:
        if model_type in {"yolo-world", "yoloworld", "yolov8s-world", "yolov8s-world.pt"}:
            from ultralytics import YOLOWorld

            model = YOLOWorld(weights)
            if prompts:
                model.set_classes(prompts)
            return model, None
        if model_type in {"yoloe", "yoloe-11s", "yoloe-11s.pt"}:
            from ultralytics import YOLOE

            model = YOLOE(weights, task="detect")
            if prompts:
                model.set_vocab(prompts, prompts)
            return model, None
        return None, f"Unsupported incident model_type '{model_type}'"
    except Exception as exc:  # pragma: no cover - external model init can fail for env/network reasons
        return None, str(exc)


def _infer_frame(detector: Any, frame: Any, spec: DemoSpec, config: dict[str, Any], track: bool = True) -> Any:
    runtime_cfg = config.get("runtime", {})
    tracking_cfg = config.get("tracking", {})
    conf = runtime_cfg.get("score_threshold", 0.35)
    iou = runtime_cfg.get("iou_threshold", 0.5)
    imgsz = runtime_cfg.get("imgsz", 640)
    tracker_cfg = spec.tracker_config or tracking_cfg.get("tracker", "bytetrack.yaml")
    if not track:
        return detector.predict(source=frame, verbose=False, conf=conf, iou=iou, imgsz=imgsz)[0]
    try:
        return detector.track(
            source=frame,
            persist=True,
            verbose=False,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            tracker=tracker_cfg,
        )[0]
    except Exception:
        return detector.predict(source=frame, verbose=False, conf=conf, iou=iou, imgsz=imgsz)[0]


def _extract_detections(result: Any, frame_idx: int, fps: float) -> list[dict[str, Any]]:
    detections: list[dict[str, Any]] = []
    names = getattr(result, "names", {}) or {}
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return detections

    xyxy = boxes.xyxy.cpu().tolist() if hasattr(boxes.xyxy, "cpu") else boxes.xyxy.tolist()
    confs = boxes.conf.cpu().tolist() if hasattr(boxes.conf, "cpu") else boxes.conf.tolist()
    clss = boxes.cls.cpu().tolist() if hasattr(boxes.cls, "cpu") else boxes.cls.tolist()
    track_ids = None
    if hasattr(boxes, "id") and boxes.id is not None:
        track_ids = boxes.id.cpu().tolist() if hasattr(boxes.id, "cpu") else boxes.id.tolist()

    for idx, (coords, conf, cls_id) in enumerate(zip(xyxy, confs, clss)):
        class_name = str(names.get(int(cls_id), int(cls_id)))
        track_id = int(track_ids[idx]) if track_ids and idx < len(track_ids) and track_ids[idx] is not None else None
        detections.append(
            {
                "track_id": track_id,
                "class_name": class_name,
                "confidence": round(float(conf), 5),
                "x1": round(float(coords[0]), 2),
                "y1": round(float(coords[1]), 2),
                "x2": round(float(coords[2]), 2),
                "y2": round(float(coords[3]), 2),
            }
        )
    return detections


def _update_rules(
    state: DemoState,
    detections: list[dict[str, Any]],
    frame: Any,
    fps: float,
    frame_idx: int,
    config: dict[str, Any],
    run_root: Path,
    prebuffer: deque[Any],
) -> tuple[list[str], dict[str, int]]:
    runtime_cfg = config.get("runtime", {})
    tracking_cfg = config.get("tracking", {})
    rules_cfg = config.get("rules", {})
    now_ts = frame_idx / max(fps, 1e-6)
    events: list[str] = []
    event_records: list[dict[str, Any]] = []
    person_count = sum(1 for det in detections if _normalize(det["class_name"]) == "person")
    vehicle_count = sum(1 for det in detections if _normalize(det["class_name"]) in VEHICLE_CLASSES)

    prompt_events = _update_prompt_events(state, detections, rules_cfg, now_ts)
    events.extend(prompt_events)
    if prompt_events:
        for event_name in prompt_events:
            state.event_counts[event_name] += 1
            artifact_info = _persist_event_clip(run_root, event_name, frame_idx, detections, frame, prebuffer, fps)
            event_records.append(_build_event_record(event_name, frame_idx, now_ts, detections, artifact_info, fps))

    stopped = _update_stopped_vehicle(state, detections, rules_cfg, tracking_cfg, now_ts, fps)
    if stopped:
        events.extend(stopped)
        for event_name in stopped:
            state.event_counts[event_name] += 1
            artifact_info = _persist_event_clip(run_root, event_name, frame_idx, detections, frame, prebuffer, fps)
            event_records.append(_build_event_record(event_name, frame_idx, now_ts, detections, artifact_info, fps))

    crowding = _update_crowding(state, person_count, vehicle_count, rules_cfg, now_ts)
    if crowding:
        events.extend(crowding)
        for event_name in crowding:
            state.event_counts[event_name] += 1
            artifact_info = _persist_event_clip(run_root, event_name, frame_idx, detections, frame, prebuffer, fps)
            event_records.append(_build_event_record(event_name, frame_idx, now_ts, detections, artifact_info, fps))

    return events, {"person_count": person_count, "vehicle_count": vehicle_count}, event_records


def _update_prompt_events(
    state: DemoState,
    detections: list[dict[str, Any]],
    rules_cfg: dict[str, Any],
    now_ts: float,
) -> list[str]:
    fired: list[str] = []
    prompt_rules = {
        "fire": FIRE_PROMPT_ALIASES,
        "smoke": FIRE_PROMPT_ALIASES,
        "flood_water": FLOOD_PROMPT_ALIASES,
    }
    for event_name, aliases in prompt_rules.items():
        rule = rules_cfg.get(event_name, {})
        window_frames = int(rule.get("confirm_window_frames", 6))
        min_hits = int(rule.get("min_confirm_frames", 3))
        cooldown = float(rule.get("cooldown_seconds", 60))
        if now_ts < state.cooldown_until.get(event_name, 0):
            continue
        seen = any(_normalize(det["class_name"]) in {_normalize(alias) for alias in aliases} for det in detections)
        history = state.prompt_histories[event_name]
        history.append(1 if seen else 0)
        while len(history) > window_frames:
            history.popleft()
        if sum(history) >= min_hits:
            fired.append(event_name)
            state.cooldown_until[event_name] = now_ts + cooldown
            history.clear()
    return fired


def _update_stopped_vehicle(
    state: DemoState,
    detections: list[dict[str, Any]],
    rules_cfg: dict[str, Any],
    tracking_cfg: dict[str, Any],
    now_ts: float,
    fps: float,
) -> list[str]:
    rule = rules_cfg.get("stopped_vehicle", {})
    motion_px = float(rule.get("motion_threshold_px", tracking_cfg.get("stopped_vehicle_motion_threshold_px", 6.0)))
    dwell_seconds = float(rule.get("dwell_seconds", tracking_cfg.get("stopped_vehicle_dwell_seconds", 15)))
    cooldown = float(rule.get("cooldown_seconds", 180))
    fired: list[str] = []
    dwell_frames = max(1, int(round(dwell_seconds * max(fps, 1.0))))
    for det in detections:
        if _normalize(det["class_name"]) not in VEHICLE_CLASSES or det.get("track_id") is None:
            continue
        track_id = int(det["track_id"])
        center = ((_as_float(det["x1"]) + _as_float(det["x2"])) / 2.0, (_as_float(det["y1"]) + _as_float(det["y2"])) / 2.0)
        track = state.track_state.setdefault(
            track_id,
            {"class_name": det["class_name"], "first_seen_frame": 0, "last_move_frame": 0, "last_center": center, "alerted": False},
        )
        if track["first_seen_frame"] == 0:
            track["first_seen_frame"] = max(1, int(round(now_ts * fps)))
        prev_center = track["last_center"]
        moved = math.hypot(center[0] - prev_center[0], center[1] - prev_center[1])
        frame_number = max(1, int(round(now_ts * fps)))
        if moved > motion_px:
            track["last_move_frame"] = frame_number
        track["last_center"] = center
        if frame_number - track["last_move_frame"] >= dwell_frames and not track["alerted"] and now_ts >= state.cooldown_until.get("stopped_vehicle", 0):
            fired.append("stopped_vehicle")
            track["alerted"] = True
            state.cooldown_until["stopped_vehicle"] = now_ts + cooldown
            state.track_alerts.add(("stopped_vehicle", track_id))
    return fired


def _update_crowding(
    state: DemoState,
    person_count: int,
    vehicle_count: int,
    rules_cfg: dict[str, Any],
    now_ts: float,
) -> list[str]:
    rule = rules_cfg.get("crowding", {})
    threshold = int(rule.get("min_count", 12))
    window_frames = int(rule.get("confirm_window_frames", 5))
    min_hits = int(rule.get("min_confirm_frames", 3))
    cooldown = float(rule.get("cooldown_seconds", 90))
    if now_ts < state.cooldown_until.get("crowding", 0):
        return []
    crowded = person_count + vehicle_count >= threshold
    history = state.prompt_histories["crowding"]
    history.append(1 if crowded else 0)
    while len(history) > window_frames:
        history.popleft()
    if sum(history) >= min_hits:
        state.cooldown_until["crowding"] = now_ts + cooldown
        history.clear()
        return ["crowding"]
    return []


def _update_incident_rules(
    state: DemoState,
    standard_detections: list[dict[str, Any]],
    incident_detections: list[dict[str, Any]],
    frame: Any,
    fps: float,
    frame_idx: int,
    config: dict[str, Any],
    run_root: Path,
    prebuffer: deque[Any],
    spec: DemoSpec,
    baseline_events: list[str],
) -> tuple[list[str], list[dict[str, Any]]]:
    thresholds = dict(spec.thresholds or config.get("thresholds", {}))
    conf = float(thresholds.get("conf", 0.25))
    temporal_k = int(thresholds.get("temporal_k", 3))
    temporal_n = int(thresholds.get("temporal_n", 5))
    min_bbox_area = float(thresholds.get("min_bbox_area", 400))
    road_roi_fraction = float(thresholds.get("road_roi_fraction", 0.55))
    overlap_iou_threshold = float(thresholds.get("overlap_iou_threshold", 0.4))
    cooldown = float(thresholds.get("cooldown_seconds", 120))
    now_ts = frame_idx / max(fps, 1e-6)
    frame_h = int(frame.shape[0]) if frame is not None else 0
    prompt_rules = {
        "pothole_candidate": POTHOLE_PROMPT_ALIASES,
        "road_damage_candidate": ROAD_DAMAGE_PROMPT_ALIASES,
        "fire_candidate": INCIDENT_FIRE_PROMPT_ALIASES,
        "smoke_candidate": INCIDENT_SMOKE_PROMPT_ALIASES,
        "accident_prompt_candidate": ACCIDENT_PROMPT_ALIASES,
    }
    prompt_hits: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for det in incident_detections:
        if _as_float(det.get("confidence")) < conf:
            continue
        if _bbox_area(det) < min_bbox_area:
            continue
        class_name = _normalize(det["class_name"])
        event_type = None
        for candidate_type, aliases in prompt_rules.items():
            if class_name in {_normalize(alias) for alias in aliases}:
                event_type = candidate_type
                break
        if event_type is None:
            continue
        if event_type in {"pothole_candidate", "road_damage_candidate"} and not _is_road_region(det, frame_h, road_roi_fraction):
            continue
        prompt_hits[event_type].append(det)

    events: list[str] = []
    event_records: list[dict[str, Any]] = []
    confirmed_prompt_events: dict[str, list[dict[str, Any]]] = {}
    for event_type, detections in prompt_hits.items():
        history = state.prompt_histories[event_type]
        history.append(1 if detections else 0)
        while len(history) > temporal_n:
            history.popleft()
        if sum(history) >= temporal_k and now_ts >= state.cooldown_until.get(event_type, 0):
            confirmed_prompt_events[event_type] = detections
            state.cooldown_until[event_type] = now_ts + cooldown
            history.clear()

    accident_prompt_detections = confirmed_prompt_events.pop("accident_prompt_candidate", None)

    if "fire_candidate" in confirmed_prompt_events and "smoke_candidate" in confirmed_prompt_events:
        merged_detections = confirmed_prompt_events["fire_candidate"] + confirmed_prompt_events["smoke_candidate"]
        event_type = "fire_smoke_candidate"
        artifact_info = _persist_event_clip(run_root, event_type, frame_idx, merged_detections, frame, prebuffer, fps, source="open_vocab")
        events.append(event_type)
        event_records.append(_build_event_record(event_type, frame_idx, now_ts, merged_detections, artifact_info, fps, source="open_vocab"))
        state.event_counts[event_type] += 1
        confirmed_prompt_events.pop("fire_candidate", None)
        confirmed_prompt_events.pop("smoke_candidate", None)

    for event_type, detections in confirmed_prompt_events.items():
        artifact_info = _persist_event_clip(run_root, event_type, frame_idx, detections, frame, prebuffer, fps, source="open_vocab")
        events.append(event_type)
        event_records.append(_build_event_record(event_type, frame_idx, now_ts, detections, artifact_info, fps, source="open_vocab"))
        state.event_counts[event_type] += 1

    collision_events, collision_records = _update_collision_candidates(
        state=state,
        detections=standard_detections,
        frame=frame,
        fps=fps,
        frame_idx=frame_idx,
        run_root=run_root,
        prebuffer=prebuffer,
        overlap_iou_threshold=overlap_iou_threshold,
        cooldown=cooldown,
    )
    events.extend(collision_events)
    event_records.extend(collision_records)

    prompt_accident = accident_prompt_detections is not None
    collision_signal = "collision_candidate" in events
    stopped_signal = "stopped_vehicle" in baseline_events
    if prompt_accident and (collision_signal or stopped_signal):
        accident_detections = list(accident_prompt_detections or []) + list(standard_detections)
        event_type = "accident_candidate"
        artifact_info = _persist_event_clip(run_root, event_type, frame_idx, accident_detections, frame, prebuffer, fps, source="fused")
        events.append(event_type)
        event_records.append(_build_event_record(event_type, frame_idx, now_ts, accident_detections, artifact_info, fps, source="fused"))
        state.event_counts[event_type] += 1
    elif prompt_accident:
        detections = accident_prompt_detections or incident_detections
        event_type = "accident_prompt_candidate"
        artifact_info = _persist_event_clip(run_root, event_type, frame_idx, detections, frame, prebuffer, fps, source="open_vocab")
        events.append(event_type)
        event_records.append(_build_event_record(event_type, frame_idx, now_ts, detections, artifact_info, fps, source="open_vocab"))
        state.event_counts[event_type] += 1
    elif collision_signal or stopped_signal:
        event_type = "traffic_incident_candidate"
        detections = standard_detections
        artifact_info = _persist_event_clip(run_root, event_type, frame_idx, detections, frame, prebuffer, fps, source="standard_tracking")
        events.append(event_type)
        event_records.append(_build_event_record(event_type, frame_idx, now_ts, detections, artifact_info, fps, source="standard_tracking"))
        state.event_counts[event_type] += 1

    return events, event_records


def _update_collision_candidates(
    state: DemoState,
    detections: list[dict[str, Any]],
    frame: Any,
    fps: float,
    frame_idx: int,
    run_root: Path,
    prebuffer: deque[Any],
    overlap_iou_threshold: float,
    cooldown: float,
) -> tuple[list[str], list[dict[str, Any]]]:
    now_ts = frame_idx / max(fps, 1e-6)
    vehicle_detections = [det for det in detections if _normalize(det["class_name"]) in VEHICLE_CLASSES and det.get("track_id") is not None]
    if len(vehicle_detections) < 2:
        return [], []
    pairs: list[dict[str, Any]] = []
    for idx, left in enumerate(vehicle_detections):
        for right in vehicle_detections[idx + 1 :]:
            iou = _bbox_iou(left, right)
            if iou >= overlap_iou_threshold:
                pairs.append({"detections": [left, right], "iou": iou})
    if not pairs or now_ts < state.cooldown_until.get("collision_candidate", 0):
        return [], []
    merged_detections = []
    for pair in pairs:
        merged_detections.extend(pair["detections"])
    state.cooldown_until["collision_candidate"] = now_ts + cooldown
    state.event_counts["collision_candidate"] += 1
    artifact_info = _persist_event_clip(run_root, "collision_candidate", frame_idx, merged_detections, frame, prebuffer, fps, source="standard_tracking")
    event_record = _build_event_record("collision_candidate", frame_idx, now_ts, merged_detections, artifact_info, fps, source="standard_tracking")
    return ["collision_candidate"], [event_record]


def _persist_event_clip(
    run_root: Path,
    event_name: str,
    frame_idx: int,
    detections: list[dict[str, Any]],
    frame: Any,
    prebuffer: deque[Any],
    fps: float,
    source: str = "open_vocab",
) -> None:
    event_id = f"{event_name}_{uuid.uuid4().hex[:10]}"
    event_dir = ensure_dir(run_root / event_id)
    event_payload = {
        "event_id": event_id,
        "event_type": event_name,
        "source": source,
        "start_frame": frame_idx,
        "end_frame": frame_idx,
        "start_time_sec": round(frame_idx / max(fps, 1e-6), 3),
        "end_time_sec": round(frame_idx / max(fps, 1e-6), 3),
        "detections": detections,
        "related_track_ids": sorted({int(det["track_id"]) for det in detections if det.get("track_id") is not None}),
        "confidence_summary": _confidence_summary(detections),
        "bbox_summary": _bbox_summary(detections),
        "timestamp": time.time(),
    }
    write_json(event_payload, event_dir / "event.json")
    thumbnail_path = event_dir / "thumbnail.jpg"
    clip_path = event_dir / "clip_annotated.mp4"
    if cv2 is not None:
        cv2.imwrite(str(thumbnail_path), frame)
        clip_frames = list(prebuffer)
        if clip_frames:
            _write_clip(clip_path, clip_frames, fps)
    return {
        "event_id": event_id,
        "event_type": event_name,
        "event_dir": str(event_dir),
        "thumbnail_path": str(thumbnail_path) if thumbnail_path.exists() else None,
        "clip_path": str(clip_path) if clip_path.exists() else None,
    }


def _write_clip(path: Path, frames: list[Any], fps: float) -> None:
    if cv2 is None or not frames:
        return
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    for frame in frames:
        writer.write(frame)
    writer.release()


def _normalize(value: str) -> str:
    return str(value).strip().lower().replace("-", " ").replace("_", " ")


def _normalize_mode(value: str) -> str:
    normalized = _normalize(value)
    if normalized in {"standard", "pretrained", "pretrained yolo11s", "pretrained yolo11", "pretrained yolo"}:
        return "pretrained_yolo11s"
    if normalized in {"pretrained yolo11s", "yolo11s"}:
        return "pretrained_yolo11s"
    if normalized in {"open vocab", "open vocabulary", "openvocab", "open_vocabulary", "open vocabular"}:
        return "open_vocab"
    if normalized in {"incident enhanced", "incident enhancement", "incident probe", "incident"}:
        return "incident_enhanced"
    return normalized.replace(" ", "_")


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _normalize_prompts(prompts: Any) -> list[str]:
    if isinstance(prompts, str):
        return [item.strip() for item in prompts.split(",") if item.strip()]
    if isinstance(prompts, list):
        return [str(item).strip() for item in prompts if str(item).strip()]
    return []


def _default_weights(model_type: str) -> str:
    if model_type in {"yolo-world", "yoloworld"}:
        return "yolov8s-world.pt"
    if model_type in {"yoloe", "yoloe-11s"}:
        return "yoloe-11s.pt"
    return "yolo11s.onnx"


def _default_report_path(mode: str) -> str:
    return "reports/open_vocab_infer_summary.json" if mode == "open_vocab" else "reports/pretrained_infer_summary.json"


def _skipped_result(source: str, run_name: str, reason: str, report_path: Path) -> dict[str, Any]:
    result = {
        "status": "skipped",
        "reason": reason,
        "source": source,
        "run_name": run_name,
        "frames_processed": 0,
        "alert_counts": {},
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    return result


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _spec_from_config(config: dict[str, Any], model_path: str | None = None) -> DemoSpec:
    infer_cfg = config.get("infer", config)
    model_cfg = infer_cfg.get("model", {})
    mode = _normalize_mode(str(infer_cfg.get("mode") or model_cfg.get("mode") or "pretrained_yolo11s"))
    thresholds = dict(infer_cfg.get("thresholds") or config.get("thresholds") or {})
    model_type = str(
        infer_cfg.get("model_type")
        or model_cfg.get("type")
        or model_cfg.get("model_type")
        or ("yolo-world" if mode == "open_vocab" else "yolo11s")
    ).lower()
    prompts = _normalize_prompts(infer_cfg.get("prompts") or model_cfg.get("prompts") or [])
    standard_weights = str(
        infer_cfg.get("standard_model")
        or infer_cfg.get("model_weights")
        or model_cfg.get("weights")
        or model_path
        or _default_weights("yolo11s")
    )
    incident_weights = infer_cfg.get("open_vocab_model") or infer_cfg.get("incident_model") or (model_path if mode == "open_vocab" else None)
    incident_model_type = str(infer_cfg.get("incident_model_type") or infer_cfg.get("open_vocab_model_type") or ("yolo-world" if mode in {"open_vocab", "incident_enhanced"} else "")).lower() or None
    incident_prompts = _normalize_prompts(infer_cfg.get("incident_prompts") or prompts)
    weights = str(model_path or infer_cfg.get("model_weights") or model_cfg.get("weights") or standard_weights)
    return DemoSpec(
        mode=mode,
        model_type=model_type,
        weights=weights,
        prompts=prompts,
        standard_weights=standard_weights,
        incident_weights=str(incident_weights) if incident_weights else None,
        incident_model_type=incident_model_type,
        incident_prompts=incident_prompts,
        thresholds=thresholds,
        report_path=str(infer_cfg.get("report_path") or _default_report_path(mode)),
        run_name=str(infer_cfg.get("run_name") or model_cfg.get("run_name") or ("open_vocab_demo" if mode == "open_vocab" else "pretrained_demo")),
        alerts_root=str(infer_cfg.get("alerts_root") or "alerts"),
        tracker_config=str(infer_cfg.get("tracker_config") or infer_cfg.get("tracking", {}).get("tracker", "bytetrack.yaml")),
        open_vocab_enabled=mode in {"open_vocab", "incident_enhanced"} or model_type in {"yolo-world", "yoloe"},
        incident_enabled=mode == "incident_enhanced",
    )


def _confidence_summary(detections: list[dict[str, Any]]) -> dict[str, Any]:
    confidences = [float(det.get("confidence", 0.0)) for det in detections]
    if not confidences:
        return {"count": 0, "max": None, "mean": None}
    return {
        "count": len(confidences),
        "max": round(max(confidences), 5),
        "mean": round(sum(confidences) / len(confidences), 5),
    }


def _bbox_summary(detections: list[dict[str, Any]]) -> dict[str, Any]:
    if not detections:
        return {"count": 0, "mean_area": None, "max_area": None, "min_area": None}
    areas = []
    for det in detections:
        width = max(0.0, _as_float(det.get("x2")) - _as_float(det.get("x1")))
        height = max(0.0, _as_float(det.get("y2")) - _as_float(det.get("y1")))
        areas.append(width * height)
    return {
        "count": len(detections),
        "mean_area": round(sum(areas) / len(areas), 2),
        "max_area": round(max(areas), 2),
        "min_area": round(min(areas), 2),
    }


def _bbox_area(det: dict[str, Any]) -> float:
    return max(0.0, _as_float(det.get("x2")) - _as_float(det.get("x1"))) * max(0.0, _as_float(det.get("y2")) - _as_float(det.get("y1")))


def _is_road_region(det: dict[str, Any], frame_h: int, road_roi_fraction: float) -> bool:
    if frame_h <= 0:
        return True
    y_center = (_as_float(det.get("y1")) + _as_float(det.get("y2"))) / 2.0
    return y_center >= frame_h * max(0.0, min(1.0, road_roi_fraction))


def _bbox_iou(left: dict[str, Any], right: dict[str, Any]) -> float:
    x1 = max(_as_float(left.get("x1")), _as_float(right.get("x1")))
    y1 = max(_as_float(left.get("y1")), _as_float(right.get("y1")))
    x2 = min(_as_float(left.get("x2")), _as_float(right.get("x2")))
    y2 = min(_as_float(left.get("y2")), _as_float(right.get("y2")))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    left_area = _bbox_area(left)
    right_area = _bbox_area(right)
    union = left_area + right_area - inter
    return inter / union if union > 0 else 0.0


def _draw_incident_detections(frame: Any, detections: list[dict[str, Any]], spec: DemoSpec, config: dict[str, Any]) -> Any:
    if cv2 is None or frame is None:
        return frame
    overlay = frame.copy()
    for det in detections:
        event_type = _incident_label_for_detection(det["class_name"])
        if event_type is None:
            continue
        color = _incident_color(event_type)
        x1 = int(round(_as_float(det["x1"])))
        y1 = int(round(_as_float(det["y1"])))
        x2 = int(round(_as_float(det["x2"])))
        y2 = int(round(_as_float(det["y2"])))
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        label = f"{event_type.upper()}? {det['confidence']:.2f}"
        _draw_label(overlay, label, (x1, max(0, y1 - 8)), color)
    return overlay


def _incident_label_for_detection(class_name: str) -> str | None:
    normalized = _normalize(class_name)
    if normalized in {_normalize(item) for item in POTHOLE_PROMPT_ALIASES}:
        return "pothole_candidate"
    if normalized in {_normalize(item) for item in ROAD_DAMAGE_PROMPT_ALIASES}:
        return "road_damage_candidate"
    if normalized in {_normalize(item) for item in INCIDENT_FIRE_PROMPT_ALIASES}:
        return "fire_candidate"
    if normalized in {_normalize(item) for item in INCIDENT_SMOKE_PROMPT_ALIASES}:
        return "smoke_candidate"
    if normalized in {_normalize(item) for item in ACCIDENT_PROMPT_ALIASES}:
        return "accident_prompt_candidate"
    return None


def _incident_color(event_type: str) -> tuple[int, int, int]:
    palette = {
        "pothole_candidate": (0, 165, 255),
        "road_damage_candidate": (0, 140, 255),
        "fire_candidate": (0, 0, 255),
        "smoke_candidate": (128, 128, 128),
        "fire_smoke_candidate": (0, 64, 255),
        "accident_prompt_candidate": (255, 0, 255),
        "accident_candidate": (0, 0, 200),
        "traffic_incident_candidate": (255, 128, 0),
        "collision_candidate": (0, 200, 200),
        "stopped_vehicle": (0, 255, 255),
    }
    return palette.get(event_type, (255, 255, 0))


def _draw_label(frame: Any, text: str, origin: tuple[int, int], color: tuple[int, int, int]) -> None:
    if cv2 is None:
        return
    x, y = origin
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 2
    (w, h), baseline = cv2.getTextSize(text, font, scale, thickness)
    top_left = (max(0, x), max(0, y - h - baseline - 2))
    bottom_right = (max(0, x + w + 4), max(0, y + baseline + 2))
    cv2.rectangle(frame, top_left, bottom_right, color, cv2.FILLED)
    cv2.putText(frame, text, (top_left[0] + 2, bottom_right[1] - baseline - 1), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def _build_event_record(
    event_type: str,
    frame_idx: int,
    timestamp_sec: float,
    detections: list[dict[str, Any]],
    artifact_info: dict[str, Any],
    fps: float,
    source: str,
) -> dict[str, Any]:
    related_track_ids = sorted({int(det["track_id"]) for det in detections if det.get("track_id") is not None})
    return {
        "event_id": artifact_info["event_id"],
        "event_type": event_type,
        "start_frame": frame_idx,
        "end_frame": frame_idx,
        "start_time_sec": round(timestamp_sec, 3),
        "end_time_sec": round(timestamp_sec, 3),
        "confidence_summary": _confidence_summary(detections),
        "bbox_summary": _bbox_summary(detections),
        "related_track_ids": related_track_ids,
        "thumbnail_path": artifact_info.get("thumbnail_path"),
        "clip_path": artifact_info.get("clip_path"),
        "source": source,
        "frame_rate": round(fps, 3),
    }


def _build_summary(
    *,
    status: str,
    reason: str | None,
    spec: DemoSpec,
    source: str,
    run_name: str,
    started_at: str,
    finished_at: str,
    report_path: Path,
    output_dir: Path | None,
    alerts_dir: Path | None,
    output_video_path: Path | None,
    predictions_jsonl: Path | None,
    predictions_csv: Path | None,
    timeline_csv: Path | None,
    timeline_json: Path | None,
    events_json: Path | None,
    artifacts_json: Path | None,
    frames_processed: int,
    detections_total: int,
    tracked_objects_total: int,
    classes_seen: list[str],
    event_counts: dict[str, int],
    mean_latency_ms: float | None,
    fps_estimate: float | None,
    prompts_used: list[str] | None = None,
    incident_model_type: str | None = None,
    incident_fallback_reason: str | None = None,
    events_detected_count: int | None = None,
    events_detected_types: list[str] | None = None,
    incident_tracks_involved: list[int] | None = None,
    stopped_vehicle_candidates: int | None = None,
    collision_candidates: int | None = None,
) -> dict[str, Any]:
    summary = {
        "status": status,
        "summary_status": status,
        "mode": spec.mode,
        "model_type": spec.model_type,
        "secondary_model_type": incident_model_type or spec.incident_model_type,
        "open_vocab_enabled": spec.open_vocab_enabled,
        "incident_enabled": spec.incident_enabled,
        "fallback_reason": spec.fallback_reason,
        "incident_fallback_reason": incident_fallback_reason or spec.incident_fallback_reason,
        "source_path": source,
        "run_name": run_name,
        "started_at": started_at,
        "finished_at": finished_at,
        "prompts_used": prompts_used if prompts_used is not None else (spec.incident_prompts if spec.incident_enabled else spec.prompts),
        "frames_processed": frames_processed,
        "detections_total": detections_total,
        "tracked_objects_total": tracked_objects_total,
        "classes_seen": classes_seen,
        "events_detected": event_counts,
        "events_detected_count": events_detected_count if events_detected_count is not None else sum(event_counts.values()),
        "events_detected_types": events_detected_types if events_detected_types is not None else sorted(event_counts.keys()),
        "incident_tracks_involved": incident_tracks_involved or [],
        "stopped_vehicle_candidates": stopped_vehicle_candidates if stopped_vehicle_candidates is not None else event_counts.get("stopped_vehicle", 0),
        "collision_candidates": collision_candidates if collision_candidates is not None else event_counts.get("collision_candidate", 0),
        "output_video_path": str(output_video_path) if output_video_path else None,
        "output_dir": str(output_dir) if output_dir else None,
        "alerts_dir": str(alerts_dir) if alerts_dir else None,
        "summary_path": str(report_path),
        "events_path": str(events_json) if events_json else None,
        "timeline_path": str(timeline_json) if timeline_json else None,
        "artifacts_path": str(artifacts_json) if artifacts_json else None,
        "predictions_jsonl_path": str(predictions_jsonl) if predictions_jsonl else None,
        "predictions_csv_path": str(predictions_csv) if predictions_csv else None,
        "timeline_csv_path": str(timeline_csv) if timeline_csv else None,
        "skipped_reason": reason,
        "mean_latency_ms": mean_latency_ms,
        "fps_estimate": fps_estimate,
        "alert_counts": event_counts,
    }
    return summary
