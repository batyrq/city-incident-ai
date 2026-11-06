"""Video-level check of the selected incident detectors on real incident clips and on normal clips.

  python -m eval.incidents_videos     -> reports/metrics/incident_videos.json

Frames are sampled every `sample_seconds`; a class fires when k of the last n samples are above its (validation) threshold.
Incident clips: demo_videos/incidents/*.mp4 (ground truth below was written by looking at the clips; start = first second the incident is visible).
Normal clips (negatives): demo_videos/a-f plus the Mixkit hard-negative clips that belong to the TEST split (never used for training or thresholds).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from eval.incidents_clips import HARD, hard_split  # noqa: E402
from src.infer.incidents import DEFAULTS, IncidentMonitor, IncidentScorer, load_detector_cfg  # noqa: E402

# name: (expected class, incident visible from s, until s, note)
INCIDENT_CLIPS = {
    "fire_building_burning": ("fire", 0.0, 20.0, "burning building, flames fill the frame"),
    "fire_grass_firefighter": ("fire", 0.0, 10.5, "grass fire with a firefighter"),
    "smoke_city_building_fire": ("smoke", 0.0, 20.0, "large smoke plume from a building fire, aerial; flames barely visible"),
    "crash_dashcam_highway": ("crash", 2.0, 7.0, "dashcam: cars collide ahead, dust and spinning vehicles about 2-7 s"),
    "crash_dashcam_truck": ("crash", 8.5, 18.0, "dashcam: truck ahead, collision at about 8.5 s, debris on the windscreen afterwards"),
    "crash_aftermath_truck": ("crash", 0.0, 8.2, "static aftermath: lorry cab lying on its side"),
    "flood_street_muddy": ("flood", 0.0, 12.3, "street fully under brown water, handheld"),
    "flood_street_zhengzhou": ("flood", 0.0, 16.0, "people wading in a flooded street, news overlay text"),
    "flood_street_manila": ("flood", 0.0, 20.0, "flooded street seen from a vehicle, 10 fps"),
}


def run_clip(path: Path, scorer: IncidentScorer, det: dict, max_s: float = 30.0) -> dict:
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    dur = n / fps
    step = max(1, round(fps * DEFAULTS["sample_seconds"]))
    mon = IncidentMonitor(det, DEFAULTS["k"], DEFAULTS["n"], DEFAULTS["cooldown_seconds"])
    series = {c: [] for c in scorer.classes}
    fired: dict[str, list] = {c: [] for c in scorer.classes}
    i = 0
    while True:
        ok, fr = cap.read()
        if not ok or i / fps > max_s:
            break
        if i % step == 0:
            sc = scorer.score(fr)
            t = i / fps
            for c in scorer.classes:
                series[c].append(round(sc[c]["score"], 3))
            for e in mon.update(sc, t):
                fired[e["class"]].append(e["t_sec"])
        i += 1
    cap.release()
    return {"seconds_scored": round(min(dur, max_s), 1), "fps": round(fps, 2), "samples": len(series[scorer.classes[0]]),
            "fired_at_s": fired, "max_score": {c: max(series[c]) if series[c] else 0 for c in scorer.classes}, "score_series": series}


def main() -> None:
    det = load_detector_cfg(DEFAULTS["detectors_file"])
    scorer = IncidentScorer(det)
    out = {"settings": {"sample_seconds": DEFAULTS["sample_seconds"], "k_of_n": [DEFAULTS["k"], DEFAULTS["n"]], "cooldown_s": DEFAULTS["cooldown_seconds"],
                        "detectors": {c: {"approach": s["approach"], "threshold": s["threshold"]} for c, s in det["classes"].items()}}, "incident_clips": {}, "normal_clips": {}}
    for name, (cls, t0, t1, note) in INCIDENT_CLIPS.items():
        r = run_clip(ROOT / "demo_videos/incidents" / f"{name}.mp4", scorer, det)
        first = r["fired_at_s"][cls][0] if r["fired_at_s"][cls] else None
        r.update({"expected": cls, "incident_from_s": t0, "incident_until_s": t1, "note": note, "detected": first is not None,
                  "first_fire_s": first, "fired_before_incident_started": bool(first is not None and first < t0),
                  "delay_after_incident_start_s": round(first - t0, 2) if first is not None else None,
                  "other_classes_fired": {c: v[:1] for c, v in r["fired_at_s"].items() if c != cls and v}})
        out["incident_clips"][name] = r
        print(name, cls, "->", first, "other:", r["other_classes_fired"], flush=True)
    normals = [(f"demo_videos/{p.name}", p) for p in sorted((ROOT / "demo_videos").glob("[a-f]_*.mp4"))]
    normals += [(f"mixkit-{vid} ({HARD[vid][1]})", ROOT / "data/incidents/raw/videos/hard" / f"{vid}.mp4") for vid in sorted(HARD) if hard_split(vid) == "test"]
    for label, p in normals:
        if not p.exists():
            continue
        r = run_clip(p, scorer, det)
        r["false_alarms"] = {c: v for c, v in r["fired_at_s"].items() if v}
        out["normal_clips"][label] = r
        print(label, "false alarms:", r["false_alarms"], flush=True)
    # dashcam / CCTV controls: titles say "collision" but I saw none when stepping through them, so they are only informal evidence about camera-style bias
    out["dashcam_controls_no_collision_seen"] = {}
    for label, f in (("c4 highway dashcam", "c4"), ("c5 street CCTV", "c5")):
        r = run_clip(ROOT / "data/incidents/raw/videos/commons" / f"{f}.webm", scorer, det, max_s=20.0)
        r["false_alarms"] = {c: v for c, v in r["fired_at_s"].items() if v}
        out["dashcam_controls_no_collision_seen"][label] = r
        print("control", label, r["false_alarms"], flush=True)
    inc = out["incident_clips"]
    out["summary"] = {"incident_clips": len(inc), "detected": sum(1 for r in inc.values() if r["detected"]),
                      "detected_before_incident_started": sum(1 for r in inc.values() if r["fired_before_incident_started"]),
                      "controls_with_false_crash": sum(1 for r in out["dashcam_controls_no_collision_seen"].values() if r["fired_at_s"]["crash"]),
                      "normal_clips": len(out["normal_clips"]), "normal_clips_with_false_alarm": sum(1 for r in out["normal_clips"].values() if r["false_alarms"]),
                      "false_alarms_total": sum(len(v) for r in out["normal_clips"].values() for v in r["false_alarms"].values()),
                      "normal_seconds_scored": round(sum(r["seconds_scored"] for r in out["normal_clips"].values()), 1)}
    (ROOT / "reports/metrics/incident_videos.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out["summary"]))


if __name__ == "__main__":
    main()
