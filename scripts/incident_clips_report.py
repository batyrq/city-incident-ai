"""Run the full pipeline (incident detectors on) on demo_videos/incidents/*.mp4 and write demo_videos/incidents/README.md.

  python scripts/incident_clips_report.py   -> reports/metrics/demo_incident_clips.json + demo_videos/incidents/README.md
Numbers: pipeline events come from the runs above; the detector-level timing numbers come from reports/metrics/incident_videos.json (eval/incidents_videos.py).
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from demo import run_demo  # noqa: E402
from demo_presets import DIR, INCIDENT_PRESETS  # noqa: E402

vid = json.loads((ROOT / "reports/metrics/incident_videos.json").read_text())
rows = []
for p in INCIDENT_PRESETS:
    result, _ = run_demo(str(ROOT / DIR / p["file"]), run_suffix="_incclips")
    ev = json.loads((ROOT / "reports" / f"{result['run_name']}_events.json").read_text())
    name = Path(p["file"]).stem
    v = vid["incident_clips"][name]
    rows.append({"id": p["id"], "file": p["file"], "label": p["label"], "licence": p["licence"], "source": p["source"], "origin": p["origin"], "shows": p["shows"], "note": p["note"],
                 "seconds": v["seconds_scored"], "expected": v["expected"], "incident_from_s": v["incident_from_s"], "frames": result["frames_processed"], "pipeline_fps": result.get("pipeline_fps"),
                 "events": [{"type": e["event_type"], "at_s": e["start_time_sec"], "confidence": (e.get("incident_detector") or {}).get("confidence")} for e in ev if e.get("source") == "incident_detector"],
                 "other_events": sorted({e["event_type"] for e in ev if e.get("source") != "incident_detector"})})
    print(p["id"], rows[-1]["events"], flush=True)
(ROOT / "reports/metrics/demo_incident_clips.json").write_text(json.dumps(rows, indent=2))
L = ["# Incident demo clips", "",
     "Nine short real clips for the incident detectors (fire, smoke, flood, crash_candidate). They are in the **Try it live** dropdown (ids g-o). "
     "Mixkit clips: Mixkit Stock Video Free License. Wikimedia Commons clips: licence per clip below. All trimmed to at most 20 s, 720p H.264, no audio.", "",
     "| File | What it shows (expected class, incident visible from) | Events the pipeline fired | Licence / source |", "|---|---|---|---|"]
for r in rows:
    ev = "; ".join(f"`{e['type']}` at {e['at_s']:.1f} s (conf {e['confidence']:.2f})" for e in r["events"]) or "none"
    L.append(f"| `{r['file'].split('/')[-1]}` ({r['seconds']} s) | {r['shows']} Expected `{r['expected']}`, from {r['incident_from_s']} s.{(' **Note:** ' + r['note']) if r['note'] else ''} | {ev} | {r['licence']}, [{r['origin']}]({r['source']}) |")
L += ["", "Notes", "",
      "- Events come from the default demo config (`incidents.enabled: true`, a frame scored every 0.5 s, fire when 3 of the last 5 scored frames are above the class threshold, so the earliest possible event is at 1.0 s; 10 s cooldown per class).",
      "- The ground-truth start times were written by looking at the clips. `crash_candidate` is the weak class: see `reports/metrics/incident_crash_audit.json` and the README.", ""]
(ROOT / DIR / "incidents/README.md").write_text("\n".join(L))
