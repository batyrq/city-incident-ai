import json
from pathlib import Path

from src.infer.incidents import EVENT_NAME, IncidentMonitor

DET = {"classes": {"fire": {"threshold": 0.5}, "crash": {"threshold": 0.7}}}


def feed(mon, series, dt=0.5):
    out = []
    for i, (f, c) in enumerate(series):
        out += mon.update({"fire": {"score": f, "box": None}, "crash": {"score": c, "box": None}}, i * dt)
    return out


def test_needs_k_of_n():
    ev = feed(IncidentMonitor(DET, k=3, n=5), [(0.9, 0), (0.1, 0), (0.9, 0), (0.1, 0)])
    assert ev == []  # only 2 positives so far
    ev = feed(IncidentMonitor(DET, k=3, n=5), [(0.9, 0), (0.1, 0), (0.9, 0), (0.1, 0), (0.9, 0)])
    assert [e["class"] for e in ev] == ["fire"] and ev[0]["t_sec"] == 2.0


def test_cooldown_and_event_names():
    ev = feed(IncidentMonitor(DET, k=2, n=3, cooldown_seconds=10), [(0.9, 0.9)] * 30)
    assert sorted(e["event_type"] for e in ev if e["t_sec"] < 10) == ["crash_candidate", "fire"]
    assert max(e["t_sec"] for e in ev) >= 10  # fires again after the cooldown
    assert EVENT_NAME["crash"] == "crash_candidate"


def test_detector_config_complete():
    cfg = json.loads((Path(__file__).parent.parent / "configs/incident_detectors.json").read_text())
    assert set(cfg["classes"]) == {"fire", "smoke", "flood", "crash"}
    for spec in cfg["classes"].values():
        assert 0 < spec["threshold"] < 1 and spec["approach"] in {"yolo", "probe", "clip_zeroshot"}
