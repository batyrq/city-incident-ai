"""The six interview demo clips in demo_videos/: one source of truth for the live-demo dropdown and demo_videos/README.md.

Every clip is a real-time (not time-lapse) free Mixkit video, trimmed to <= 20 s, re-encoded to 720p H.264.
Numbers about what each clip produces are NOT stored here: scripts/demo_videos_report.py runs the pipeline and writes them.
"""
from __future__ import annotations

LICENCE = "Mixkit Stock Video Free License (https://mixkit.co/license/#videoFree)"

PRESETS = [
    dict(id="a", note="", file="a_busy_street_elevated.mp4", label="a · Busy street (elevated)", pose=False, max_frames=0,
         shows="Fixed elevated camera over a crosswalk at golden hour, cars and pedestrians. The clean-tracking case.",
         source="https://mixkit.co/free-stock-video/aerial-view-of-people-at-pedestrian-crossing-61/", origin="mixkit 61, first 20 s"),
    dict(id="b", note='A pushchair is labelled `motorcycle` in this clip (visible at about 5 s).', file="b_pedestrian_crowd.mp4", label="b · Pedestrian crowd", pose=False, max_frames=0,
         shows="Dense group of pedestrians walking towards a fixed camera. Meant to fire the crowding event.",
         source="https://mixkit.co/free-stock-video/popular-street-with-pedestrians-taking-a-walk-9261/", origin="mixkit 9261, whole clip"),
    dict(id="c", note='The event fires at the first moment the 15 s rule allows, because the parked and queued cars are stationary from the first frame. No incident is shown. Among the street clips I scanned, this is the only one that fires it within 20 s.', file="c_stopped_vehicle.mp4", label="c · Stopped vehicles", pose=False, max_frames=0,
         shows="Fixed camera over a street with parked and waiting cars. The only clip found where stopped_vehicle fires within 20 s.",
         source="https://mixkit.co/free-stock-video/traffic-light-directing-traffic-4272/", origin="mixkit 4272, first 20 s"),
    dict(id="d", note='Known weakness, see `docs/ERROR_ANALYSIS.md` d): many short-lived track IDs for a continuous flow of pedestrians.', file="d_rainy_night.mp4", label="d · Rainy night (weakness)", pose=False, max_frames=0,
         shows="Times Square on a rainy night: glare, motion blur and umbrellas. Shows the track-ID churn described in docs/ERROR_ANALYSIS.md.",
         source="https://mixkit.co/free-stock-video/times-square-during-a-rainy-night-4332/", origin="mixkit 4332, whole clip"),
    dict(id="e", note='The class list also contains `bicycle` and `motorcycle`, which are false detections: the scene has neither.', file="e_treadmill_runner_side.mp4", label="e · Treadmill runner · pose", pose=True, max_frames=0,
         shows="Man running on a treadmill filmed from the side, full body visible. The pose-mode / gait-analysis connection.",
         source="https://mixkit.co/free-stock-video/young-man-does-cardio-workout-at-the-gym-49276/", origin="mixkit 49276, first 20 s"),
    dict(id="f", note="", file="f_track_runner_front.mp4", label="f · Track jogger · pose", pose=True, max_frames=0,
         shows="Man jogging towards the camera on an outdoor running track, full body visible. A second pose example (front view).",
         source="https://mixkit.co/free-stock-video/man-jogging-on-a-running-track-on-a-sunny-day-32799/", origin="mixkit 32799, whole clip"),
]
DEFAULT_PRESET = "b"
DIR = "demo_videos"
