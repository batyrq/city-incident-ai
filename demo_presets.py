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
MK = "Mixkit Stock Video Free License"
WC = "https://commons.wikimedia.org/wiki/File:"
# Real incident clips for the incident detectors (fire / smoke / flood / crash). `licence` is per clip. What each clip does in the detector is in
# reports/metrics/incident_videos.json (scripts/incident_clips_report.py writes demo_videos/incidents/README.md from it).
INCIDENT_PRESETS = [
    dict(id="g", note="", file="incidents/fire_building_burning.mp4", label="g · Fire: building burning", pose=False, max_frames=0, licence=MK,
         shows="Flames and smoke fill the frame.", source="https://mixkit.co/free-stock-video/building-burning-down-22587/", origin="mixkit 22587, first 20 s"),
    dict(id="h", note="", file="incidents/fire_grass_firefighter.mp4", label="h · Fire: grass fire", pose=False, max_frames=0, licence=MK,
         shows="Grass fire with a firefighter walking past.", source="https://mixkit.co/free-stock-video/an-aggressive-fire-consuming-the-nature-32370/", origin="mixkit 32370, whole clip"),
    dict(id="i", note="", file="incidents/smoke_city_building_fire.mp4", label="i · Smoke: city building fire", pose=False, max_frames=0, licence=MK,
         shows="Aerial view of a large smoke plume over a street with fire engines.", source="https://mixkit.co/free-stock-video/firefighters-putting-out-a-big-fire-5280/", origin="mixkit 5280, first 20 s"),
    dict(id="j", note="The crash detector fires at its first possible moment, before the collision. It also fires on dashcam clips with no crash (see the audit in the README).", file="incidents/crash_dashcam_highway.mp4",
         label="j · Crash: dashcam highway", pose=False, max_frames=0, licence="Public domain (Wikimedia Commons; dashcam footage)",
         shows="Dashcam: cars collide ahead at about 2-7 s.", source=WC + "White_cars_hit_the_lane_divider_at_high_speed_on_the_highway.webm", origin="Commons, first 20 s"),
    dict(id="k", note="Fires long before the collision at about 8.5 s (camera-style bias).", file="incidents/crash_dashcam_truck.mp4", label="k · Crash: dashcam truck", pose=False, max_frames=0,
         licence="Public domain (Wikimedia Commons; dashcam footage)", shows="Dashcam behind a truck, collision at about 8.5 s.",
         source=WC + "The_speeding_car_rear-ended_into_the_truck_in_front%2C_and_the_front_hood_was_blown_open.webm", origin="Commons, 12-30 s"),
    dict(id="l", note="", file="incidents/crash_aftermath_truck.mp4", label="l · Crash: aftermath", pose=False, max_frames=0, licence=MK,
         shows="Static shot of a lorry cab lying on its side.", source="https://mixkit.co/free-stock-video/trailer-crash-on-a-highway-49147/", origin="mixkit 49147, whole clip"),
    dict(id="m", note="", file="incidents/flood_street_muddy.mp4", label="m · Flood: muddy street", pose=False, max_frames=0, licence="CC BY-SA 4.0 (Amuzujoe, Wikimedia Commons)",
         shows="Street fully under brown water, handheld.", source=WC + "Flood_on_the_street.webm", origin="Commons, whole clip"),
    dict(id="n", note="Has a Chinese news overlay; the crash detector also fires once.", file="incidents/flood_street_zhengzhou.mp4", label="n · Flood: Zhengzhou street", pose=False, max_frames=0,
         licence="CC BY 3.0 (China News Service, Wikimedia Commons)", shows="People wading through a flooded street.", source=WC + "Zhengzhou_streets_during_the_flood_2021-07-20.webm", origin="Commons, first 16 s"),
    dict(id="o", note="", file="incidents/flood_street_manila.mp4", label="o · Flood: Manila street", pose=False, max_frames=0, licence="CC BY-SA 4.0 (Exec8, Wikimedia Commons)",
         shows="Flooded street seen from a vehicle (10 fps source).", source=WC + "Flooding_in_Hereford_Street%2C_Project_8%2C_Quezon_City%2C_Philippines_from_Typhoon_Ketsana_%28Ondoy%29_2.webm", origin="Commons, first 20 s"),
]
PRESETS = PRESETS + INCIDENT_PRESETS
DEFAULT_PRESET = "b"
DIR = "demo_videos"
