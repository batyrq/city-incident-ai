"""Mixkit clips used to build the incident test sets: hard negatives (by theme), real-incident clips for the video-level check.

Mixkit Stock Video Free License (https://mixkit.co/license/#videoFree). slug -> https://mixkit.co/free-stock-video/<slug>/
  python -m eval.incidents_clips        # downloads data/incidents/raw/videos/{hard,incident}/<id>.mp4
"""
from __future__ import annotations

import subprocess
import time
from pathlib import Path

# id: (slug, theme)   themes: sunset | lights | neon | steam | water | trafficjam | emergency
HARD = {
    3983: ("sunset-perspective-in-the-city-3983", "sunset"), 50134: ("two-cars-followed-each-other-on-the-road-at-sunset-50134", "sunset"),
    41375: ("tour-high-above-a-city-at-dusk-41375", "sunset"), 20298: ("street-lights-at-sunset-20298", "sunset"),
    2168: ("bright-orange-sunset-on-beach-2168", "sunset"), 4999: ("beautiful-sunset-on-a-bay-from-above-4999", "sunset"),
    4119: ("stunning-sunset-seen-from-the-sea-4119", "sunset"), 38948: ("sunset-in-the-open-sea-from-the-top-of-a-38948", "sunset"),
    25566: ("young-woman-skating-down-the-street-in-sunset-light-25566", "sunset"),
    6754: ("city-street-traffic-by-night-6754", "lights"), 3118: ("busy-avenue-3118", "lights"), 22778: ("blurred-traffic-lights-and-cars-22778", "lights"),
    22367: ("car-lights-throughout-the-night-22367", "lights"), 19498: ("car-lights-in-traffic-bokeh-at-night-19498", "lights"),
    29931: ("car-lights-in-traffic-on-an-avenue-29931", "lights"), 1585: ("cars-drive-on-a-two-way-street-at-night-1585", "lights"),
    7367: ("busy-night-street-7367", "lights"), 42039: ("view-from-inside-a-car-in-night-traffic-42039", "lights"),
    44693: ("white-and-red-car-lights-in-a-defocused-shot-44693", "lights"), 22825: ("car-lights-blurred-against-the-street-22825", "lights"),
    16574: ("car-lights-across-the-city-16574", "lights"),
    4447: ("neon-signs-with-japanese-letters-4447", "neon"), 41154: ("neon-light-of-a-hotel-flickering-in-the-dark-41154", "neon"),
    4232: ("business-advertising-signs-in-the-street-4232", "neon"), 2012: ("chinatown-street-at-night-2012", "neon"),
    17263: ("orange-lights-around-a-mosque-17263", "neon"), 25952: ("blinking-white-and-orange-lights-25952", "neon"),
    28825: ("steam-coming-out-of-household-chimneys-28825", "steam"), 9678: ("night-fog-in-the-city-highway-9678", "steam"),
    9907: ("industrial-chimney-unloading-pollution-to-the-air-9907", "steam"), 47738: ("industrial-factory-chimney-with-emissions-47738", "steam"),
    4362: ("smoke-from-power-plant-4362", "steam"),
    27704: ("cars-heading-through-heavy-rain-27704", "water"), 9405: ("heavy-rain-in-traffic-9405", "water"),
    48004: ("drops-of-rain-falling-on-the-street-48004", "water"), 22646: ("crossing-a-wet-road-at-night-22646", "water"),
    28116: ("blurred-car-lights-through-the-rain-28116", "water"), 20447: ("car-lights-in-the-rain-20447", "water"),
    4331: ("traffic-on-a-rainy-night-4331", "water"), 10000: ("a-wild-river-between-the-trees-10000", "water"),
    15299: ("dangerous-river-flowing-over-rocks-15299", "water"), 6773: ("swiming-pool-in-resort-garden-6773", "water"),
    3469: ("children-playing-with-a-dancing-fountain-3469", "water"), 43615: ("aerial-view-of-a-river-with-people-in-and-around-43615", "water"),
    24797: ("waiting-in-a-traffic-jam-24797", "trafficjam"), 36371: ("traffic-jam-on-a-highway-taken-from-above-36371", "trafficjam"),
    26834: ("an-ambulance-in-the-streets-tracking-shot-26834", "emergency"), 14089: ("ambulance-crossing-the-roads-14089", "emergency"),
    17376: ("police-car-lights-flashing-17376", "emergency"), 27695: ("hazard-lights-flashing-on-a-car-27695", "emergency"),
    33464: ("man-frustrated-with-his-broken-down-car-on-a-road-33464", "emergency"), 46936: ("fire-truck-arriving-at-the-tunnel-to-conduct-the-rescue-46936", "emergency"),
}
# real incident clips (stock footage, not CCTV): id: (slug, kind)
INCIDENT = {
    22587: ("building-burning-down-22587", "fire"), 32370: ("an-aggressive-fire-consuming-the-nature-32370", "fire"),
    5280: ("firefighters-putting-out-a-big-fire-5280", "fire"),
    49233: ("aerial-view-of-a-car-accident-on-a-busy-highway-49233", "crash"), 47109: ("car-accident-inside-a-tunnel-with-rescue-team-in-action-47109", "crash"),
    49147: ("trailer-crash-on-a-highway-49147", "crash"),
    9953: ("aerial-view-of-flood-and-disrupted-roads-9953", "flood"), 28089: ("flooded-construction-site-28089", "flood"),
}


def fetch(group: str, table: dict) -> None:
    out = Path("data/incidents/raw/videos") / group
    out.mkdir(parents=True, exist_ok=True)
    for vid in table:
        dst = out / f"{vid}.mp4"
        if dst.exists() and dst.stat().st_size > 10_000:
            continue
        for res in ("720", "360"):
            r = subprocess.run(["curl", "-sL", "-m", "120", "-o", str(dst), "-w", "%{http_code}", f"https://assets.mixkit.co/videos/{vid}/{vid}-{res}.mp4"], capture_output=True, text=True)
            if r.stdout.strip() == "200" and dst.stat().st_size > 10_000:
                break
            dst.unlink(missing_ok=True)
        print(group, vid, "ok" if dst.exists() else "FAILED", flush=True)


if __name__ == "__main__":
    fetch("hard", HARD)
    fetch("incident", INCIDENT)


# Wikimedia Commons clips (real CCTV / dashcam / phone footage). title -> kind. Licence is read from the file page by fetch_commons().
COMMONS = {
    "Flood on the street.webm": "flood", "Zhengzhou streets during the flood 2021-07-20.webm": "flood",
    "Flooded Street of Obiwali road in Rumuigbo Rivers State after a heavy down pour.webm": "flood",
    "Flooding in Hereford Street, Project 8, Quezon City, Philippines from Typhoon Ketsana (Ondoy) 2.webm": "flood",
    "Multiple cars rear-end collision on highway.webm": "crash", "Jilin Tonghua car rear-end collision accident, 2025-03-11.webm": "crash",
    "The car hit the woman crossing the zebra crossing and the white car on the left at the exit of the roundabout at the same time.webm": "crash",
    "White cars hit the lane divider at high speed on the highway.webm": "crash",
    "The speeding car rear-ended into the truck in front, and the front hood was blown open.webm": "crash",
}


def fetch_commons() -> list[dict]:
    import json
    import urllib.parse
    import urllib.request
    out = []
    dst = Path("data/incidents/raw/videos/commons")
    dst.mkdir(parents=True, exist_ok=True)
    hdr = {"User-Agent": "city-incident-ai-research/0.1 (interview prep; local use)"}
    for i, (title, kind) in enumerate(COMMONS.items()):
        q = "https://commons.wikimedia.org/w/api.php?" + urllib.parse.urlencode({"action": "query", "titles": "File:" + title, "prop": "imageinfo", "iiprop": "url|size|extmetadata", "format": "json"})
        for attempt in range(6):  # Commons answers 429 when asked too fast
            try:
                info = next(iter(json.load(urllib.request.urlopen(urllib.request.Request(q, headers=hdr), timeout=60))["query"]["pages"].values()))["imageinfo"][0]
                break
            except Exception:
                time.sleep(20 * (attempt + 1))
        md = info["extmetadata"]
        lic = md.get("LicenseShortName", {}).get("value", "?")
        raw = dst / f"c{i}.webm"
        if not raw.exists():
            for attempt in range(6):
                try:
                    raw.write_bytes(urllib.request.urlopen(urllib.request.Request(info["url"], headers=hdr), timeout=300).read())
                    break
                except Exception:
                    time.sleep(30 * (attempt + 1))
            time.sleep(6)
        out.append(dict(file=raw.name, kind=kind, title=title, licence=lic, author=md.get("Artist", {}).get("value", "?")[:80], page="https://commons.wikimedia.org/wiki/File:" + urllib.parse.quote(title.replace(" ", "_"))))
        print(out[-1], flush=True)
    (dst / "commons.json").write_text(json.dumps(out, indent=2))
    return out


def hard_split(vid: int) -> str:
    """Same rule as eval/incidents_data.py: per theme, every 5th clip is test, the next is val, the rest train."""
    theme = HARD[vid][1]
    vids = [v for v, (_, t) in sorted(HARD.items()) if t == theme]
    i = vids.index(vid)
    return "test" if i % 5 == 0 else "val" if i % 5 == 1 else "train"
