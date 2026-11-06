"""Build the image-level incident datasets (fire, smoke, flood, crash) from the raw downloads.

  python -m eval.incidents_data    -> data/incidents/<task>/<split>/*.jpg, data/incidents/manifest.csv,
                                      data/incidents/SOURCES.md, reports/metrics/incident_data.json

Rules: splits never share a source video (negatives) and test positives are de-duplicated against the training
split with a perceptual hash (dHash, Hamming <= 4). All numbers in SOURCES.md / incident_data.json are counted here.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from eval.incidents_clips import HARD  # noqa: E402

RAW = ROOT / "data/incidents/raw"
OUT = ROOT / "data/incidents"
MAXSIDE = 640
CAP = {"test": (300, 600), "val": (200, 400)}  # (positives, negatives) per class and split
rng = random.Random(0)


def dhash(img: Image.Image) -> int:
    a = np.asarray(img.convert("L").resize((9, 8), Image.BILINEAR), dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def ham(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def shrink(img: Image.Image) -> Image.Image:
    img = img.convert("RGB")
    s = MAXSIDE / max(img.size)
    return img.resize((round(img.width * s), round(img.height * s)), Image.BILINEAR) if s < 1 else img


def h(s: str) -> int:
    return int(hashlib.md5(s.encode()).hexdigest(), 16)


class Sink:
    def __init__(self) -> None:
        self.rows: list[dict] = []

    def add(self, task, split, label, group, img: Image.Image, name: str, source: str, dh: int | None = None) -> None:
        d = OUT / task / split
        d.mkdir(parents=True, exist_ok=True)
        p = d / f"{name}.jpg"
        shrink(img).save(p, quality=90)
        self.rows.append(dict(task=task, split=split, label=label, group=group, path=str(p.relative_to(ROOT)), source=source,
                              dhash=dhash(img) if dh is None else dh))


def video_frames(path: Path, n: int, skip_edges: float = 0.05) -> list[Image.Image]:
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = []
    for i in np.linspace(int(total * skip_edges), max(1, total - 1 - int(total * skip_edges)), n).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, fr = cap.read()
        if ok:
            out.append(Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)))
    return out


def mot_frames(seq: Path, n: int) -> list[Image.Image]:
    fs = sorted((seq / "img1").glob("*.jpg"))
    return [Image.open(fs[int(i)]) for i in np.linspace(0, len(fs) - 1, n)]


# ---------------------------------------------------------------- negatives from video, grouped by source clip
def video_negatives() -> list[dict]:
    """Every item: dict(clip, group, split, frames). Splits are per clip, so no clip appears in two splits."""
    items = []
    for sid, split, n in (("02", "train", 40), ("04", "val", 40), ("09", "train", 40)):
        seq = ROOT / f"data/mot/MOT17/train/MOT17-{sid}-FRCNN"
        if seq.exists():
            items.append(dict(clip=f"mot17-{sid}", group="street", split=split, frames=mot_frames(seq, n), source=f"MOT17-{sid}"))
    for f, split in (("a_busy_street_elevated", "test"), ("b_pedestrian_crowd", "test"), ("c_stopped_vehicle", "test"), ("d_rainy_night", "test")):
        p = ROOT / "demo_videos" / f"{f}.mp4"
        if p.exists():
            items.append(dict(clip=f, group="street", split=split, frames=video_frames(p, 40), source=f"demo_videos/{f}.mp4 (Mixkit)"))
    by_theme = defaultdict(list)
    for vid, (_, theme) in sorted(HARD.items()):
        by_theme[theme].append(vid)
    for theme, vids in by_theme.items():
        for i, vid in enumerate(vids):
            split = "test" if i % 5 == 0 else "val" if i % 5 == 1 else "train"
            p = RAW / "videos/hard" / f"{vid}.mp4"
            if p.exists():
                items.append(dict(clip=f"mixkit-{vid}", group="hard", theme=theme, split=split, frames=video_frames(p, 12), source=f"Mixkit {vid} ({theme})"))
    return items


def parquet_rows(pattern: str, cols: list[str]):
    for p in sorted(RAW.glob(pattern)):
        f = pq.ParquetFile(p)
        for rg in range(f.num_row_groups):
            t = f.read_row_group(rg, columns=cols).to_pylist()
            yield from t


def pick(items: list, k: int) -> list:
    items = list(items)
    rng.shuffle(items)
    return items[:k]


def main() -> None:
    sink = Sink()
    vneg = video_negatives()

    # ---- add video frames to every task as negatives (street / hard), per clip split
    def add_video(task: str, groups: set[str] | None = None, themes: set[str] | None = None) -> None:
        for it in vneg:
            if it["group"] == "hard" and themes is not None and it["theme"] not in themes:
                continue
            for j, im in enumerate(it["frames"]):
                sink.add(task, it["split"], 0, it["group"], im, f"{it['clip']}_{j:02d}", it["source"])

    ALL = {"sunset", "lights", "neon", "steam", "water", "trafficjam", "emergency"}
    add_video("fire", themes={"sunset", "lights", "neon", "steam", "emergency"})
    add_video("smoke", themes={"sunset", "lights", "neon", "steam", "emergency"})
    add_video("flood", themes={"water", "lights", "trafficjam"})
    add_video("crash", themes={"lights", "trafficjam", "emergency", "water"})

    # ---- D-Fire (fire, smoke)
    dfire = ROOT / "data/incidents/raw/dfire"
    recs = {"train": [], "test": []}
    for s in recs:
        for lf in sorted((dfire / s / "labels").glob("*.txt")):
            ids = {int(l.split()[0]) for l in lf.read_text().splitlines() if l.strip()}
            ip = dfire / s / "images" / (lf.stem + ".jpg")
            if not ip.exists():
                cand = list((dfire / s / "images").glob(lf.stem + ".*"))
                if not cand:
                    continue
                ip = cand[0]
            recs[s].append((ip, ids))
    # hold 10 % of the official train split out for validation (thresholds, YOLO val); the YOLO training set is the rest
    trn, val = [], []
    for r in recs["train"]:
        (val if h(r[0].stem) % 10 == 0 else trn).append(r)
    dropped = Counter()
    # near-duplicate filter: test vs train
    train_hashes = []
    for ip, _ in trn:
        train_hashes.append(dhash(Image.open(ip)))
    th = np.array(train_hashes, dtype=np.uint64)

    def near_dup(dh: int) -> bool:
        x = np.bitwise_xor(th, np.uint64(dh))
        counts = np.unpackbits(x.view(np.uint8).reshape(-1, 8), axis=1).sum(1)
        return bool((counts <= 4).any())

    def dfire_pick(pool, split, task):
        cls = 1 if task == "fire" else 0
        pos = [r for r in pool if cls in r[1]]
        neg = [r for r in pool if not r[1]]
        pk, nk = CAP[split]
        out_pos = []
        for ip, ids in pick(pos, pk * 2 if split == "test" else pk):
            im = Image.open(ip)
            dh = dhash(im)
            if split == "test" and near_dup(dh):
                dropped[(task, "positive")] += 1
                continue
            out_pos.append((ip, im, dh))
            if len(out_pos) == pk:
                break
        n_video = sum(1 for r in sink.rows if r["task"] == task and r["split"] == split and r["label"] == 0)
        out_neg = []
        for ip, ids in pick(neg, nk * 2 if split == "test" else nk):
            if len(out_neg) >= max(0, nk - n_video):
                break
            im = Image.open(ip)
            dh = dhash(im)
            if split == "test" and near_dup(dh):
                dropped[(task, "negative")] += 1
                continue
            out_neg.append((ip, im, dh))
        for ip, im, dh in out_pos:
            sink.add(task, split, 1, "positive", im, f"dfire_{ip.stem}", "D-Fire", dh)
        for ip, im, dh in out_neg:
            sink.add(task, split, 0, "background", im, f"dfire_{ip.stem}", "D-Fire (no label)", dh)

    for task in ("fire", "smoke"):
        dfire_pick(recs["test"], "test", task)
        dfire_pick(val, "val", task)

    # ---- crash: justjuu positives, hiennguyen negatives
    jp = {"train": [], "val": [], "test": []}
    for p in sorted(RAW.glob("justjuu*/data/*.parquet")):
        sp = "train" if "train" in p.name else "val" if "validation" in p.name else "test"
        f = pq.ParquetFile(p)
        for rg in range(f.num_row_groups):
            for r in f.read_row_group(rg, columns=["image", "objects"]).to_pylist():
                if 0 in r["objects"]["category"]:
                    jp[sp].append(r["image"]["bytes"])
    train_dh = np.array([dhash(Image.open(io.BytesIO(b))) for b in jp["train"]], dtype=np.uint64)

    def near_dup_arr(arr, dh):
        x = np.bitwise_xor(arr, np.uint64(dh))
        return bool((np.unpackbits(x.view(np.uint8).reshape(-1, 8), axis=1).sum(1) <= 4).any())

    for sp, key in (("test", "test"), ("val", "val"), ("train", "train")):
        k = CAP[sp][0] if sp in CAP else 1500
        n = 0
        for b in pick(jp[key], k * 2):
            im = Image.open(io.BytesIO(b))
            dh = dhash(im)
            if sp != "train" and near_dup_arr(train_dh, dh):
                dropped[("crash", "positive")] += 1
                continue
            sink.add("crash", sp, 1, "positive", im, f"jj_{sp}_{n:04d}", "justjuu/traffic-accident-cctv-object-detection (CC0)", dh)
            n += 1
            if n == k:
                break
    hn = RAW / "hiennguyen9874__traffic-accident-detection/data/train-00000-of-00001.parquet"
    negs = []
    f = pq.ParquetFile(hn)
    for rg in range(f.num_row_groups):
        for r in f.read_row_group(rg, columns=["image", "is_accident"]).to_pylist():
            if not r["is_accident"]:
                negs.append(r["image"]["bytes"])
    rng.shuffle(negs)
    cut = {"test": negs[: int(len(negs) * 0.3)], "val": negs[int(len(negs) * 0.3): int(len(negs) * 0.5)], "train": negs[int(len(negs) * 0.5):]}
    for sp, bl in cut.items():
        for i, b in enumerate(bl):
            sink.add("crash", sp, 0, "street", Image.open(io.BytesIO(b)), f"hn_{sp}_{i:04d}", "hiennguyen9874/traffic-accident-detection (non-accident, licence not stated)")

    # ---- flood: zinnia (label 0 = flood, 1 = non_flood), val parquet is the test split; train parquet split into train/val
    def flood_rows(pattern):
        for r in parquet_rows(pattern, ["image", "label", "category"]):
            yield r
    tr = list(flood_rows("zinnia*/data/train-*.parquet"))
    te = list(flood_rows("zinnia*/data/val-*.parquet"))
    hold = [r for r in tr if h(hashlib.md5(r["image"]["bytes"]).hexdigest()) % 5 == 0]
    trn_f = [r for r in tr if h(hashlib.md5(r["image"]["bytes"]).hexdigest()) % 5 != 0]
    fl_dh = np.array([dhash(Image.open(io.BytesIO(r["image"]["bytes"]))) for r in trn_f], dtype=np.uint64)
    GROUP = {"street_clear": "street", "river": "hard", "swimming_pool": "hard", "park_walkway": "hard"}

    def add_flood(rows, sp, check):
        pk, nk = CAP.get(sp, (1300, 10**6))
        pos = [r for r in rows if r["label"] == 0]
        neg = [r for r in rows if r["label"] == 1]
        n = 0
        for r in pick(pos, pk * 2):
            im = Image.open(io.BytesIO(r["image"]["bytes"]))
            dh = dhash(im)
            if check and near_dup_arr(fl_dh, dh):
                dropped[("flood", "positive")] += 1
                continue
            sink.add("flood", sp, 1, "positive", im, f"fl_{sp}_p{n:04d}", "zinnia82/flood-binary-hnm-benchmark", dh)
            n += 1
            if n == pk:
                break
        nv = sum(1 for x in sink.rows if x["task"] == "flood" and x["split"] == sp and x["label"] == 0)
        n = 0
        for r in pick(neg, nk * 2):
            if n >= max(0, nk - nv):
                break
            im = Image.open(io.BytesIO(r["image"]["bytes"]))
            dh = dhash(im)
            if check and near_dup_arr(fl_dh, dh):
                dropped[("flood", "negative")] += 1
                continue
            sink.add("flood", sp, 0, GROUP.get(r["category"], "background"), im, f"fl_{sp}_n{n:04d}", f"zinnia82 ({r['category']})", dh)
            n += 1

    add_flood(te, "test", True)
    add_flood(hold, "val", False)
    add_flood(trn_f, "train", False)

    # ---- manifest + stats
    for r in sink.rows:
        r["dhash"] = str(r["dhash"])
    with open(OUT / "manifest.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(sink.rows[0].keys()))
        w.writeheader()
        w.writerows(sink.rows)
    stats = defaultdict(dict)
    for r in sink.rows:
        k = f"{r['task']}/{r['split']}"
        stats[k][f"{'pos' if r['label'] else 'neg_' + r['group']}"] = stats[k].get(f"{'pos' if r['label'] else 'neg_' + r['group']}", 0) + 1
    dfire_counts = {s: {"images": len(recs[s]), "with_fire": sum(1 for _, i in recs[s] if 1 in i), "with_smoke": sum(1 for _, i in recs[s] if 0 in i),
                        "no_label": sum(1 for _, i in recs[s] if not i)} for s in recs}
    out = {"image_counts": stats, "dfire_official": dfire_counts, "dfire_train_used_for_yolo": len(trn), "dfire_holdout_val": len(val),
           "near_duplicates_removed_from_eval": {f"{k[0]}:{k[1]}": v for k, v in dropped.items()},
           "negative_clips_by_split": dict(Counter((it["split"], it["group"]) .__str__() for it in vneg)),
           "max_side_px": MAXSIDE}
    (ROOT / "reports/metrics").mkdir(parents=True, exist_ok=True)
    (ROOT / "reports/metrics/incident_data.json").write_text(json.dumps(out, indent=2))
    # YOLO dataset for D-Fire fine-tuning
    ydir = OUT / "dfire_yolo"
    for sp, pool in (("train", trn), ("val", val)):
        (ydir / "images" / sp).mkdir(parents=True, exist_ok=True)
        (ydir / "labels" / sp).mkdir(parents=True, exist_ok=True)
        for ip, _ in pool:
            dst = ydir / "images" / sp / ip.name
            if not dst.exists():
                dst.symlink_to(ip.resolve())
            lbl = dfire / "train" / "labels" / (ip.stem + ".txt")
            (ydir / "labels" / sp / (ip.stem + ".txt")).write_text(lbl.read_text())
    (ydir / "data.yaml").write_text(f"path: {ydir}\ntrain: images/train\nval: images/val\nnames:\n  0: smoke\n  1: fire\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
