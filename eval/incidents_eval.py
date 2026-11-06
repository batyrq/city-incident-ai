"""Image-level comparison of three incident-detection approaches on the balanced sets built by eval/incidents_data.py.

  python -m eval.incidents_eval scores      # compute per-image scores (cached in data/incidents/scores/)
  python -m eval.incidents_eval metrics     # thresholds on val, metrics on test -> reports/metrics/incidents.json, confusion matrices
  python -m eval.incidents_eval latency     # per-image latency of each approach on this machine (run when nothing else uses the GPU)

A  zero-shot open-vocabulary detector (YOLO-World yolov8s-world), positive if any box of the class prompts >= threshold
B  zero-shot CLIP ViT-B/16 (text prompts, softmax mass on the incident prompts)
D  ready-made community weights from Hugging Face, no training by me (see PRETRAINED below)
C  trained: fire/smoke -> YOLO11n fine-tuned on D-Fire; crash/flood -> logistic regression on frozen CLIP embeddings
Thresholds are chosen on the validation split (max F1); the test split is only used to report.
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data/incidents"
SC = OUT / "scores"
TASKS = ["fire", "smoke", "flood", "crash"]
# fire/smoke prompts: best of the variants in eval/incidents_prompt_probe.py (by ROC-AUC on val); flood/crash prompts are a first guess
YOLO_W_PROMPTS = {"fire": ["a fire"], "smoke": ["smoke cloud"], "flood": ["flooded street", "flood water"],
                  "crash": ["car crash", "damaged car", "overturned car"]}
CLIP_PROMPTS = {
    "fire": (["a photo of a fire", "a photo of flames burning", "a photo of a burning building"],
             ["a photo of a normal street", "a photo of a city at sunset", "a photo of an ordinary outdoor scene", "a photo of car headlights at night"]),
    "smoke": (["a photo of smoke", "a photo of smoke rising from a fire", "a photo of a smoky sky"],
              ["a photo of a clear sky", "a photo of clouds", "a photo of a normal street", "a photo of fog or steam"]),
    "flood": (["a photo of a flooded street", "a photo of flood water covering a road", "a photo of a flood"],
              ["a photo of a normal dry street", "a photo of a river", "a photo of a swimming pool", "a photo of a wet road after rain"]),
    "crash": (["a photo of a car crash", "a photo of a traffic accident with damaged vehicles", "a photo of two cars that collided"],
              ["a photo of normal street traffic", "a photo of a busy road with cars", "a photo of a city street", "a photo of a parking lot"]),
}


PRETRAINED = {  # approach D: downloaded as-is from Hugging Face
    "fire": dict(repo="rabahdev/fire-smoke-yolov8n", file="best.pt", kind="yolo", cls=1, licence="AGPL-3.0", data="D-Fire (official train split 17,221 imgs = its train+val; its test = official D-Fire test 4,306)"),
    "smoke": dict(repo="rabahdev/fire-smoke-yolov8n", file="best.pt", kind="yolo", cls=0, licence="AGPL-3.0", data="D-Fire (as above)"),
    "crash": dict(repo="Enos-123/traffic-accident-detection-yolo11x", file="weights/epoch61.pt", kind="yolo", cls=0, licence="MIT", data="Roboflow 'Traffic Accident Detection' (hilmantm), surveillance frames, classes accident/vehicle"),
    "flood": dict(repo="prithivMLmods/Flood-Image-Detection", file=None, kind="siglip", cls=0, licence="Apache-2.0", data="TheNetherWatcher/DisasterClassification, SigLIP2-base-512 fine-tune"),
}


def manifest() -> list[dict]:
    return list(csv.DictReader(open(OUT / "manifest.csv")))


def rows_for(task: str, splits=("val", "test")) -> list[dict]:
    return [r for r in manifest() if r["task"] == task and r["split"] in splits]


def device() -> str:
    import torch
    return "mps" if torch.backends.mps.is_available() else "cpu"


# ------------------------------------------------------------------------------- scoring
def clip_model():
    import open_clip
    import torch
    m, _, pre = open_clip.create_model_and_transforms("ViT-B-16-quickgelu", pretrained="openai")
    m = m.to(device()).eval()
    return m, pre, open_clip.get_tokenizer("ViT-B-16-quickgelu"), torch


def clip_embed(paths: list[str], bs: int = 64) -> np.ndarray:
    m, pre, _, torch = clip_model()
    out = []
    with torch.no_grad():
        for i in range(0, len(paths), bs):
            x = torch.stack([pre(Image.open(ROOT / p).convert("RGB")) for p in paths[i:i + bs]]).to(device())
            f = m.encode_image(x)
            out.append((f / f.norm(dim=-1, keepdim=True)).cpu().numpy())
    return np.concatenate(out)


def embeddings(task: str, splits=("train", "val", "test")) -> dict[str, tuple[list[dict], np.ndarray]]:
    res = {}
    for sp in splits:
        cache = SC / f"emb_{task}_{sp}.npy"
        rs = rows_for(task, (sp,))
        if cache.exists() and len(np.load(cache)) == len(rs):
            e = np.load(cache)
        else:
            e = clip_embed([r["path"] for r in rs])
            np.save(cache, e)
        res[sp] = (rs, e)
    return res


def save_scores(name: str, task: str, rows: list[dict], s: np.ndarray) -> None:
    with open(SC / f"{name}_{task}.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["split", "label", "group", "path", "score"])
        for r, v in zip(rows, s):
            w.writerow([r["split"], r["label"], r["group"], r["path"], f"{float(v):.6f}"])


def clip_text_feats(task: str):
    m, _, tok, torch = clip_model()
    pos, neg = CLIP_PROMPTS[task]
    with torch.no_grad():
        t = m.encode_text(tok(pos + neg).to(device()))
        t = (t / t.norm(dim=-1, keepdim=True)).cpu().numpy()
    return t[: len(pos)], t[len(pos):]


def run_scores(which: set[str]) -> None:
    SC.mkdir(parents=True, exist_ok=True)
    from sklearn.linear_model import LogisticRegression
    for task in TASKS:
        if "B" in which or "Cprobe" in which:
            emb = embeddings(task)
        if "B" in which:
            tp, tn = clip_text_feats(task)
            T = np.concatenate([tp, tn])
            rows = emb["val"][0] + emb["test"][0]
            e = np.concatenate([emb["val"][1], emb["test"][1]])
            lg = 100.0 * e @ T.T
            p = np.exp(lg - lg.max(1, keepdims=True))
            p /= p.sum(1, keepdims=True)
            save_scores("B_clip_zeroshot", task, rows, p[:, : len(tp)].sum(1))
        if "Cprobe" in which and task in ("flood", "crash"):
            trr, tre = emb["train"]
            y = np.array([int(r["label"]) for r in trr])
            clf = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced").fit(tre, y)
            rows = emb["val"][0] + emb["test"][0]
            e = np.concatenate([emb["val"][1], emb["test"][1]])
            save_scores("C_clip_probe", task, rows, clf.predict_proba(e)[:, 1])
            (ROOT / "models").mkdir(exist_ok=True)
            np.savez(ROOT / f"models/incident_probe_{task}.npz", coef=clf.coef_[0].astype(np.float32), intercept=np.float32(clf.intercept_[0]),
                     clip="ViT-B-16-quickgelu/openai", n_train=len(y))
    if "A" in which:
        from ultralytics import YOLOWorld
        mdl = YOLOWorld(str(ROOT / "yolov8s-world.pt"))
        for task in TASKS:
            mdl.set_classes(YOLO_W_PROMPTS[task])
            rows = rows_for(task)
            s = np.zeros(len(rows))
            for i in range(0, len(rows), 16):
                for j, r in enumerate(mdl.predict([str(ROOT / x["path"]) for x in rows[i:i + 16]], conf=0.005, imgsz=640, device=device(), verbose=False)):
                    s[i + j] = float(r.boxes.conf.max()) if len(r.boxes) else 0.0
            save_scores("A_yolo_world", task, rows, s)
            print("A", task, flush=True)
    if "D" in which:
        from huggingface_hub import hf_hub_download
        from ultralytics import YOLO
        for task, spec in PRETRAINED.items():
            rows = rows_for(task)
            s = np.zeros(len(rows))
            if spec["kind"] == "yolo":
                mdl = YOLO(hf_hub_download(spec["repo"], spec["file"], local_dir=str(ROOT / "models/pretrained" / spec["repo"].replace("/", "_").replace("_traffic-accident-detection", "_accident"))))
                for i in range(0, len(rows), 16):
                    for j, r in enumerate(mdl.predict([str(ROOT / x["path"]) for x in rows[i:i + 16]], conf=0.005, imgsz=640, device=device(), verbose=False)):
                        c = r.boxes.cls.cpu().numpy() if len(r.boxes) else np.array([])
                        cf = r.boxes.conf.cpu().numpy() if len(r.boxes) else np.array([])
                        s[i + j] = float(cf[c == spec["cls"]].max()) if (c == spec["cls"]).any() else 0.0
            else:
                import torch
                from transformers import AutoImageProcessor, SiglipForImageClassification
                proc = AutoImageProcessor.from_pretrained(spec["repo"])
                mdl = SiglipForImageClassification.from_pretrained(spec["repo"]).to(device()).eval()
                for i in range(0, len(rows), 16):
                    ims = [Image.open(ROOT / x["path"]).convert("RGB") for x in rows[i:i + 16]]
                    with torch.no_grad():
                        pr = mdl(**{k: v.to(device()) for k, v in proc(images=ims, return_tensors="pt").items()}).logits.softmax(-1)[:, spec["cls"]]
                    s[i:i + len(ims)] = pr.cpu().numpy()
            save_scores("D_pretrained", task, rows, s)
            print("D", task, flush=True)
    if "Cyolo" in which:
        from ultralytics import YOLO
        w = json.loads((ROOT / "reports/metrics/incident_yolo_train.json").read_text())["weights"]
        mdl = YOLO(w)
        for task, cid in (("fire", 1), ("smoke", 0)):
            rows = rows_for(task)
            s = np.zeros(len(rows))
            for i in range(0, len(rows), 16):
                for j, r in enumerate(mdl.predict([str(ROOT / x["path"]) for x in rows[i:i + 16]], conf=0.005, imgsz=640, device=device(), verbose=False)):
                    c = r.boxes.cls.cpu().numpy() if len(r.boxes) else np.array([])
                    cf = r.boxes.conf.cpu().numpy() if len(r.boxes) else np.array([])
                    s[i + j] = float(cf[c == cid].max()) if (c == cid).any() else 0.0
            save_scores("C_yolo11n_dfire", task, rows, s)
            print("Cyolo", task, flush=True)


# ------------------------------------------------------------------------------- metrics
def load_scores(name: str, task: str):
    p = SC / f"{name}_{task}.csv"
    if not p.exists():
        return None
    rs = list(csv.DictReader(open(p)))
    return rs


def best_threshold(y: np.ndarray, s: np.ndarray) -> float:
    cands = np.unique(np.concatenate([s, [0.0]]))
    best, bt = -1.0, 0.5
    for t in cands:
        pred = s >= t
        tp = (pred & (y == 1)).sum(); fp = (pred & (y == 0)).sum(); fn = ((~pred) & (y == 1)).sum()
        f1 = 2 * tp / max(1, 2 * tp + fp + fn)
        if f1 > best + 1e-12:
            best, bt = f1, float(t)
    return bt


def metrics_at(rows: list[dict], thr: float) -> dict:
    from sklearn.metrics import roc_auc_score
    y = np.array([int(r["label"]) for r in rows]); s = np.array([float(r["score"]) for r in rows]); g = np.array([r["group"] for r in rows])
    pred = s >= thr
    tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum()); fn = int(((~pred) & (y == 1)).sum()); tn = int(((~pred) & (y == 0)).sum())
    pr = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    out = {"threshold": round(thr, 4), "tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": round(pr, 4), "recall": round(rc, 4),
           "f1": round(2 * pr * rc / (pr + rc), 4) if pr + rc else 0.0, "roc_auc": round(float(roc_auc_score(y, s)), 4),
           "fpr_overall": round(fp / max(1, fp + tn), 4)}
    for grp in ("street", "hard", "background"):
        m = (y == 0) & (g == grp)
        out[f"fpr_{grp}"] = round(float((pred & m).sum() / m.sum()), 4) if m.sum() else None
        out[f"n_neg_{grp}"] = int(m.sum())
    out["n_pos"] = int((y == 1).sum())
    return out


def run_metrics() -> None:
    names = {"A": "A_yolo_world", "B": "B_clip_zeroshot"}
    res: dict = {}
    for task in TASKS:
        res[task] = {}
        cands = {"A": "A_yolo_world", "B": "B_clip_zeroshot", "D": "D_pretrained", "C": "C_yolo11n_dfire" if task in ("fire", "smoke") else "C_clip_probe"}
        for ap, name in cands.items():
            rows = load_scores(name, task)
            if rows is None:
                continue
            val = [r for r in rows if r["split"] == "val"]; test = [r for r in rows if r["split"] == "test"]
            thr = best_threshold(np.array([int(r["label"]) for r in val]), np.array([float(r["score"]) for r in val]))
            m = metrics_at(test, thr)
            m["val_f1_at_threshold"] = metrics_at(val, thr)["f1"]
            m["model"] = name
            res[task][ap] = m
    (ROOT / "reports/metrics/incidents.json").write_text(json.dumps(res, indent=2))
    write_detector_config(res)
    # confusion matrices
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = ROOT / "docs/incident_confusion"
    out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(len(TASKS), 4, figsize=(12, 11))
    for i, task in enumerate(TASKS):
        for j, ap in enumerate("ABDC"):
            a = ax[i][j]; m = res[task].get(ap)
            if not m:
                a.axis("off"); continue
            cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
            a.imshow(cm, cmap="Blues")
            for (r, c), v in np.ndenumerate(cm):
                a.text(c, r, str(v), ha="center", va="center", color="white" if v > cm.max() / 2 else "black")
            a.set_xticks([0, 1], ["pred normal", "pred " + task], fontsize=7); a.set_yticks([0, 1], ["normal", task], fontsize=7)
            a.set_title(f"{task} · {ap}", fontsize=9)
    fig.tight_layout(); fig.savefig(out / "confusion_matrices.png", dpi=120)
    print(json.dumps({t: {a: {k: v[k] for k in ("threshold", "precision", "recall", "f1", "roc_auc", "fpr_street", "fpr_hard", "fpr_background")} for a, v in d.items()} for t, d in res.items()}, indent=1))


NO_TRAINING = {"A", "B", "D"}
MARGIN = 0.03  # prefer a no-training approach when its validation F1 is within this of the best


def write_detector_config(res: dict) -> None:
    """Default per class, from validation F1 only. Candidates wired into the pipeline: B (CLIP zero-shot) and C (trained).
    D (ready community weights) is evaluated but not wired in. D on fire/smoke is excluded from the choice because the validation
    holdout is carved from D-Fire train, which D was trained on (its validation score and threshold would be optimistic).
    Rule: take the best candidate; if a no-training candidate is within MARGIN val F1 of it, take the best no-training one."""
    import shutil
    classes = {}
    for task in TASKS:
        cand = {ap: m for ap, m in res[task].items() if ap in ("B", "C")}
        best = max(cand, key=lambda a: cand[a]["val_f1_at_threshold"])
        pool = {a: m for a, m in cand.items() if a in NO_TRAINING and cand[best]["val_f1_at_threshold"] - m["val_f1_at_threshold"] <= MARGIN}
        ap = max(pool, key=lambda a: pool[a]["val_f1_at_threshold"]) if pool else best
        m = cand[ap]
        if ap == "C" and task in ("fire", "smoke"):
            spec = {"approach": "yolo", "yolo_class_id": 1 if task == "fire" else 0}
        elif ap == "C":
            spec = {"approach": "probe", "probe_file": f"models/incident_probe_{task}.npz"}
        else:
            spec = {"approach": "clip_zeroshot"}
        classes[task] = {**spec, "letter": ap, "trained_by_me": ap not in NO_TRAINING, "threshold": m["threshold"], "val_f1": m["val_f1_at_threshold"], "test_f1": m["f1"],
                         "rule": f"best val F1, but a no-training approach within {MARGIN} F1 wins"}
    w = json.loads((ROOT / "reports/metrics/incident_yolo_train.json").read_text())["weights"]
    (ROOT / "models").mkdir(exist_ok=True)
    shutil.copy(w, ROOT / "models/incident_dfire_yolo11n.pt")
    cfg = {"yolo_weights": "models/incident_dfire_yolo11n.pt", "clip_prompts": {t: CLIP_PROMPTS[t] for t in TASKS}, "classes": classes}
    (ROOT / "configs/incident_detectors.json").write_text(json.dumps(cfg, indent=2))


# ------------------------------------------------------------------------------- latency
def run_latency() -> None:
    import torch
    out = {}
    dev = device()
    for task in TASKS:
        rows = [r for r in rows_for(task, ("test",))][:60]
        paths = [str(ROOT / r["path"]) for r in rows]
        out[task] = {}

        def timeit(fn, n=len(paths)):
            for p in paths[:5]:
                fn(p)
            if dev == "mps":
                torch.mps.synchronize()
            t = time.perf_counter()
            for p in paths[5:]:
                fn(p)
            if dev == "mps":
                torch.mps.synchronize()
            return round((time.perf_counter() - t) * 1000 / (n - 5), 1)
        from ultralytics import YOLO, YOLOWorld
        mw = YOLOWorld(str(ROOT / "yolov8s-world.pt")); mw.set_classes(YOLO_W_PROMPTS[task])
        out[task]["A"] = timeit(lambda p: mw.predict(p, conf=0.005, imgsz=640, device=dev, verbose=False))
        m, pre, tok, _ = clip_model()
        tp, tn = clip_text_feats(task)
        T = torch.tensor(np.concatenate([tp, tn])).to(dev)

        def clipfn(p):
            with torch.no_grad():
                f = m.encode_image(pre(Image.open(p).convert("RGB")).unsqueeze(0).to(dev))
                f = f / f.norm(dim=-1, keepdim=True)
                return (100 * f @ T.T).softmax(-1).cpu()
        out[task]["B"] = timeit(clipfn)
        out[task]["C"] = out[task]["B"] if task in ("flood", "crash") else None  # linear probe adds a negligible dot product
        if task in ("fire", "smoke"):
            w = json.loads((ROOT / "reports/metrics/incident_yolo_train.json").read_text())["weights"]
            my = YOLO(w)
            out[task]["C"] = timeit(lambda p: my.predict(p, conf=0.005, imgsz=640, device=dev, verbose=False))
        spec = PRETRAINED[task]
        if spec["kind"] == "yolo":
            from huggingface_hub import hf_hub_download
            md = YOLO(hf_hub_download(spec["repo"], spec["file"], local_dir=str(ROOT / "models/pretrained" / spec["repo"].replace("/", "_").replace("_traffic-accident-detection", "_accident"))))
            out[task]["D"] = timeit(lambda p: md.predict(p, conf=0.005, imgsz=640, device=dev, verbose=False))
        else:
            from transformers import AutoImageProcessor, SiglipForImageClassification
            proc = AutoImageProcessor.from_pretrained(spec["repo"]); ms = SiglipForImageClassification.from_pretrained(spec["repo"]).to(dev).eval()

            def sig(p):
                with torch.no_grad():
                    return ms(**{k: v.to(dev) for k, v in proc(images=Image.open(p).convert("RGB"), return_tensors="pt").items()}).logits.softmax(-1).cpu()
            out[task]["D"] = timeit(sig)
        print(task, out[task], flush=True)
    (ROOT / "reports/metrics/incident_latency.json").write_text(json.dumps({"device": dev, "unit": "ms per image, batch 1, incl. image load + preprocessing, images <= 640 px", "ms": out}, indent=2))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "scores":
        run_scores(set(sys.argv[2:]) or {"A", "B", "Cprobe", "Cyolo"})
    elif cmd == "metrics":
        run_metrics()
    elif cmd == "latency":
        run_latency()
