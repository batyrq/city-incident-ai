"""Export the precision/recall curves (overall + per sequence, IoU 0.5) from the cached detections to reports/metrics/pr_curve.json.

  python -m eval.export_pr_curve      # uses data/mot/cache/dets_*.npz written by eval.eval_detection
Curves are thinned to <= 150 points each (evenly spaced along recall); AP50 is recomputed from the full curve.
"""
import json
from pathlib import Path

import numpy as np

from eval.eval_detection import CACHE, DEMO_CONF, match_sequence, merge, summarize
from eval.mot_common import SEQS


def thin(c, p, r, n=150):
    idx = np.unique(np.linspace(0, len(r) - 1, min(n, len(r))).astype(int))
    return [{"recall": float(r[i]), "precision": float(p[i]), "conf": float(c[i])} for i in idx]


def main() -> None:
    parts, out = {}, {"iou": 0.5, "demo_conf": DEMO_CONF, "curves": {}}
    for seq in SEQS:
        parts[seq] = match_sequence(seq, np.load(CACHE / f"dets_{seq}.npz")["dets"])
    for name, m in {**parts, "overall": merge(list(parts.values()))}.items():
        s = summarize(m); c, p, r = s["_curve"]
        k = int(np.flatnonzero(c >= DEMO_CONF)[-1]) if (c >= DEMO_CONF).any() else None
        out["curves"][name] = {"AP50": s["AP50"], "points": thin(c, p, r),
                               "demo_point": ({"recall": float(r[k]), "precision": float(p[k]), "conf": DEMO_CONF} if k is not None else None)}
    Path("reports/metrics/pr_curve.json").write_text(json.dumps(out))
    print("overall AP50", out["curves"]["overall"]["AP50"], "points", len(out["curves"]["overall"]["points"]))


if __name__ == "__main__":
    main()
