"""Build the static results site: site/index.html (+ site/img/*) from reports/metrics/*.json, docs/ERROR_ANALYSIS.md and docs/failures/.

  python scripts/build_site.py        # or: make site

Every number on the page is read from a JSON file here (or computed from JSON values); the prose is static text without numbers
except the 'why' paragraphs, which are taken verbatim from docs/ERROR_ANALYSIS.md. Videos (site/videos/) and the local Chart.js copy
(site/vendor/) are committed assets; this script only checks that they exist.
"""
from __future__ import annotations

import html
import json
import re
import shutil
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
M = ROOT / "reports" / "metrics"
SITE = ROOT / "site"
REPO = "https://github.com/batyrq/city-incident-ai"


def jl(name: str):
    return json.loads((M / name).read_text())


def pct(x: float, d: int = 1) -> str:
    return f"{x * 100:.{d}f}%"


def md_inline(text: str) -> str:
    t = html.escape(text.strip())
    t = re.sub(r"`([^`]+)`", r"<code>\1</code>", t)
    t = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", t)
    t = re.sub(r"(?<![\w*])\*([^*\n]+)\*(?![\w*])", r"<em>\1</em>", t)
    return t.replace("\n", " ")


def error_sections() -> dict[str, dict[str, str]]:
    md = (ROOT / "docs" / "ERROR_ANALYSIS.md").read_text()
    out = {}
    for m in re.finditer(r"^## ([a-e])\) (.+?)\n(.*?)(?=\n---|\Z)", md, re.S | re.M):
        key, title, body = m.group(1), m.group(2), m.group(3)
        why = re.search(r"\*\*Why \(hypothesis\):\*\*(.*?)(?=\n\*\*Fix|\Z)", body, re.S)
        fix = re.search(r"\*\*Fix[^*]*\*\*(.*?)(?=\n\n|\Z)", body, re.S)
        out[key] = {"title": title.strip(), "why": md_inline(why.group(1)) if why else "", "fix": md_inline(fix.group(1)) if fix else ""}
    return out


def main() -> None:
    det, trk, abl, spd = jl("detection.json"), jl("tracking.json"), jl("tracking_conf_ablation.json"), jl("speed.json")
    err, i8, prc = jl("error_analysis.json"), jl("int8_debug.json"), jl("pr_curve.json")
    cfg = yaml.safe_load((ROOT / "configs" / "infer_pretrained_yolo11s.yaml").read_text())
    env, V = spd["environment"], spd["variants"]
    ov = det["results"]["overall"]
    bt, bs = trk["trackers"]["bytetrack"]["scores"]["OVERALL"], trk["trackers"]["botsort"]["scores"]["OVERALL"]
    A = {k: v["scores"]["OVERALL"] for k, v in abl["configs"].items()}
    a0, a1, a2 = A["a_conf0.35_default"], A["b_conf0.10_default"], A["c_conf0.10_buffer60"]
    mps, cpu = V["fp32_pt_mps"], V["fp32_onnx_cpu"]
    conf_track, conf_disp = cfg["tracking"]["detector_conf"], cfg["runtime"]["score_threshold"]
    for need in ("videos/demo_clip61.mp4", "videos/demo_pose_clip4332.mp4", "vendor/chart.umd.min.js"):
        if not (SITE / need).exists():
            sys.exit(f"missing site asset: site/{need}")

    # ---- images
    (SITE / "img").mkdir(exist_ok=True)
    import cv2
    for f in (ROOT / "docs" / "failures").glob("*"):
        if f.suffix in (".jpg", ".png"):
            shutil.copy(f, SITE / "img" / f.name)
    shutil.copy(ROOT / "docs" / "demo_pose_runner.gif", SITE / "img" / "demo_pose_runner.gif")
    pair = cv2.imread(str(ROOT / "docs" / "failures" / "a_small_people_640_vs_960.jpg"))
    half = pair.shape[1] // 2
    cv2.imwrite(str(SITE / "img" / "a_640.jpg"), pair[:, :half], [cv2.IMWRITE_JPEG_QUALITY, 90])
    cv2.imwrite(str(SITE / "img" / "a_960.jpg"), pair[:, half:], [cv2.IMWRITE_JPEG_QUALITY, 90])

    # ---- chart data (read by the page's JS)
    short = {"fp32_pt_cpu": "PyTorch CPU", "fp32_pt_mps": "PyTorch MPS", "fp32_onnx_cpu": "ONNX CPU (repo)", "fp32_onnx_coreml": "ONNX CoreML",
             "int8_onnx_cpu": "INT8 static", "int8dyn_onnx_cpu": "INT8 dynamic", "n_onnx_cpu": "yolo11n", "s960_onnx_cpu": "yolo11s @960"}
    data = {
        "pr": prc,
        "recall_vis": {k: v["recall"] for k, v in ov["recall_by_visibility"].items()},
        "recall_vis_n": {k: v["count"] for k, v in ov["recall_by_visibility"].items()},
        "recall_h": {k: v["recall"] for k, v in ov["recall_by_height_px"].items()},
        "recall_h_n": {k: v["count"] for k, v in ov["recall_by_height_px"].items()},
        "trackers": {"ByteTrack": {k: bt[k] for k in ("mota", "idf1", "num_switches")}, "BoT-SORT": {k: bs[k] for k in ("mota", "idf1", "num_switches")}},
        "ablation": {"labels": ["conf 0.35", "conf 0.10", "conf 0.10 + buffer 60"],
                     "mota": [x["mota"] for x in (a0, a1, a2)], "idf1": [x["idf1"] for x in (a0, a1, a2)], "idsw": [x["num_switches"] for x in (a0, a1, a2)]},
        "speed": [{"key": k, "name": short[k], "label": v["label"], "x": v["latency"]["model_only"]["p50_ms"], "y": v["accuracy"]["AP50"],
                   "fps": v["pipeline"]["fps"], "mb": v["size_mb"], "imgsz": v["imgsz"]} for k, v in V.items() if v.get("status") == "ok"],
    }
    es = error_sections()
    ea = err["a_small_people"]["all_eval_frames_hits_over_gt"]
    ec = err["c_id_switch"]["chosen"]
    ed = err["d_night_clip"]
    ee = err["e_collision_rule"]

    def stat(big: str, label: str, sub: str) -> str:
        return f'<div class="stat"><div class="big">{big}</div><div class="lab">{label}</div><div class="sub">{sub}</div></div>'

    stats = "".join([
        stat(f"{ov['AP50']:.3f}", "Person AP50", f"MOT17-02/04/09 · AP50:95 {ov['AP50_95']:.3f}"),
        stat(f"{pct(ov['at_conf_0.35']['precision'])} / {pct(ov['at_conf_0.35']['recall'])}", "Precision / recall", "person, conf 0.35, IoU 0.5"),
        stat(f"{pct(a0['mota'])} <span class='arrow'>→</span> {pct(a1['mota'])}", "MOTA after the threshold fix", f"tracker fed conf {conf_track} instead of {conf_disp}"),
        stat(f"{mps['latency']['model_only']['p50_ms']:.1f} ms", "Model latency on Apple GPU (MPS)", f"p50, {env['chip']} · pipeline {mps['pipeline']['fps']:.0f} FPS"),
        stat(f"{cpu['pipeline']['fps']:.0f} FPS", "Full pipeline on CPU", "ONNX, decode + track + rules + draw + write"),
        stat(f"{bs['tracker_ms_per_frame_mean']:.1f} vs {bt['tracker_ms_per_frame_mean']:.1f} ms", "Tracker time: BoT-SORT vs ByteTrack", "per frame, detector excluded"),
    ])

    # ---- tables
    def row(cells, cls=""):
        return f"<tr{' class=' + cls if cls else ''}>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>"

    trk_table = "".join(row([n, pct(s["mota"]), pct(s["idf1"]), s["num_switches"], s["num_false_positives"], s["num_misses"], s["mostly_tracked"], s["mostly_lost"],
                             f"{s['pred_identities']} / {s['gt_identities']}", f"{s['tracker_ms_per_frame_mean']:.2f}"]) for n, s in (("ByteTrack", bt), ("BoT-SORT (no ReID)", bs)))
    abl_names = {"a_conf0.35_default": "A · detector conf 0.35 (before)", "b_conf0.10_default": "B · detector conf 0.10 (demo default)", "c_conf0.10_buffer60": "C · conf 0.10 + track_buffer 60"}
    abl_table = "".join(row([abl_names[k], pct(s["mota"]), pct(s["idf1"]), s["num_switches"], s["num_false_positives"], s["num_misses"],
                             f"{s['pred_identities']} / {s['gt_identities']}",
                             pct(abl["configs"][k]["recall_by_visibility_from_tracker_output"]["<0.3"]["recall"])],
                            "hl" if k.startswith("b_") else "") for k, s in A.items())
    spd_table = "".join(row([v["label"], v["imgsz"], f"{v['size_mb']}",
                             f"{v['latency']['model_only']['mean_ms']:.1f} / {v['latency']['model_only']['p50_ms']:.1f} / {v['latency']['model_only']['p95_ms']:.1f}",
                             f"{v['pipeline']['fps']:.1f}", f"{v['accuracy']['AP50']:.3f}",
                             pct(v["accuracy"]["recall_by_height_px"]["<50"]["recall"]), pct(v["accuracy"]["recall_by_height_px"]["50-100"]["recall"]),
                             pct(v["accuracy"]["recall_by_height_px"][">100"]["recall"])], "hl" if k == "fp32_pt_mps" else "")
                        for k, v in V.items() if v.get("status") == "ok")

    # ---- error cards
    def card(key, hero, sub, images, extra=""):
        s = es[key]
        return (f'<article class="case" id="case-{key}"><div class="case-head"><span class="tag">{key.upper()}</span><h3>{html.escape(s["title"])}</h3></div>'
                f'<div class="hero-num">{hero}<span>{sub}</span></div>{images}{extra}'
                f'<p class="why"><b>Why (hypothesis)</b> {s["why"]}</p><p class="fix"><b>Fix</b> {s["fix"]}</p></article>')

    slider = ('<div class="cmp" id="cmp"><img src="img/a_960.jpg" alt="imgsz 960"><div class="cmp-top"><img src="img/a_640.jpg" alt="imgsz 640"></div>'
              '<input type="range" min="0" max="100" value="50" aria-label="compare 640 and 960"><div class="cmp-line"></div>'
              '<span class="cmp-l">640</span><span class="cmp-r">960</span></div>')
    img = lambda f, alt: f'<a href="img/{f}" target="_blank"><img class="shot" src="img/{f}" alt="{alt}"></a>'
    cases = "".join([
        card("a", f"{ea['640']['lt50']['hit']}/{ea['640']['lt50']['gt']} → {ea['960']['lt50']['hit']}/{ea['960']['lt50']['gt']}", "people under 50 px found, 640 → 960",
             slider + img("a2_tiny_people_640_vs_960.jpg", "tiny people at 640 and 960"),
             f'<p class="facts">50–100 px: {pct(ea["640"]["50-100"]["recall"])} → {pct(ea["960"]["50-100"]["recall"])} · '
             f'latency p50 {err["a_small_people"]["benchmark_latency_p50_ms"]["fp32_onnx_cpu"]:.1f} → {err["a_small_people"]["benchmark_latency_p50_ms"]["s960_onnx_cpu"]:.1f} ms</p>'),
        card("b", pct(ov["recall_by_visibility"]["<0.3"]["recall"]), "recall when visibility &lt; 0.3 (&gt; 0.7: " + pct(ov["recall_by_visibility"][">0.7"]["recall"]) + ")",
             img("b_occlusion_crowd.jpg", "crowd occlusion"),
             f'<p class="facts">worst eval frame: {err["b_occlusion"]["this_frame"]["gt_vis_lt_0.3_hit"]} of {err["b_occlusion"]["this_frame"]["gt_vis_lt_0.3"]} heavily occluded people found</p>'),
        card("c", f"GT {ec['gt_id']}: T{ec['prev_hyp']} → T{ec['new_hyp']}", f"{err['c_id_switch']['total_switches_recomputed']} ID switches in total, this one at frame {ec['frame']}",
             img("c_id_switch.jpg", "ID switch"),
             f'<p class="facts">visibility at the switch {ec["vis_at_switch"]:.2f} · overlap with another person IoU {ec["max_overlap_with_other_gt"]:.2f} · {ec["frames_matched_before_with_prev"]} frames tracked before, {ec["frames_matched_after_with_new"]} after</p>'),
        card("d", f"{ed['tracks_shorter_than_15_frames']} of {ed['tracks_total']}", f"track IDs live under 15 frames (clip 4332, {ed['frames_in_clip']} frames)",
             img("d2_short_tracks.jpg", "short-lived tracks") + img("d3_umbrellas.jpg", "umbrellas"),
             f'<p class="facts">only {ed["tracks_with_more_than_one_class"]} tracks change class · {ed["umbrella_unfiltered_detector"]["umbrella_boxes"]} umbrella boxes in {ed["umbrella_unfiltered_detector"]["sampled_frames"]} sampled frames (not in the class filter)</p>'),
        card("e", f"{ee['clips']['4332']['frames_with_vehicle_pair_iou_ge_thr']} frames", f"with vehicle boxes at IoU ≥ {ee['overlap_iou_threshold']} in clip 4332 (clip 61: {ee['clips']['61']['frames_with_vehicle_pair_iou_ge_thr']}); none inspected was a collision",
             img("e_collision_iou.jpg", "IoU collision candidates"),
             '<p class="facts">rule is off in the default demo config</p>'),
    ])

    fp, hp, fl = i8["outputs"]["fp32"], i8["outputs"]["int8_head_fp32"], i8["outputs"]["int8_full_graph"]
    gait = Path(ROOT / "docs" / "demo_pose_runner.gif").exists()
    page = TEMPLATE
    repl = {
        "__STATS__": stats, "__TRK_TABLE__": trk_table, "__ABL_TABLE__": abl_table, "__SPD_TABLE__": spd_table, "__CASES__": cases,
        "__DATA__": json.dumps(data), "__REPO__": REPO,
        "__ENV__": html.escape(f"{env['chip']} ({env['cpu_cores']} cores), {env['ram_gb']:.0f} GB, macOS {env['macos']} · Python {env['python']} · torch {env['torch']} · onnxruntime {env['onnxruntime']} · ultralytics {env['ultralytics']}"),
        "__CONF_TRACK__": str(conf_track), "__CONF_DISP__": str(conf_disp),
        "__PIPE_DET_T__": cfg["model_type"].replace("yolo", "YOLO") + " detection", "__PIPE_DET__": "ONNX (CPU) or PyTorch (MPS)",
        "__PIPE_TRK__": f"tracker sees conf ≥ {conf_track}", "__PIPE_EVT__": f"stopped vehicle, crowding · shown ≥ {conf_disp}",
        "__ABL_FN__": f"{a0['num_misses'] - a1['num_misses']:,}", "__ABL_FP__": f"{a1['num_false_positives'] - a0['num_false_positives']:,}", "__ABL_SW__": str(a1["num_switches"] - a0["num_switches"]),
        "__ABL_MOTA__": f"{(a1['mota'] - a0['mota']) * 100:.1f}", "__ABL_IDF1__": f"{(a1['idf1'] - a0['idf1']) * 100:.1f}",
        "__ABL_BUF_SW__": f"{a2['num_switches']} vs {a1['num_switches']}",
        "__I8_FULL_MAX__": f"{fl['class_score_max']:.3f}", "__I8_FP_MAX__": f"{fp['class_score_max']:.3f}", "__I8_HEAD_MAX__": f"{hp['class_score_max']:.3f}",
        "__I8_FULL_AP__": f"{fl['AP50']:.3f}", "__I8_FP_AP__": f"{fp['AP50']:.3f}", "__I8_HEAD_AP__": f"{hp['AP50']:.3f}", "__I8_DISTINCT__": str(fl["distinct_class_score_values"]),
        "__LIM_TINY__": f"{ea['640']['lt50']['hit']} of {ea['640']['lt50']['gt']}", "__LIM_OCC__": pct(ov["recall_by_visibility"]["<0.3"]["recall"]),
        "__LIM_FRAMES__": str(spd["variants"]["fp32_pt_cpu"]["accuracy"]["frames_evaluated"]),
    }
    for k, v in repl.items():
        page = page.replace(k, v)
    left = re.findall(r"__[A-Z0-9_]+__", page)
    if left:
        sys.exit(f"unfilled placeholders: {sorted(set(left))}")
    (SITE / "index.html").write_text(page)
    print("wrote site/index.html", len(page) // 1024, "KB")


TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>city-incident-ai · results</title>
<meta name="description" content="Video analytics for city cameras: YOLO11 detection, ByteTrack tracking, event rules. Measured results.">
<style>
:root{--bg:#fff;--soft:#f6f8fa;--panel:#fff;--line:#e1e6ec;--text:#1f2328;--muted:#586069;--accent:#0f766e;--accent-soft:#e6f4f1;--shadow:0 1px 2px rgba(16,24,40,.05),0 2px 10px rgba(16,24,40,.06)}
*{box-sizing:border-box}html{scroll-behavior:smooth}
body{margin:0;background:var(--bg);color:var(--text);font:16px/1.6 -apple-system,BlinkMacSystemFont,"SF Pro Text","Inter","Segoe UI",Roboto,sans-serif;-webkit-font-smoothing:antialiased;overflow-x:hidden}
a{color:var(--accent);text-decoration:none}a:hover{text-decoration:underline}
code{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.88em;background:#eef1f4;padding:.1em .35em;border-radius:4px;overflow-wrap:anywhere}
.wrap{max-width:1120px;margin:0 auto;padding:0 24px;min-width:0}
nav{position:sticky;top:0;z-index:20;background:rgba(255,255,255,.92);backdrop-filter:blur(10px);border-bottom:1px solid var(--line)}
nav .wrap{display:flex;gap:22px;align-items:center;height:52px;overflow-x:auto;white-space:nowrap;scrollbar-width:none}nav .wrap::-webkit-scrollbar{display:none}
nav b{color:var(--text);margin-right:8px}nav a{color:var(--muted);font-size:14px}nav a:hover{color:var(--text);text-decoration:none}
section{padding:72px 0 8px}h1{font-size:clamp(34px,6vw,60px);line-height:1.05;margin:0 0 16px;letter-spacing:-.03em}
h2{font-size:clamp(26px,3.6vw,36px);letter-spacing:-.02em;margin:0 0 8px;overflow-wrap:anywhere}h2 small{display:block;font-size:13px;letter-spacing:.14em;text-transform:uppercase;color:var(--accent);font-weight:600;margin-bottom:10px}
h3{margin:0;font-size:19px}.lead{color:var(--muted);font-size:19px;max-width:760px;margin:0 0 28px}p{margin:.6em 0}.muted{color:var(--muted)}
.hero{padding:56px 0 8px}.hero .wrap{display:grid;gap:28px}
.tog{display:inline-flex;flex-wrap:wrap;border:1px solid var(--line);border-radius:999px;padding:4px;background:var(--soft);margin-bottom:12px;max-width:100%}
.tog button{background:none;border:0;color:var(--muted);padding:8px 16px;border-radius:999px;cursor:pointer;font:inherit;font-size:14px}.tog button.on{background:var(--accent);color:#fff;font-weight:600}
video{width:100%;border-radius:14px;border:1px solid var(--line);background:#000;display:block;box-shadow:var(--shadow)}
.stats{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}@media(max-width:900px){.stats{grid-template-columns:repeat(2,minmax(0,1fr))}}@media(max-width:560px){.stats{grid-template-columns:minmax(0,1fr)}}
.stat,.box,.case{background:var(--panel);border:1px solid var(--line);border-radius:14px;box-shadow:var(--shadow);min-width:0}
.stat{padding:22px}.stat .big{font-size:clamp(24px,2.6vw,32px);font-weight:700;letter-spacing:-.02em;color:var(--text);line-height:1.15;overflow-wrap:anywhere}
.stat .arrow{color:var(--accent)}.stat .lab{margin-top:6px;font-weight:600}.stat .sub{color:var(--muted);font-size:13.5px;margin-top:2px}
.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(380px,100%),1fr));gap:16px}.box{padding:18px}
.box h4{margin:0 0 4px;font-size:15px}.box .cap{color:var(--muted);font-size:13px;margin:0 0 10px}.chart{position:relative;height:320px}.chart.tall{height:440px}
table{width:100%;border-collapse:collapse;font-size:14px;margin-top:14px}th,td{padding:10px 12px;text-align:right;border-bottom:1px solid var(--line);white-space:nowrap}
th:first-child,td:first-child{text-align:left;white-space:normal;min-width:150px}th{color:var(--muted);font-weight:500;font-size:12.5px}tr.hl td{background:var(--accent-soft)}tr.hl td:first-child{color:var(--accent);font-weight:600}
.scroll{overflow-x:auto}.swipe{display:none;font-size:12.5px;color:var(--muted);margin:12px 0 4px}@media(max-width:700px){.swipe{display:block}.chart-scroll{overflow-x:auto}.chart-scroll .chart{min-width:720px}}.note{background:var(--accent-soft);border-left:3px solid var(--accent);padding:14px 18px;border-radius:8px;margin:18px 0;color:#24292f}
.flow{display:grid;grid-template-columns:minmax(0,1fr) 34px minmax(0,1fr) 34px minmax(0,1fr) 34px minmax(0,1fr) 34px minmax(0,1fr);align-items:stretch}
.node{border:1px solid var(--line);border-radius:12px;background:var(--soft);padding:14px 12px;text-align:center;min-height:100px;display:flex;flex-direction:column;justify-content:center;gap:4px;min-width:0}
.node b{font-size:16px;line-height:1.25}.node span{font-size:13px;color:var(--muted);line-height:1.35;overflow-wrap:break-word;text-wrap:balance}
.arr{display:flex;align-items:center;justify-content:center;color:var(--accent)}.arr svg{width:22px;height:22px}
@media(max-width:1100px) and (min-width:901px){.flow{grid-template-columns:minmax(0,1fr) 24px minmax(0,1fr) 24px minmax(0,1fr) 24px minmax(0,1fr) 24px minmax(0,1fr)}.node{padding:14px 8px}.arr svg{width:18px;height:18px}}
@media(max-width:900px){.flow{grid-template-columns:minmax(0,1fr)}.node{min-height:84px}.arr{height:34px}.arr svg{transform:rotate(90deg)}}
.live{padding:0;overflow:hidden;background:#fff}.live>p{padding:18px}.live iframe{width:100%;height:760px;border:0;display:block;background:#fff}
.live .off{padding:26px;display:grid;grid-template-columns:minmax(0,1.2fr) minmax(0,1fr);gap:24px;align-items:center}@media(max-width:760px){.live .off{grid-template-columns:minmax(0,1fr)}}
.live .off h3{margin-bottom:6px}pre.cmd{background:var(--soft);border:1px solid var(--line);border-radius:10px;padding:12px 14px;margin:12px 0;font:14px ui-monospace,SFMono-Regular,Menlo,monospace;color:var(--accent);overflow-x:auto}
.live .off img{width:100%;border-radius:10px;border:1px solid var(--line)}
.cases{display:grid;gap:18px}.case{border-radius:16px;padding:22px}
.case-head{display:flex;gap:12px;align-items:center}.case-head h3{min-width:0;overflow-wrap:anywhere}.tag{background:var(--accent);color:#fff;font-weight:700;border-radius:8px;padding:2px 10px;font-size:13px;flex:none}
.hero-num{font-size:clamp(24px,3vw,30px);font-weight:700;margin:12px 0;letter-spacing:-.02em;overflow-wrap:anywhere}.hero-num span{display:block;font-size:14px;font-weight:400;color:var(--muted);letter-spacing:0}
.shot{width:100%;border-radius:10px;border:1px solid var(--line);margin:6px 0;display:block}.facts{color:var(--text);font-size:14px;margin:10px 0}
.why,.fix{font-size:14.5px;color:#3b434c}.why b,.fix b{color:var(--accent);margin-right:6px}
.cmp{position:relative;border-radius:10px;overflow:hidden;border:1px solid var(--line);margin:6px 0;user-select:none}.cmp img{width:100%;display:block}
.cmp-top{position:absolute;inset:0;clip-path:inset(0 50% 0 0)}.cmp-top img{height:100%;object-fit:cover}
.cmp input{position:absolute;inset:0;width:100%;height:100%;opacity:0;cursor:ew-resize;margin:0}.cmp-line{position:absolute;top:0;bottom:0;left:50%;width:2px;background:var(--accent);pointer-events:none}
.cmp-l,.cmp-r{position:absolute;bottom:10px;background:rgba(255,255,255,.94);color:var(--text);border:1px solid var(--line);padding:2px 10px;border-radius:6px;font-size:13px;pointer-events:none}.cmp-l{left:10px}.cmp-r{right:10px}
.two{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(340px,100%),1fr));gap:16px}ul.b{padding-left:20px;color:#3b434c}ul.b li{margin:8px 0}
footer{padding:60px 0 80px;color:var(--muted);font-size:13.5px;border-top:1px solid var(--line);margin-top:72px}
@media(max-width:560px){.tog{display:flex;flex-direction:column;width:100%;border-radius:16px}.tog button{text-align:center;border-radius:12px}.wrap{padding:0 16px}.chart{height:280px}.chart.tall{height:340px}.lead{font-size:17px}}
</style></head><body>
<nav><div class="wrap"><b>city-incident-ai</b><a href="#live">Try it live</a><a href="#pipeline">Pipeline</a><a href="#numbers">Numbers</a><a href="#detection">Detection</a><a href="#tracking">Tracking</a><a href="#speed">Speed</a><a href="#errors">Errors</a><a href="#debug">Debugging</a><a href="#gait">Gait</a><a href="#limits">Limits</a></div></nav>

<header class="hero"><div class="wrap">
  <div><h1>Video analytics for<br><span style="color:var(--accent)">city cameras</span></h1>
  <p class="lead">YOLO11 detection and ByteTrack tracking on video, with rule-based events, measured against MOT17 and profiled on a laptop. Every number on this page is read from the evaluation outputs.</p></div>
  <div><div class="tog" role="tablist"><button class="on" data-src="videos/demo_clip61.mp4">Intersection · tracks + trails</button><button data-src="videos/demo_pose_clip4332.mp4">Night street · pose + event banner</button></div>
  <video id="hero" src="videos/demo_clip61.mp4" autoplay muted loop playsinline controls></video>
  <p class="muted" style="font-size:13px">Boxes with class and track ID, 1 s motion trails, FPS/latency counter. Second clip: yolo11n-pose skeletons; the red banner is the <code>crowding</code> event firing.</p></div>
</div></header>

<section id="live"><div class="wrap"><h2><small>Try it live</small>Run the pipeline on a clip</h2>
<p class="lead">Pick one of six example clips or upload your own, choose pose on/off, device and frame limit, and the real pipeline returns the annotated video and its events.</p>
<div id="live-box" class="box live"><p class="muted" style="margin:0">Checking for the local demo server…</p></div></div></section>

<section id="pipeline"><div class="wrap"><h2><small>How it works</small>Pipeline</h2>
<p class="lead">One pass per frame. The tracker is fed low-confidence boxes; drawing and events only use confident ones.</p>
<div class="box"><div class="flow" role="img" aria-label="video, detection, tracking, event rules, outputs"><div class="node"><b>Video</b><span>file or RTSP stream</span></div><div class="arr" aria-hidden="true"><svg viewBox="0 0 24 24"><path d="M4 12h15M13 6l6 6-6 6" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg></div><div class="node"><b>__PIPE_DET_T__</b><span>__PIPE_DET__</span></div><div class="arr" aria-hidden="true"><svg viewBox="0 0 24 24"><path d="M4 12h15M13 6l6 6-6 6" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg></div><div class="node"><b>ByteTrack</b><span>__PIPE_TRK__</span></div><div class="arr" aria-hidden="true"><svg viewBox="0 0 24 24"><path d="M4 12h15M13 6l6 6-6 6" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg></div><div class="node"><b>Event rules</b><span>__PIPE_EVT__</span></div><div class="arr" aria-hidden="true"><svg viewBox="0 0 24 24"><path d="M4 12h15M13 6l6 6-6 6" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg></div><div class="node"><b>Outputs</b><span>annotated video + JSON</span></div></div></div>
</div></section>

<section id="numbers"><div class="wrap"><h2><small>At a glance</small>Key numbers</h2><div class="stats">__STATS__</div></div></section>

<section id="detection"><div class="wrap"><h2><small>Detection</small>Person detection on MOT17</h2>
<p class="lead">YOLO11s, no fine-tuning, on three static-camera MOT17 sequences (02, 04, 09). The detector is good when people are large and visible, and weak when they are small or hidden.</p>
<div class="grid2"><div class="box" style="grid-column:1/-1"><h4>Precision–recall curve (IoU 0.5)</h4><p class="cap">Overall in accent, per sequence in grey. The dot marks the demo threshold, conf 0.35.</p><div class="chart tall"><canvas id="c-pr"></canvas></div></div>
<div class="box"><h4>Recall by visibility</h4><p class="cap">Fraction of ground-truth people found at conf 0.35, by how visible they are.</p><div class="chart"><canvas id="c-vis"></canvas></div></div>
<div class="box"><h4>Recall by box height</h4><p class="cap">Same threshold, by person height in pixels (frames are 1920×1080).</p><div class="chart"><canvas id="c-h"></canvas></div></div></div></div></section>

<section id="tracking"><div class="wrap"><h2><small>Tracking</small>ByteTrack vs BoT-SORT</h2>
<p class="lead">Same detections fed to both trackers, motion-only (ReID off), pooled over the three sequences.</p>
<p class="swipe">Swipe sideways for all columns →</p><div class="box scroll"><table><thead><tr><th>Tracker</th><th>MOTA</th><th>IDF1</th><th>ID switches</th><th>FP</th><th>FN</th><th>Mostly tracked</th><th>Mostly lost</th><th>Pred / GT IDs</th><th>Tracker ms/frame</th></tr></thead><tbody>__TRK_TABLE__</tbody></table></div>
<div class="grid2" style="margin-top:16px"><div class="box"><h4>MOTA and IDF1</h4><div class="chart"><canvas id="c-trk"></canvas></div></div><div class="box"><h4>ID switches (lower is better)</h4><div class="chart"><canvas id="c-trk-sw"></canvas></div></div></div>
<h3 style="margin-top:44px">Detector threshold vs ByteTrack</h3>
<p class="muted">ByteTrack's second association pass uses low-confidence boxes. The demo used to run the detector at conf __CONF_DISP__, so that pass never received any boxes. Now the tracker gets boxes from conf __CONF_TRACK__ while drawing and events still use only conf ≥ __CONF_DISP__.</p>
<div class="grid2"><div class="box"><h4>MOTA and IDF1</h4><div class="chart"><canvas id="c-abl"></canvas></div></div><div class="box"><h4>ID switches</h4><div class="chart"><canvas id="c-abl-sw"></canvas></div></div></div>
<p class="swipe">Swipe sideways for all columns →</p><div class="box scroll" style="margin-top:16px"><table><thead><tr><th>Configuration</th><th>MOTA</th><th>IDF1</th><th>ID switches</th><th>FP</th><th>FN</th><th>Pred / GT IDs</th><th>Recall at visibility &lt; 0.3</th></tr></thead><tbody>__ABL_TABLE__</tbody></table></div>
<div class="note">Feeding the tracker conf __CONF_TRACK__ boxes raised MOTA by __ABL_MOTA__ points and IDF1 by __ABL_IDF1__, and removed __ABL_FN__ missed detections, at the cost of __ABL_FP__ more false positives and __ABL_SW__ more ID switches. A longer track buffer (60) changed little (ID switches __ABL_BUF_SW__).</div>
</div></section>

<section id="speed"><div class="wrap"><h2><small>Speed vs accuracy</small>What it costs on a laptop</h2>
<p class="lead">Eight detector variants, batch 1, measured on one Mac. Hover a point for pipeline FPS and model size.</p>
<div class="box"><p class="swipe" style="margin-top:0">Swipe the chart sideways →</p><div class="chart-scroll"><div class="chart tall"><canvas id="c-speed"></canvas></div></div></div>
<p class="swipe">Swipe sideways for all columns →</p><div class="box scroll" style="margin-top:16px"><table><thead><tr><th>Variant</th><th>imgsz</th><th>MB</th><th>Model ms mean / p50 / p95</th><th>Pipeline FPS</th><th>AP50</th><th>Recall &lt;50 px</th><th>50–100 px</th><th>&gt;100 px</th></tr></thead><tbody>__SPD_TABLE__</tbody></table></div>
<p class="muted" style="font-size:13px">__ENV__. Model-only = backend forward pass; pipeline = decode + detect + track + rules + draw + write. Accuracy on every 5th MOT17 frame, recall at conf 0.35. Single runs.</p>
</div></section>

<section id="errors"><div class="wrap"><h2><small>Error analysis</small>Five ways it fails</h2>
<p class="lead">Each case is a concrete example chosen by a rule, with the measured number. The explanations are hypotheses unless a number says otherwise.</p>
<div class="cases">__CASES__</div></div></section>

<section id="debug"><div class="wrap"><h2><small>Debugging stories</small>Two bugs worth telling</h2>
<div class="two"><div class="box"><h4>INT8 gave AP50 = __I8_FULL_AP__</h4>
<p>The YOLO output is one tensor mixing box coordinates in pixels (0–640) with class scores (0–1). Quantizing the whole graph gave that tensor a single INT8 scale sized for the pixels, so every class score rounded to zero: max class score __I8_FULL_MAX__, __I8_DISTINCT__ distinct value. The boxes still looked fine, which hid it.</p>
<p>Keeping the detection-head decode in FP32 restored it: max score __I8_HEAD_MAX__ (FP32: __I8_FP_MAX__), AP50 __I8_HEAD_AP__ vs __I8_FP_AP__.</p></div>
<div class="box"><h4>ByteTrack never saw its second pass</h4>
<p>ByteTrack's key idea is a second matching round on low-confidence boxes. The pipeline filtered detections at conf __CONF_DISP__ before tracking, above the tracker's own thresholds, so that round had nothing to match. Splitting one threshold into two (tracker input __CONF_TRACK__, display and events __CONF_DISP__) fixed it: MOTA __ABL_MOTA__ points higher, measured above.</p></div></div></div></section>

<section id="gait"><div class="wrap"><h2><small>Next step</small>From city cameras to running gait</h2>
<p class="lead">The same video-in, structured-data-out pipeline carries over to gait analysis. The pose overlay below is the first piece.</p>
<div class="two"><div><img class="shot" src="img/demo_pose_runner.gif" alt="pose skeleton overlay on a runner on a treadmill, side view" loading="lazy"></div>
<div><ul class="b"><li><b>Pose keypoints.</b> A pose model replaces the box: hips, knees and ankles per frame become the signal, and a runner-specific model can be fine-tuned on running footage.</li>
<li><b>Single-person tracking.</b> The tracker would follow one runner and keep the identity through brief occlusions, which is the case the ablation above showed matters.</li>
<li><b>Step events.</b> Foot strike and toe-off can be detected from ankle keypoint trajectories, the same way this pipeline turns tracks into events.</li>
<li><b>Timing evaluation.</b> Event timing would be scored against ground truth (e.g. force-plate or manually labelled strikes), which is the evaluation this project does not yet have for its own event rules.</li></ul></div></div></div></section>

<section id="limits"><div class="wrap"><h2><small>Honest limitations</small>What this does not do</h2>
<ul class="b"><li>COCO classes only and no fine-tuning: nothing beyond the pretrained YOLO11 vocabulary is detected.</li>
<li>Small and occluded people are mostly missed: __LIM_TINY__ people under 50 px found at 640, and __LIM_OCC__ recall at visibility under 0.3.</li>
<li>The event rules are pixel-threshold heuristics with no event-level ground truth, so their precision and recall are unmeasured.</li>
<li>The collision rule is 2D box IoU and is off in the default configuration.</li>
<li>Evaluation covers three MOT17 sequences (person only, __LIM_FRAMES__ frames for the speed benchmark) with py-motmetrics, not the official TrackEval/HOTA; the demo clips have no ground truth. Speed numbers are single runs on one machine.</li></ul>
<p style="margin-top:28px"><a href="__REPO__/tree/interview-prep">View the code and reproduce every number on GitHub →</a></p></div></section>
<footer><div class="wrap">city-incident-ai · results generated from <code>reports/metrics/*.json</code> by <code>scripts/build_site.py</code></div></footer>

<script src="vendor/chart.umd.min.js" onerror="var s=document.createElement('script');s.src='https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js';s.onload=window.initCharts;document.head.appendChild(s)"></script>
<script>
const D = __DATA__;
const ACC='#0f766e', ACC2='#7fc4bb', G1='#8b96a3', G2='#c5ccd4', TXT='#586069', GRID='#e6eaef', INK='#1f2328';
// hero toggle
document.querySelectorAll('.tog button').forEach(b=>b.onclick=()=>{document.querySelectorAll('.tog button').forEach(x=>x.classList.remove('on'));b.classList.add('on');const v=document.getElementById('hero');v.src=b.dataset.src;v.play().catch(()=>{});});
// live demo: iframe when the local server answers /demo, otherwise a card (static hosting, file://, server down)
(async function(){
 const box=document.getElementById('live-box');
 const off=()=>{box.innerHTML='<div class="off"><div><h3>The live demo runs locally</h3><p class="muted">This page is a static copy, so there is no server behind it. Clone the repo and run one command; the demo opens in this section at <code>http://localhost:8000</code>.</p><pre class="cmd">make run</pre><p class="muted" style="font-size:13px">Upload a clip or pick one of the six example clips, optionally with pose skeletons (shown right).</p></div><img src="img/demo_pose_runner.gif" alt="pose skeleton on a treadmill runner" loading="lazy"></div>';};
 try{
  const ctl=new AbortController(), t=setTimeout(()=>ctl.abort(),2500);
  const r=await fetch('demo/',{signal:ctl.signal,cache:'no-store'}); clearTimeout(t);
  const txt=await r.text();
  if(!r.ok||!/gradio/i.test(txt)) throw new Error('no demo');
  const f=document.createElement('iframe'); f.src='demo/?__theme=light'; f.title='Live demo'; box.replaceChildren(f);
  const fit=()=>{try{const d=f.contentDocument,c=d&&d.querySelector('.gradio-container');if(c){const h=Math.ceil(c.getBoundingClientRect().height);if(h>200&&Math.abs(h-f.offsetHeight)>6)f.style.height=h+'px';}}catch(e){}};
  f.addEventListener('load',()=>{fit();setInterval(fit,400);});
 }catch(e){off();}
})();
// compare slider
(function(){const c=document.getElementById('cmp');if(!c)return;const top=c.querySelector('.cmp-top'),line=c.querySelector('.cmp-line');c.querySelector('input').oninput=e=>{const v=e.target.value;top.style.clipPath='inset(0 '+(100-v)+'% 0 0)';line.style.left=v+'%';};})();
function initCharts(){
if(!window.Chart||window.__done)return;window.__done=1;
Chart.defaults.color=TXT;Chart.defaults.font.family=getComputedStyle(document.body).fontFamily;Chart.defaults.borderColor=GRID;Object.assign(Chart.defaults.plugins.tooltip,{backgroundColor:'#fff',titleColor:INK,bodyColor:INK,borderColor:'#d0d7de',borderWidth:1,padding:10,boxPadding:4});
const P=v=>(v*100).toFixed(1)+'%';
const bars=(id,labels,sets,fmt,opts={})=>new Chart(document.getElementById(id),{type:'bar',data:{labels,datasets:sets},options:Object.assign({responsive:true,maintainAspectRatio:false,
 plugins:{legend:{display:sets.length>1,labels:{usePointStyle:true,boxWidth:8,boxHeight:8}},tooltip:{callbacks:{label:c=>(c.dataset.label?c.dataset.label+': ':'')+fmt(c.parsed.y)}}},
 scales:{y:{beginAtZero:true,grid:{color:GRID},ticks:{callback:v=>fmt(v)}},x:{grid:{display:false},ticks:{maxRotation:0,autoSkip:false}}}},opts)});
// PR
const cv=D.pr.curves, mk=(key,label,c,w)=>({label:label,data:cv[key].points.map(p=>({x:p.recall,y:p.precision})),showLine:true,pointRadius:0,borderColor:c,borderWidth:w,tension:0});
const SEQ=['MOT17-02','MOT17-04','MOT17-09'];
const ds=SEQ.map((n,i)=>mk(n,n+' (AP50 '+cv[n].AP50.toFixed(3)+')',['#b6bec8','#8b96a3','#6b7785'][i],1.2));
ds.push(mk('overall','Overall (AP50 '+cv.overall.AP50.toFixed(3)+')',ACC,3));
const dp=cv.overall.demo_point;ds.push({label:'conf 0.35: P '+P(dp.precision)+', R '+P(dp.recall),data:[{x:dp.recall,y:dp.precision}],showLine:false,pointRadius:7,backgroundColor:'#fff',borderColor:ACC,borderWidth:3});
new Chart(document.getElementById('c-pr'),{type:'scatter',data:{datasets:ds},options:{responsive:true,maintainAspectRatio:false,scales:{x:{min:0,max:1,title:{display:true,text:'Recall'},grid:{color:GRID}},y:{min:0,max:1.02,title:{display:true,text:'Precision'},grid:{color:GRID}}},plugins:{legend:{labels:{usePointStyle:true,boxWidth:8,boxHeight:8}},tooltip:{callbacks:{label:c=>c.dataset.label.split(' (')[0]+'  R '+c.parsed.x.toFixed(2)+'  P '+c.parsed.y.toFixed(2)}}}}});
// recall bars
const rv=D.recall_vis, rh=D.recall_h;
bars('c-vis',Object.keys(rv).map(k=>['visibility '+k,'n='+D.recall_vis_n[k].toLocaleString()]),[{data:Object.values(rv),backgroundColor:[G2,G1,ACC],borderRadius:6}],P,{scales:{y:{beginAtZero:true,max:1,ticks:{callback:v=>P(v)},grid:{color:GRID}},x:{grid:{display:false},ticks:{maxRotation:0,autoSkip:false}}}});
bars('c-h',Object.keys(rh).map(k=>[k+' px','n='+D.recall_h_n[k].toLocaleString()]),[{data:Object.values(rh),backgroundColor:[G2,G1,ACC],borderRadius:6}],P,{scales:{y:{beginAtZero:true,max:1,ticks:{callback:v=>P(v)},grid:{color:GRID}},x:{grid:{display:false},ticks:{maxRotation:0,autoSkip:false}}}});
// trackers
const tn=Object.keys(D.trackers);
bars('c-trk',tn,[{label:'MOTA',data:tn.map(n=>D.trackers[n].mota),backgroundColor:ACC,borderRadius:6},{label:'IDF1',data:tn.map(n=>D.trackers[n].idf1),backgroundColor:G1,borderRadius:6}],P);
bars('c-trk-sw',tn,[{data:tn.map(n=>D.trackers[n].num_switches),backgroundColor:[ACC,G1],borderRadius:6}],v=>String(Math.round(v)));
// ablation: accent = demo default (B)
const al=D.ablation.labels;
bars('c-abl',al,[{label:'MOTA',data:D.ablation.mota,backgroundColor:[G2,ACC,G2],borderRadius:6},{label:'IDF1',data:D.ablation.idf1,backgroundColor:[G1,ACC2,G1],borderRadius:6}],P);
bars('c-abl-sw',al,[{data:D.ablation.idsw,backgroundColor:[G2,ACC,G2],borderRadius:6}],v=>String(Math.round(v)));
// speed scatter with labels
const sp=D.speed, OFF={fp32_pt_mps:[-4,-36,'left'],fp32_onnx_coreml:[10,-66,'left'],n_onnx_cpu:[10,22,'left'],fp32_pt_cpu:[-10,40,'right'],fp32_onnx_cpu:[14,-44,'left'],int8_onnx_cpu:[18,30,'left'],int8dyn_onnx_cpu:[12,-14,'left'],s960_onnx_cpu:[-12,24,'right']};
const lbl={id:'lbl',afterDatasetsDraw(ch){const c=ch.ctx;c.save();c.font='12px '+Chart.defaults.font.family;c.lineWidth=1;
 sp.forEach((p,i)=>{const m=ch.getDatasetMeta(0).data[i],[dx,dy,al]=OFF[p.key]||[10,-10,'left'];const tx=m.x+dx,ty=m.y+dy;
  if(Math.abs(dy)>24){c.strokeStyle='#b4bdc7';c.beginPath();c.moveTo(m.x,m.y+(dy<0?-9:9));c.lineTo(tx+(al=='left'?-2:2),ty+(dy<0?3:-12));c.stroke();}
  c.fillStyle=p.key=='fp32_pt_mps'?ACC:INK;c.font=(p.key=='fp32_pt_mps'?'600 ':'')+'12px '+Chart.defaults.font.family;c.textAlign=al;c.fillText(p.name,tx,ty);});c.restore();}};
new Chart(document.getElementById('c-speed'),{type:'scatter',data:{datasets:[{data:sp.map(p=>({x:p.x,y:p.y})),pointRadius:8,pointHoverRadius:10,backgroundColor:sp.map(p=>p.key=='fp32_pt_mps'?ACC:G1),borderColor:'#fff',borderWidth:2}]},plugins:[lbl],
 options:{responsive:true,maintainAspectRatio:false,layout:{padding:{right:30,top:20}},scales:{x:{min:0,title:{display:true,text:'model-only latency, p50 (ms) — lower is faster'},grid:{color:GRID}},y:{title:{display:true,text:'person AP50 — higher is better'},grid:{color:GRID},ticks:{callback:v=>v.toFixed(2)}}},
 plugins:{legend:{display:false},tooltip:{callbacks:{title:c=>sp[c[0].dataIndex].label,label:c=>{const p=sp[c.dataIndex];return ['p50 '+p.x.toFixed(1)+' ms','AP50 '+p.y.toFixed(3),'pipeline '+p.fps.toFixed(1)+' FPS','model '+p.mb+' MB, imgsz '+p.imgsz];}}}}}});
}
window.initCharts=initCharts;
if(window.Chart)initCharts();else window.addEventListener('load',()=>{if(window.Chart)initCharts();});
</script></body></html>
"""

if __name__ == "__main__":
    main()
