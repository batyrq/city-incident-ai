"""Print the README 'Speed vs accuracy' markdown table from reports/metrics/speed.json."""
import json

d = json.load(open("reports/metrics/speed.json"))
print("| Variant | imgsz | size MB | model-only ms mean / p50 / p95 | detector total p50 ms | pipeline FPS | AP50 | recall@0.35 h<50 | 50-100 | >100 |")
print("|---|---:|---:|---|---:|---:|---:|---:|---:|---:|")
for k, v in d["variants"].items():
    if v.get("status") != "ok":
        print(f"| {v['label']} | | | skipped/failed: {v.get('reason','')[:80]} | | | | | | |"); continue
    m, t, a, r = v["latency"]["model_only"], v["latency"]["detector_total_incl_pre_nms"], v["accuracy"], v["accuracy"]["recall_by_height_px"]
    print(f"| {v['label']} | {v['imgsz']} | {v['size_mb']} | {m['mean_ms']:.1f} / {m['p50_ms']:.1f} / {m['p95_ms']:.1f} | {t['p50_ms']:.1f} | {v['pipeline']['fps']:.1f} | {a['AP50']:.3f} | "
          f"{r['<50']['recall']*100:.1f}% | {r['50-100']['recall']*100:.1f}% | {r['>100']['recall']*100:.1f}% |")
e = d["environment"]; print()
print(f"{e['chip']} ({e['cpu_cores']} cores), {e['ram_gb']} GB, macOS {e['macos']}, Python {e['python']}, torch {e['torch']}, onnxruntime {e['onnxruntime']}, ultralytics {e['ultralytics']}")
print({k:(v['pipeline']['tracker_fallback'],v['pipeline']['frames']) for k,v in d['variants'].items() if v.get('status')=='ok'})
print({k:(v['accuracy']['frames_evaluated'],v['accuracy']['n_gt_boxes'],{b:x['count'] for b,x in v['accuracy']['recall_by_height_px'].items()}) for k,v in list(d['variants'].items())[:1]})
