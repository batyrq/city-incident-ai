"""Reproduce the static-INT8 'AP50 0.000' bug and its fix -> reports/metrics/int8_debug.json.

Quantizes the repo's yolo11s.onnx twice (whole graph vs detect-head decode kept FP32), compares raw outputs against FP32 on
one calibration frame, and measures AP50 with the benchmark's accuracy protocol (MOT17 every 5th frame).
"""
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

from eval.benchmark_speed import WORK, calibration_frames, measure_accuracy, quantize_static_qdq


def main() -> None:
    from ultralytics import YOLO
    WORK.mkdir(parents=True, exist_ok=True)
    quantize_static_qdq(keep_head_fp32=False, dst_name="yolo11s_int8_full.onnx")
    quantize_static_qdq(keep_head_fp32=True, dst_name="yolo11s_int8_qdq.onnx")
    x = calibration_frames()[5]
    outs = {}
    for name, path in (("fp32", "yolo11s.onnx"), ("int8_full_graph", WORK / "yolo11s_int8_full.onnx"), ("int8_head_fp32", WORK / "yolo11s_int8_qdq.onnx")):
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        o = sess.run(None, {"images": x})[0][0]  # (84, 8400): rows 0-3 = xywh in pixels (0-640), rows 4-83 = class scores (0-1)
        acc = measure_accuracy(YOLO(str(path), task="detect"), 640, "cpu")
        outs[name] = {"box_rows_min": float(o[:4].min()), "box_rows_max": float(o[:4].max()), "class_score_max": float(o[4:].max()),
                      "person_score_max": float(o[4].max()), "distinct_class_score_values": int(len(np.unique(o[4:]))), "AP50": acc["AP50"]}
        print(name, outs[name], flush=True)
    Path("reports/metrics/int8_debug.json").write_text(json.dumps({"frame": "calibration_frames()[5]", "outputs": outs}, indent=2))


if __name__ == "__main__":
    main()
