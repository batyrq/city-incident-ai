# Benchmark Plan

Supported matrix:

- Detection: YOLO11n/s/m, RT-DETR-L
- Segmentation: YOLO11n-seg, YOLO11s-seg
- Anomaly: EfficientAD-S, PatchCore

An honest benchmark means:

- skipped branches are marked as `skipped`;
- branches without a sufficient train set are marked as `pretrained_only`;
- `smoke_test_only` is not presented as a full production result.
