# Benchmark Plan

Поддерживаемая матрица:

- Detection: YOLO11n/s/m, RT-DETR-L
- Segmentation: YOLO11n-seg, YOLO11s-seg
- Anomaly: EfficientAD-S, PatchCore

Честный benchmark означает:

- пропущенные ветки помечаются как `skipped`;
- ветки без достаточного train set помечаются как `pretrained_only`;
- `smoke_test_only` не выдаётся как полноценный production result.
