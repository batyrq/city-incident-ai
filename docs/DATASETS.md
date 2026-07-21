# Datasets

The project splits data sources into `auto_download`, `manual_download`, `local_only`.

- `auto_download`: used only when there is a stable public link and a clear verification step.
- `manual_download`: the pipeline does not fail; it creates the expected layout and writes instructions into `docs/MANUAL_DOWNLOAD.md`.
- `local_only`: used only when the user has already placed the data locally.

Adapters supported in this skeleton:

- `dfire`: fire/smoke detection, manual.
- `fasdd`: fire/smoke detection, manual.
- `onfire_local`: local videos, weak/local source.
- `r2s100k`: optional flood/water segmentation, manual.
- `cadp`: traffic incident detection, manual.
- `aicity2021_track4`: anomaly-ready challenge-style source, manual.
- `local_anomaly_rois`: local anomaly source.

If the data is missing, the training gate must choose `pretrained_only` or `skip_branch`.
