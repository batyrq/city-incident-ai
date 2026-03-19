# Datasets

Проект разделяет источники данных на `auto_download`, `manual_download`, `local_only`.

- `auto_download`: используется только если есть стабильная публичная ссылка и понятная проверка.
- `manual_download`: pipeline не падает, а создаёт ожидаемую раскладку и пишет инструкцию в `docs/MANUAL_DOWNLOAD.md`.
- `local_only`: используется только если пользователь уже положил данные локально.

Поддерживаемые адаптеры в этом каркасе:

- `dfire`: fire/smoke detection, manual.
- `fasdd`: fire/smoke detection, manual.
- `onfire_local`: локальные видео, weak/local source.
- `r2s100k`: optional flood/water segmentation, manual.
- `cadp`: traffic incident detection, manual.
- `aicity2021_track4`: anomaly-ready challenge-style source, manual.
- `local_anomaly_rois`: локальный anomaly source.

Если данных нет, training gate обязан выбрать `pretrained_only` или `skip_branch`.
