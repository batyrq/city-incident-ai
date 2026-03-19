# Training Policy

Каждая ветка проходит `should_train_branch(branch_name, dataset_stats, config)`.

Возможные решения:

- `train_full`
- `train_smoke_test`
- `use_pretrained_only`
- `skip_branch`

Ключевые принципы:

- benchmark matrix не гарантирует обучение;
- `stopped_vehicle` идёт через tracking/rules;
- `pipe burst / leak` живут в anomaly branch, а не forced detector class;
- segmentation тренируется только при наличии mask labels;
- anomaly branch тренируется только при наличии normal-only / ROI-ready данных.
