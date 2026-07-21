# Training Policy

Every branch goes through `should_train_branch(branch_name, dataset_stats, config)`.

Possible decisions:

- `train_full`
- `train_smoke_test`
- `use_pretrained_only`
- `skip_branch`

Key principles:

- the benchmark matrix does not guarantee training;
- `stopped_vehicle` is handled through tracking/rules;
- `pipe burst / leak` live in the anomaly branch, not as a forced detector class;
- segmentation is trained only when mask labels are available;
- the anomaly branch is trained only when normal-only / ROI-ready data is available.
