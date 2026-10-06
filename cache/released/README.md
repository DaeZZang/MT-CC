# Released MT-CC cache

Backbone outputs for running MT-CC without dataset images, backbone checkpoints,
or buffer index files. These are regular Git files, downloaded by `git clone`.

| Dataset | Seeds | Tasks per seed | Files per seed | Download size |
|---|---|---|---|---|
| CIFAR-100 | 1–5 | 10 | 50 | 238.2 MiB |

Only CIFAR-100 is included: 250 files, approximately 238.2 MiB.

Each task includes:

- `extract_group_train_*.pt`: Stage 1 features, full-head logits and labels.
- `extract_cal_*.pt`: Stage 2 features, full-head logits and labels.
- `extract_test_*.pt`: accumulated test-task features, full-head logits and labels.
- `tcil_tcil_*.pt`: single temperature (used for task 0 reporting).
- `rng_rng_*.pt`: RNG state before router training.

Files are under `<dataset>/ER/seed<seed>/`. Filename hashes come from
`cache.LogitCache.identity`; changing data-dependent settings needs matching
cache entries. Released settings: ER backbones, sequential class order,
adversarial calibration, group-data ratio 0.5, calibration epochs 100, batch size
128 and temperature scaling learning rate 0.1. Buffer and validation sizes follow
`data.py:DATASET_CONFIG`.

Adversarial image tensors are omitted because cache-only execution starts from
features/logits. It still fits the MT-CC router and temperatures and evaluates
cached test outputs. It does not run backbone inference on new images. Results
can vary across PyTorch versions and devices.

```bash
python main.py --dataset cifar100 --method mtcc --cache_only --seeds 1
```

See the [step-by-step README](../../README.md) for setup and training from scratch.
