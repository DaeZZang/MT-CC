#!/bin/bash
# MT-CC on the saved ER checkpoints (Table 1, lower block). All MT-CC
# hyper-parameters default to the paper configuration; only the dataset differs.
set -euo pipefail
DATA_ROOT="${DATA_ROOT:-./data}"

for DATASET in cifar10 cifar100 tiny-imagenet; do
    python main.py --dataset "$DATASET" --data_root "$DATA_ROOT" --method mtcc
done

# "MT-CC w/o aug": fit on the validation split instead of T-CIL's adversarial samples.
# python main.py --dataset cifar100 --data_root "$DATA_ROOT" --method mtcc --calibration_data_source validation
