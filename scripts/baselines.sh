#!/bin/bash
# Temperature scaling baselines on the same checkpoints.
#   adversarial : T-CIL calibration set
#   validation  : new-task validation split
set -euo pipefail
DATA_ROOT="${DATA_ROOT:-./data}"

for DATASET in cifar10 cifar100 tiny-imagenet; do
    for SOURCE in adversarial validation; do
        python main.py --dataset "$DATASET" --data_root "$DATA_ROOT" \
            --method ts --calibration_data_source "$SOURCE"
    done
    python main.py --dataset "$DATASET" --data_root "$DATA_ROOT" --method vanilla
done
