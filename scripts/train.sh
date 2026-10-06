#!/bin/bash
# Train the ER backbones that main.py calibrates (5 seeds, all tasks).
# Checkpoints go to saved_model/, index files to saved_buffer_indices/.
set -euo pipefail
DATA_ROOT="${DATA_ROOT:-./data}"

python train.py --dataset cifar10       --data_root "$DATA_ROOT"
python train.py --dataset cifar100      --data_root "$DATA_ROOT"
python train.py --dataset tiny-imagenet --data_root "$DATA_ROOT"
# python train.py --dataset imagenet    --data_root "$DATA_ROOT" --factor 2
