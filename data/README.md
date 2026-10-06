# Dataset downloads and locations

Run commands from the repository root. Actual dataset files are ignored by Git;
this directory is included so the expected location is visible on GitHub.
The default `--data_root` is `./data`.

| Dataset | Download | Expected location |
|---|---|---|
| CIFAR-10 | [Official CIFAR page](https://www.cs.toronto.edu/~kriz/cifar.html); automatic download | `data/cifar-10-batches-py/` |
| CIFAR-100 | [Official CIFAR page](https://www.cs.toronto.edu/~kriz/cifar.html); automatic download | `data/cifar-100-python/` |
| Tiny-ImageNet-200 | [Stanford archive](https://cs231n.stanford.edu/tiny-imagenet-200.zip) | `data/tiny-imagenet-200/` |
| ImageNet-1k (ILSVRC2012) | [Official download page](https://www.image-net.org/challenges/LSVRC/2012/2012-downloads.php), requires an account | `data/imagenet/` |

## CIFAR-10 / CIFAR-100

`train.py` downloads the selected dataset automatically. To download both before training:

```bash
python -c "from data import load_datasets; load_datasets('cifar10', './data', 'train'); load_datasets('cifar100', './data', 'train')"
```

## Tiny-ImageNet-200

```bash
mkdir -p data
curl -fL https://cs231n.stanford.edu/tiny-imagenet-200.zip -o data/tiny-imagenet-200.zip
unzip data/tiny-imagenet-200.zip -d data
```

Keep the original validation layout; `data.py` reads `val_annotations.txt` directly.

```text
data/tiny-imagenet-200/
├── wnids.txt
├── train/<wnid>/images/*.JPEG
└── val/
    ├── images/*.JPEG
    └── val_annotations.txt
```

## ImageNet-1k

Download the ILSVRC2012 training and validation images and development kit from
the official page. Extract training images into synset folders and organize
validation images into matching synset folders using the development kit labels.
Both splits must use the same class folder names:

```text
data/imagenet/
├── train/<wnid>/*.JPEG
└── val/<wnid>/*.JPEG
```

The loader expects this ImageFolder layout, not a flat validation directory.
ImageNet has no released cache here; train its backbone first.

## Use a different disk

Use the same parent directory for training and calibration. If Tiny-ImageNet is
at `/datasets/tiny-imagenet-200/`:

```bash
python train.py --dataset tiny-imagenet --data_root /datasets --seeds 1
python main.py --dataset tiny-imagenet --data_root /datasets --method mtcc --seeds 1
```

Dataset downloads are unnecessary with `--cache_only`; see the
[main README](../README.md) for the released-cache workflow.
