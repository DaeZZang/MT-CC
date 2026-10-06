"""Datasets, class-incremental task splits and checkpoint paths.

Shared by ``train.py`` (trains the ER backbone) and ``main.py`` (post-hoc
calibration). Classes are presented in their original order: task 0 holds the
first ``base_classes`` classes, every later task the next ``new_classes_per_task``.
"""

import os

import torch
from PIL import Image
from torchvision import datasets, transforms

_IMAGENET_MEAN, _IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

# Per-dataset defaults. ``train_std``/``eval_std`` differ for CIFAR-10 only: the
# released checkpoints were trained with the first and evaluated/calibrated with
# the second, so both are kept for checkpoint compatibility.
DATASET_CONFIG = {
    'cifar10': dict(
        num_classes=10, base_classes=2, new_classes_per_task=2,
        replay_buffer_size=200, validation_size=500, train_batch_size=128,
        mean=(0.4914, 0.4822, 0.4465),
        train_std=(0.2470, 0.2435, 0.2616), eval_std=(0.2023, 0.1994, 0.2010),
        folder='CIFAR10'),
    'cifar100': dict(
        num_classes=100, base_classes=10, new_classes_per_task=10,
        replay_buffer_size=2000, validation_size=500, train_batch_size=128,
        mean=(0.5071, 0.4867, 0.4408),
        train_std=(0.2675, 0.2565, 0.2761), eval_std=(0.2675, 0.2565, 0.2761),
        folder='CIFAR100'),
    'tiny-imagenet': dict(
        num_classes=200, base_classes=20, new_classes_per_task=20,
        replay_buffer_size=2000, validation_size=100, train_batch_size=256,
        mean=_IMAGENET_MEAN, train_std=_IMAGENET_STD, eval_std=_IMAGENET_STD,
        folder='TINY-IMAGENET'),
    'imagenet': dict(
        num_classes=1000, base_classes=100, new_classes_per_task=100,
        replay_buffer_size=20000, validation_size=5000, train_batch_size=256,
        mean=_IMAGENET_MEAN, train_std=_IMAGENET_STD, eval_std=_IMAGENET_STD,
        folder='Imagenet'),
}


class TinyImageNet(torch.utils.data.Dataset):
    """Tiny-ImageNet-200 from the official directory layout (``wnids.txt`` etc.)."""

    def __init__(self, root, split, transform=None):
        self.transform = transform
        with open(os.path.join(root, 'wnids.txt')) as f:
            class_names = [line.strip() for line in f]
        class_to_idx = {c: i for i, c in enumerate(class_names)}
        self.images, self.labels = [], []
        if split == 'train':
            for c in class_names:
                d = os.path.join(root, 'train', c, 'images')
                for fn in os.listdir(d):  # directory order, as used when the index files were written
                    if fn.endswith('.JPEG'):
                        self.images.append(os.path.join(d, fn))
                        self.labels.append(class_to_idx[c])
        else:
            with open(os.path.join(root, 'val', 'val_annotations.txt')) as f:
                for line in f:
                    fn, c = line.strip().split('\t')[:2]
                    self.images.append(os.path.join(root, 'val', 'images', fn))
                    self.labels.append(class_to_idx[c])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, self.labels[idx]


class ImageNet(torch.utils.data.Dataset):
    """ImageNet-1k from ``<root>/train`` and ``<root>/val`` ImageFolder trees."""

    def __init__(self, root, split, transform=None):
        folder = datasets.ImageFolder(os.path.join(root, split))
        self.loader = folder.loader
        self.transform = transform
        self.images = [p for p, _ in folder.imgs]
        self.labels = [t for _, t in folder.imgs]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.loader(self.images[idx])
        if self.transform is not None:
            image = self.transform(image)
        return image, self.labels[idx]


def build_transform(name, augment, std):
    cfg = DATASET_CONFIG[name]
    normalize = transforms.Normalize(cfg['mean'], std)
    if name in ('cifar10', 'cifar100'):
        aug = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
        base = []
    elif name == 'tiny-imagenet':
        aug = [transforms.RandomCrop(64, padding=8), transforms.RandomHorizontalFlip()]
        base = []
    else:
        aug = [transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip()]
        base = [transforms.Resize(256), transforms.CenterCrop(224)]
    ops = aug if augment else base
    return transforms.Compose(ops + [transforms.ToTensor(), normalize])


def load_datasets(name, data_root, stage):
    """Return ``(train_data, test_data)``.

    stage='train': augmented training split, normalised with ``train_std``.
    stage='calib': no augmentation anywhere (buffer / validation samples are
    consumed as-is by calibration), normalised with ``eval_std``.
    """
    cfg = DATASET_CONFIG[name]
    std = cfg['train_std'] if stage == 'train' else cfg['eval_std']
    t_train = build_transform(name, augment=(stage == 'train'), std=std)
    t_test = build_transform(name, augment=False, std=std)
    if name == 'cifar100':
        train = datasets.CIFAR100(data_root, train=True, download=True, transform=t_train)
        test = datasets.CIFAR100(data_root, train=False, download=True, transform=t_test)
    elif name == 'cifar10':
        train = datasets.CIFAR10(data_root, train=True, download=True, transform=t_train)
        test = datasets.CIFAR10(data_root, train=False, download=True, transform=t_test)
    elif name == 'tiny-imagenet':
        root = os.path.join(data_root, 'tiny-imagenet-200')
        train = TinyImageNet(root, 'train', t_train)
        test = TinyImageNet(root, 'val', t_test)
    else:
        root = os.path.join(data_root, 'imagenet')
        train = ImageNet(root, 'train', t_train)
        test = ImageNet(root, 'val', t_test)
    return train, test


def labels_of(dataset):
    return list(dataset.targets) if hasattr(dataset, 'targets') else list(dataset.labels)


def classes_per_task(cfg):
    """[base_classes, new, new, ...] covering all classes."""
    remaining = cfg['num_classes'] - cfg['base_classes']
    n_new = (remaining + cfg['new_classes_per_task'] - 1) // cfg['new_classes_per_task']
    sizes = [cfg['base_classes']] + [cfg['new_classes_per_task']] * n_new
    if remaining % cfg['new_classes_per_task'] != 0:
        sizes[-1] = remaining - cfg['new_classes_per_task'] * (n_new - 1)
    return sizes


def class_indices(dataset, num_classes):
    """class id -> dataset indices, without decoding any image."""
    idx = {c: [] for c in range(num_classes)}
    for i, y in enumerate(labels_of(dataset)):
        idx[y].append(i)
    return idx


def task_indices(dataset, sizes, num_classes):
    """Dataset indices of every task, classes taken in order."""
    idx = class_indices(dataset, num_classes)
    out, start = [], 0
    for k in sizes:
        out.append([i for c in range(start, start + k) for i in idx[c]])
        start += k
    return out


def run_dir(root, name, base_classes, new_classes_per_task, replay_buffer_size,
            validation_size, seed):
    """Directory of one training run (checkpoints or index files)."""
    folder = f'base{base_classes}_new{new_classes_per_task}_replay{replay_buffer_size}'
    if name == 'cifar100':  # historical naming of the released CIFAR-100 runs
        folder += f'_val{validation_size}'
    return os.path.join(root, 'ER', DATASET_CONFIG[name]['folder'], folder, f'seed_{seed}')


def checkpoint_name(seed, num_tasks, validation_size, task):
    return f'ER_{seed}_seed_{num_tasks}_tasks_{validation_size}_val_{task}_task.pt'


def indices_name(kind, seed, num_tasks, validation_size, task):
    """kind in {'buffer', 'valid', 'train'}."""
    return f'ER_{kind}_indices_{seed}_seed_{num_tasks}_tasks_{validation_size}_val_{task}_task.npy'
