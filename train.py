"""Class-incremental training with Experience Replay (ER).

For every seed and task the script saves the backbone and the replay-buffer /
validation / training index files that ``main.py`` consumes. Directory layout
(see ``data.run_dir``):

    saved_model/ER/<DATASET>/base{B}_new{N}_replay{M}[_val{V}]/seed_{s}/ER_{s}_seed_{T}_tasks_{V}_val_{t}_task.pt
    saved_buffer_indices/ER/<DATASET>/.../seed_{s}/ER_{buffer,valid,train}_indices_{s}_seed_{T}_tasks_{V}_val_{t}_task.npy
"""

import argparse
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import ConcatDataset, DataLoader, Subset
from tqdm import tqdm

from data import (DATASET_CONFIG, checkpoint_name, class_indices, classes_per_task,
                  indices_name, load_datasets, run_dir, task_indices)
from model import build_model


def parse_args():
    p = argparse.ArgumentParser(description='Experience-replay training for class-incremental learning')
    p.add_argument('--dataset', default='cifar100', choices=list(DATASET_CONFIG))
    p.add_argument('--data_root', default='./data')
    p.add_argument('--model_path', default='./saved_model')
    p.add_argument('--buffer_path', default='./saved_buffer_indices')
    p.add_argument('--seeds', nargs='+', type=int, default=[1, 2, 3, 4, 5])
    p.add_argument('--replay_buffer_size', type=int, default=None, help='default: dataset-specific')
    p.add_argument('--validation_size', type=int, default=None,
                   help='validation samples held out per task (default: dataset-specific)')
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--factor', type=int, default=1,
                   help='CIFAR-100 / ImageNet: divide epochs and milestones by this from task 1 on')
    p.add_argument('--num_workers', type=int, default=12)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()
    cfg = DATASET_CONFIG[args.dataset]
    if args.replay_buffer_size is None:
        args.replay_buffer_size = cfg['replay_buffer_size']
    if args.validation_size is None:
        args.validation_size = cfg['validation_size']
    return args


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    args = parse_args()
    cfg = DATASET_CONFIG[args.dataset]
    device = torch.device(args.device)
    sizes = classes_per_task(cfg)
    num_tasks = len(sizes)
    print(f'tasks: {sizes}  buffer: {args.replay_buffer_size}  validation/task: {args.validation_size}')

    train_data, test_data = load_datasets(args.dataset, args.data_root, stage='train')
    train_idx = class_indices(train_data, cfg['num_classes'])
    test_tasks = task_indices(test_data, sizes, cfg['num_classes'])

    base_lr, milestones, batch_size = 0.1, [100, 150], cfg['train_batch_size']
    decay_per_task = args.dataset in ('cifar100', 'imagenet')
    criterion = nn.CrossEntropyLoss()
    start = time.time()

    for seed in args.seeds:
        set_seed(seed)
        model_dir = run_dir(args.model_path, args.dataset, cfg['base_classes'], cfg['new_classes_per_task'],
                            args.replay_buffer_size, args.validation_size, seed)
        index_dir = run_dir(args.buffer_path, args.dataset, cfg['base_classes'], cfg['new_classes_per_task'],
                            args.replay_buffer_size, args.validation_size, seed)
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(index_dir, exist_ok=True)

        model = build_model(args.dataset, cfg['num_classes']).to(device)
        buffer_idx = []
        avg_acc_over_tasks = 0.0

        for t in range(num_tasks):
            k_t = sizes[t]
            num_known = sum(sizes[:t])
            num_total = num_known + k_t
            if decay_per_task:
                lr = base_lr / (t + 1)
                epochs = args.epochs // args.factor if t >= 1 else args.epochs
                task_milestones = [m // args.factor for m in milestones] if t >= 1 else milestones
            else:
                lr, epochs, task_milestones = base_lr, args.epochs, milestones
            print(f'--- seed {seed} task {t}: classes {num_known}-{num_total - 1}, lr={lr}, epochs={epochs}')
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=2e-4)
            scheduler = MultiStepLR(optimizer, milestones=task_milestones, gamma=0.1)

            # Hold out a class-balanced validation split from the new task's data.
            new_idx = [i for c in range(num_known, num_total) for i in train_idx[c]]
            per_class = len(new_idx) // k_t
            val_per_class = args.validation_size // k_t
            valid_idx = []
            for i in range(k_t):
                valid_idx += random.sample(new_idx[i * per_class:(i + 1) * per_class], val_per_class)
            valid_set = set(valid_idx)
            new_idx = [i for i in new_idx if i not in valid_set]
            new_task_data = Subset(train_data, new_idx)
            train_set = new_task_data if (t == 0 or args.replay_buffer_size == 0) \
                else ConcatDataset([new_task_data, Subset(train_data, buffer_idx)])
            loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                num_workers=args.num_workers, pin_memory=True)

            for _ in tqdm(range(epochs), desc=f'task {t}'):
                model.train()
                for x, y in loader:
                    x, y = x.to(device), y.to(device)
                    loss = criterion(model(x)[:, :num_total], y)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                scheduler.step()

            torch.save(model.state_dict(),
                       os.path.join(model_dir, checkpoint_name(seed, num_tasks, args.validation_size, t)))

            # Class-balanced replay buffer: shrink the old classes' share, add the new ones.
            per_class_buffer = args.replay_buffer_size // num_total
            new_buffer = []
            if t > 0:
                prev_per_class = args.replay_buffer_size // num_known
                for i in range(num_known):
                    new_buffer += random.sample(buffer_idx[i * prev_per_class:(i + 1) * prev_per_class], per_class_buffer)
            per_class = len(new_idx) // k_t
            for i in range(k_t):
                new_buffer += random.sample(new_idx[i * per_class:(i + 1) * per_class], per_class_buffer)
            buffer_idx = new_buffer
            for kind, idx in (('buffer', buffer_idx), ('valid', valid_idx), ('train', new_idx)):
                np.save(os.path.join(index_dir, indices_name(kind, seed, num_tasks, args.validation_size, t)),
                        np.array(idx))

            model.eval()
            avg_acc = 0.0
            for tt in range(t + 1):
                test_loader = DataLoader(Subset(test_data, test_tasks[tt]), batch_size=batch_size, shuffle=False,
                                         num_workers=args.num_workers, pin_memory=True)
                correct = 0
                with torch.no_grad():
                    for x, y in test_loader:
                        pred = model(x.to(device))[:, :num_total].argmax(1)
                        correct += (pred == y.to(device)).sum().item()
                acc = correct / len(test_tasks[tt])
                avg_acc += acc / (t + 1)
                print(f'  test task {tt}: {acc * 100:.2f}%')
            avg_acc_over_tasks += avg_acc / num_tasks
        print(f'seed {seed}: average incremental accuracy {avg_acc_over_tasks * 100:.2f}%')

    print(f'total runtime: {(time.time() - start) / 60:.1f} min')


if __name__ == '__main__':
    main()
