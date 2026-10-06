"""Post-hoc calibration of saved ER checkpoints.

Methods (``--method``):
  vanilla  no calibration
  ts       single temperature (T-CIL when the calibration set is adversarial)
  mtcc     the proposed group-wise calibration (W-Net + group temperatures)

The calibration set (``--calibration_data_source``) is either T-CIL's adversarial
perturbation of the replay buffer (default) or the current task's validation split.
Checkpoints and buffer/validation index files are read from the layout written by
``train.py``.
"""

import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset

from cache import LogitCache
from data import (DATASET_CONFIG, checkpoint_name, classes_per_task, indices_name,
                  labels_of, load_datasets, run_dir, task_indices)
from metrics import all_metrics, format_metrics
from methods.mtcc import run_mtcc_task
from methods.temp_scaling import validation_temperature
from model import build_model
from tcil import AdversarialTrainer, find_optimal_epsilon

METHODS = ['vanilla', 'ts', 'mtcc']


def parse_args():
    p = argparse.ArgumentParser(description='Post-hoc calibration for class-incremental learning')
    p.add_argument('--dataset', default='cifar100', choices=list(DATASET_CONFIG))
    p.add_argument('--data_root', default='./data')
    p.add_argument('--model_path', default='./saved_model')
    p.add_argument('--buffer_path', default='./saved_buffer_indices')
    p.add_argument('--results_dir', default='./results')
    p.add_argument('--experiment', default='', help='sub-folder name under results/<dataset>/')
    p.add_argument('--replay_buffer_size', type=int, default=None, help='default: dataset-specific')
    p.add_argument('--validation_size', type=int, default=None, help='default: dataset-specific')
    p.add_argument('--seeds', nargs='+', type=int, default=[1, 2, 3, 4, 5])
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--ece_bins', type=int, default=10)
    p.add_argument('--use_logit_cache', action='store_true',
                   help='cache adversarial sets, single temperatures and extracted logits per (seed, task) '
                        'so that re-runs skip the model forward passes (see cache.py)')
    p.add_argument('--cache_only', action='store_true',
                   help='run MT-CC from released features/logits without datasets or checkpoints')
    p.add_argument('--logit_cache_dir', default=None,
                   help='default: cache/released in cache-only mode, cache/logits otherwise')
    p.add_argument('--logit_cache_mode', default='auto', choices=['auto', 'refresh', 'off'],
                   help="auto: read hits / write misses; refresh: ignore existing entries and overwrite")

    p.add_argument('--method', default='mtcc', choices=METHODS)
    p.add_argument('--calibration_data_source', default='adversarial', choices=['adversarial', 'validation'])
    p.add_argument('--calibration_epochs', type=int, default=100, help='epochs of every temperature fit')
    p.add_argument('--calibration_batch_size', type=int, default=128)
    p.add_argument('--temperature_scaling_lr', type=float, default=0.1)

    # MT-CC (defaults = paper configuration)
    g = p.add_argument_group('MT-CC')
    g.add_argument('--num_groups', type=int, default=2)
    g.add_argument('--num_partitions', type=int, default=1, help='router ensembles (probabilities averaged)')
    g.add_argument('--group_data_ratio', type=float, default=0.5,
                   help='fraction of every class of the calibration set used to train the router; '
                        'the rest fits the group temperatures')
    g.add_argument('--w_net_input_mode', default='confidence+logit_margin',
                   help="'+'-joined subset of features, confidence, top1_confidence, logit_margin, entropy")
    g.add_argument('--normalize_w_net_inputs', action=argparse.BooleanOptionalAction, default=True)
    g.add_argument('--use_old_new_signal', action=argparse.BooleanOptionalAction, default=True,
                   help='append [p_old, p_new] (mass on previous-task vs current-task classes)')
    g.add_argument('--alpha', type=float, default=0.0, help='hard/soft mix of the old/new signal in training')
    g.add_argument('--first_task_old_ratio', type=float, default=1.0)
    g.add_argument('--w_net_bias', action=argparse.BooleanOptionalAction, default=None,
                   help='router bias (default: off for CIFAR-10, on otherwise)')
    g.add_argument('--temp_init_values', nargs='+', type=float, default=[1.5, 1.5],
                   help='first-task initial temperature per group')
    g.add_argument('--use_stage1_tau_init', action=argparse.BooleanOptionalAction, default=True,
                   help='initialise Stage-2 group temperatures from Stage 1')
    g.add_argument('--group_weight_decay', type=float, default=0.5)
    g.add_argument('--group_optimizer', default='adam', choices=['adam', 'sgd', 'sgd_momentum'])
    g.add_argument('--group_learning_rate', type=float, default=None, help='default: 0.01 CIFAR-10, 0.001 otherwise')
    g.add_argument('--group_batch_size', type=int, default=16)
    g.add_argument('--group_epochs', type=int, default=50)
    g.add_argument('--groupwise_dca_weight', type=float, default=1.0, help='0 disables the DCA term')

    args = p.parse_args()
    if args.cache_only:
        if args.method != 'mtcc':
            p.error('--cache_only requires --method mtcc')
        if args.logit_cache_mode != 'auto':
            p.error('--cache_only requires --logit_cache_mode auto')
        args.use_logit_cache = True
    if args.logit_cache_dir is None:
        args.logit_cache_dir = 'cache/released' if args.cache_only else 'cache/logits'
    cfg = DATASET_CONFIG[args.dataset]
    for key in ('num_classes', 'base_classes', 'new_classes_per_task'):
        setattr(args, key, cfg[key])
    if args.replay_buffer_size is None:
        args.replay_buffer_size = cfg['replay_buffer_size']
    if args.validation_size is None:
        args.validation_size = cfg['validation_size']
    if args.group_learning_rate is None:
        args.group_learning_rate = 0.01 if args.dataset == 'cifar10' else 0.001
    if not args.experiment:
        args.experiment = f'{args.method}_{args.calibration_data_source}'
    args.cache = LogitCache(args.logit_cache_dir, args.logit_cache_mode if args.use_logit_cache else 'off')
    return args


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Tee:
    """Mirror stdout to a log file."""

    def __init__(self, path):
        self.stdout, self.file = sys.stdout, open(path, 'w', encoding='utf-8')

    def write(self, s):
        self.stdout.write(s)
        self.file.write(s)

    def flush(self):
        self.stdout.flush()
        self.file.flush()


@torch.no_grad()
def collect_logits(model, dataset, device, batch_size):
    """Logits over the full head and labels of ``dataset``."""
    model.eval()
    logits, labels = [], []
    for x, y in DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True):
        logits.append(model(x.to(device, non_blocking=True)))
        labels.append(y.to(device))
    return torch.cat(logits), torch.cat(labels)


def to_tensors(dataset):
    xs, ys = zip(*[dataset[i] for i in range(len(dataset))])
    return torch.stack(xs), torch.tensor(ys)


def run_seed(args, train_data, test_data, test_tasks, sizes, seed, device):
    print(f'\n=== seed {seed} ===')
    set_seed(seed)
    ckpt_dir = run_dir(args.model_path, args.dataset, args.base_classes, args.new_classes_per_task,
                       args.replay_buffer_size, args.validation_size, seed)
    index_dir = run_dir(args.buffer_path, args.dataset, args.base_classes, args.new_classes_per_task,
                        args.replay_buffer_size, args.validation_size, seed)
    num_tasks = len(sizes)
    train_labels = labels_of(train_data)
    epochs, bs, lr = args.calibration_epochs, args.calibration_batch_size, args.temperature_scaling_lr
    prev_temps, prev_w_nets = None, None
    cache = args.cache
    history = []

    for t in range(num_tasks):
        k_t = sizes[t]
        num_seen = sum(sizes[:t + 1])  # classes learned so far
        test_set = Subset(test_data, [i for tt in range(t + 1) for i in test_tasks[tt]])

        model = build_model(args.dataset, args.num_classes).to(device)
        ckpt = os.path.join(ckpt_dir, checkpoint_name(seed, num_tasks, args.validation_size, t))
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        model.eval()
        buffer_idx = np.load(os.path.join(index_dir, indices_name('buffer', seed, num_tasks, args.validation_size, t)))
        valid_idx = np.load(os.path.join(index_dir, indices_name('valid', seed, num_tasks, args.validation_size, t)))
        buffer = Subset(train_data, buffer_idx)
        valid = Subset(train_data, valid_idx)

        if args.method == 'vanilla':
            logits, labels = collect_logits(model, test_set, device, bs)
            probs = F.softmax(logits[:, :num_seen], dim=1)
        else:
            # Temperature of the new task's validation split: the target that T-CIL's
            # adversarial calibration set is tuned to reproduce.
            target_temp = validation_temperature(model, valid, num_seen, epochs, bs, device)
            trainer = AdversarialTrainer(model, device)
            if args.calibration_data_source == 'adversarial':
                adv_ident = cache.identity('adv', args, seed, t)
                cached = cache.load(adv_ident)
                if cached is not None:
                    cal_x, cal_y, epsilon = cached['adv_data'], cached['labels'], cached['best_epsilon']
                else:
                    new_task_pos = [j for j, i in enumerate(buffer_idx) if num_seen - k_t <= train_labels[i] < num_seen]
                    epsilon = find_optimal_epsilon(trainer, buffer, Subset(buffer, new_task_pos), target_temp,
                                                   num_seen, t + 1, epochs, bs, lr)
                    cal_x, cal_y = trainer.generate_adversarial_data(buffer, buffer, num_seen, bs, epsilon, k_t)
                    cache.save(adv_ident, {'adv_data': cal_x, 'labels': cal_y, 'best_epsilon': epsilon})
                print(f'[Task {t}] adversarial calibration set: {len(cal_y)} samples, epsilon={epsilon:.4f}')
            else:
                cal_x, cal_y = to_tensors(valid)
            cal_x, cal_y = cal_x.to(device), cal_y.to(device)
            cal_loader = DataLoader(TensorDataset(cal_x, cal_y), batch_size=bs, shuffle=True)

            if args.method in ('ts', 'mtcc'):
                tcil_ident = cache.identity('tcil', args, seed, t)
                cached = cache.load(tcil_ident)
                if cached is not None:
                    temperature = float(cached['tcil_temperature'])
                else:
                    temperature = trainer.get_temperature(cal_loader, num_seen, epochs, bs, lr).item()
                    cache.save(tcil_ident, {'tcil_temperature': temperature})
                print(f'[Task {t}] single temperature: {temperature:.4f}')
            if args.method == 'ts' or (args.method == 'mtcc' and t == 0):
                # MT-CC trains its router on task 0 but, with a single task there is
                # nothing to group yet, so task 0 is reported with the single temperature.
                if args.method == 'mtcc':
                    out = run_mtcc_task(model, cal_x, cal_y, test_set, args, device, t, seed=seed)
                    prev_temps, prev_w_nets = out['init_temps'], out['w_nets']
                logits, labels = collect_logits(model, test_set, device, bs)
                probs = F.softmax(logits[:, :num_seen] / temperature, dim=1)
            elif args.method == 'mtcc':
                out = run_mtcc_task(model, cal_x, cal_y, test_set, args, device, t, prev_temps, prev_w_nets, seed=seed)
                prev_temps, prev_w_nets = out['init_temps'], out['w_nets']
                probs, labels = out['probs'], out['labels']
                print(f'[Task {t}] group temperatures: ' + ', '.join(
                    '[' + ', '.join(f'{v:.4f}' for v in temps) + ']' for temps in out['temperatures']))

        m = all_metrics(probs, labels, args.ece_bins)
        history.append(m)
        print(f'[Task {t}] ' + format_metrics(m))

    avg = {k: float(np.mean([m[k] for m in history])) for k in history[0]}
    print(f'[Seed {seed} Avg] ' + format_metrics(avg))
    return history


def check_released_cache(args):
    """Check every required entry before starting any router training."""
    missing = []
    for seed in args.seeds:
        for task in range(args.num_tasks):
            identities = [args.cache.identity('extract', args, seed, task, split)
                          for split in ('group_train', 'cal', 'test')]
            identities.append(args.cache.identity('rng', args, seed, task))
            if task == 0:
                identities.append(args.cache.identity('tcil', args, seed, task))
            for ident in identities:
                path = args.cache.path(ident)
                if not os.path.isfile(path):
                    missing.append(path)
                    continue
                with open(path, 'rb') as f:
                    if f.read(80).startswith(b'version https://git-lfs.github.com/spec/v1'):
                        raise RuntimeError('Cache file is an old pointer instead of tensor data. '
                                           'Update this checkout with git pull and retry.')
    if missing:
        raise FileNotFoundError(
            f'{len(missing)} required cache entries are missing; first: {missing[0]}. '
            'Update this checkout with git pull and use the released dataset/seeds and cache identity settings '
            '(adversarial calibration, group_data_ratio=0.5, default calibration settings), '
            'or generate a matching cache after training with --use_logit_cache.')


def run_cached_seed(args, sizes, seed, device):
    """Run the same MT-CC stages using only cached backbone outputs."""
    print(f'\n=== seed {seed} (cache only) ===')
    set_seed(seed)
    prev_temps, prev_w_nets = None, None
    history = []
    for task in range(len(sizes)):
        splits = {split: args.cache.load(args.cache.identity('extract', args, seed, task, split))
                  for split in ('group_train', 'cal', 'test')}
        out = run_mtcc_task(None, None, None, None, args, device, task,
                            prev_temps, prev_w_nets, seed=seed, cached_splits=splits)
        prev_temps, prev_w_nets = out['init_temps'], out['w_nets']
        if task == 0:
            # Match the checkpoint path: report task 0 with the single T-CIL temperature.
            cached = args.cache.load(args.cache.identity('tcil', args, seed, task))
            temperature = float(cached['tcil_temperature'])
            logits = splits['test']['logits'][:, :sizes[0]]
            probs = F.softmax(logits / temperature, dim=1)
            labels = splits['test']['labels']
            print(f'[Task {task}] single temperature: {temperature:.4f}')
        else:
            probs, labels = out['probs'], out['labels']
            print(f'[Task {task}] group temperatures: ' + ', '.join(
                '[' + ', '.join(f'{v:.4f}' for v in temps) + ']' for temps in out['temperatures']))
        m = all_metrics(probs, labels, args.ece_bins)
        history.append(m)
        print(f'[Task {task}] ' + format_metrics(m))
    avg = {k: float(np.mean([m[k] for m in history])) for k in history[0]}
    print(f'[Seed {seed} Avg] ' + format_metrics(avg))
    return history


def main():
    args = parse_args()
    out_dir = os.path.join(args.results_dir, args.dataset, args.experiment)
    os.makedirs(out_dir, exist_ok=True)
    stamp = time.strftime('%Y%m%d_%H%M%S')
    sys.stdout = Tee(os.path.join(out_dir, f'{stamp}.log'))
    print('command:', ' '.join(sys.argv))
    for k, v in sorted(vars(args).items()):
        print(f'  {k:<26}: {v}')

    device = torch.device(args.device)
    cfg = DATASET_CONFIG[args.dataset]
    sizes = classes_per_task(cfg)
    args.num_tasks = len(sizes)
    print(f'tasks: {sizes}')
    if args.cache_only:
        check_released_cache(args)
        per_seed = {seed: run_cached_seed(args, sizes, seed, device) for seed in args.seeds}
    else:
        train_data, test_data = load_datasets(args.dataset, args.data_root, stage='calib')
        test_tasks = task_indices(test_data, sizes, cfg['num_classes'])
        per_seed = {seed: run_seed(args, train_data, test_data, test_tasks, sizes, seed, device)
                    for seed in args.seeds}

    keys = ['ece', 'aece', 'sce', 'nll', 'brier', 'acc']
    seed_avg = {k: [np.mean([m[k] for m in hist]) for hist in per_seed.values()] for k in keys}
    print('\n=== FINAL (mean +- std over seeds, averaged over tasks) ===')
    scale = {'ece': 100, 'aece': 100, 'sce': 1000, 'nll': 1, 'brier': 1, 'acc': 100}
    for k in keys:
        v = np.array(seed_avg[k]) * scale[k]
        print(f'{k.upper():<6} {v.mean():.4f} +- {v.std():.4f}')
    print('\nper-task ECE (%):')
    per_task = np.array([[m['ece'] for m in hist] for hist in per_seed.values()]) * 100
    for t in range(len(sizes)):
        print(f'  task {t}: {per_task[:, t].mean():.2f} +- {per_task[:, t].std():.2f}')

    with open(os.path.join(out_dir, f'{stamp}.json'), 'w') as f:
        json.dump({'args': {k: v for k, v in vars(args).items() if k != 'cache'}, 'per_seed': {str(s): h for s, h in per_seed.items()},
                   'mean': {k: float(np.mean(seed_avg[k])) for k in keys},
                   'std': {k: float(np.std(seed_avg[k])) for k in keys}}, f, indent=2)


if __name__ == '__main__':
    main()
