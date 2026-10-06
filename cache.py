"""On-disk cache of the expensive, calibration-independent parts of a run.

Per (seed, task) four things can be cached:
  adv      the generated adversarial calibration set and its epsilon
  tcil     the single T-CIL temperature
  extract  (features, logits, labels) of the group-train / cal / test splits
  rng      the RNG state right before MT-CC training, so that a cache hit
           trains the router on exactly the same random stream as a miss

None of these depends on the MT-CC hyper-parameters, so one cache serves any
sweep over them. Entries are keyed by a hash of the fields that determine the
checkpoint and the data; the layout matches the cache written by the original
research code, so existing caches are reused as-is.
"""

import hashlib
import json
import os
import random

import numpy as np
import torch


class LogitCache:
    def __init__(self, cache_dir='cache/logits', mode='off'):
        self.cache_dir = cache_dir
        self.mode = mode  # 'auto' (read + write), 'refresh' (write only), 'off'

    @property
    def enabled(self):
        return self.mode != 'off'

    # ----- identities --------------------------------------------------------
    @staticmethod
    def _base(args, seed, task):
        return {
            'dataset': args.dataset, 'method_type': 'ER', 'seed': seed,
            'num_tasks': args.num_tasks, 'base_classes': args.base_classes,
            'new_classes_per_task': args.new_classes_per_task,
            'replay_buffer_size': args.replay_buffer_size,
            'validation_size': args.validation_size,
            'class_order_mode': 'sequential', 'class_order_file': '', 'task': task,
        }

    @staticmethod
    def _adv_fields(args):
        return {
            'calibration_data_source': args.calibration_data_source,
            'augmented_data_size': None,
            'calibration_epochs': args.calibration_epochs,
            'calibration_batch_size': args.calibration_batch_size,
            'temperature_scaling_lr': args.temperature_scaling_lr,
        }

    def identity(self, kind, args, seed, task, split=None):
        ident = self._base(args, seed, task)
        ident['kind'] = kind
        if kind == 'extract':
            ident['split'] = split
            ident['feature_layer'] = 'final'
            ident['feature_pool_size'] = 8
            if split != 'test':
                ident.update(self._adv_fields(args))
                ident['group_data_ratio'] = args.group_data_ratio
                ident['reuse_data'] = False
        else:  # adv, tcil, rng
            ident.update(self._adv_fields(args))
        return ident

    def path(self, ident):
        h = hashlib.md5(json.dumps(ident, sort_keys=True, default=str).encode('utf-8')).hexdigest()[:16]
        split = ident.get('split', ident['kind'])
        return os.path.join(self.cache_dir, str(ident['dataset']), 'ER', f"seed{ident['seed']}",
                            f"{ident['kind']}_{split}_task{ident['task']}_{h}.pt")

    # ----- I/O ---------------------------------------------------------------
    def load(self, ident):
        if self.mode != 'auto':
            return None
        path = self.path(ident)
        if not os.path.exists(path):
            return None
        obj = torch.load(path, map_location='cpu', weights_only=False)
        print(f'  [cache] hit  {path}')
        return obj

    def save(self, ident, obj):
        if not self.enabled:
            return
        path = self.path(ident)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(_to_cpu(obj), path + '.tmp')
        os.replace(path + '.tmp', path)
        print(f'  [cache] save {path}')


def _to_cpu(obj):
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: _to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_cpu(v) for v in obj)
    return obj


def snapshot_rng():
    state = {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state()}
    if torch.cuda.is_available():
        state['cuda'] = torch.cuda.get_rng_state_all()
    return state


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(torch.as_tensor(state['torch'], dtype=torch.uint8).cpu())
    if 'cuda' in state and torch.cuda.is_available():
        cuda_states = [torch.as_tensor(s, dtype=torch.uint8).cpu() for s in state['cuda']]
        if len(cuda_states) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all(cuda_states)
