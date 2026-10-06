"""Regression checks for the dataset/checkpoint-free MT-CC path."""

import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import TensorDataset

from data import DATASET_CONFIG, classes_per_task
from main import check_released_cache, parse_args
from methods.mtcc import run_mtcc_task


def make_args(cache_dir):
    with patch.object(sys, 'argv', ['main.py', '--dataset', 'cifar10', '--cache_only',
                                   '--seeds', '1', '--logit_cache_dir', cache_dir]):
        args = parse_args()
    args.num_tasks = len(classes_per_task(DATASET_CONFIG[args.dataset]))
    args.group_epochs = 1
    return args


class CacheOnlyTests(unittest.TestCase):
    def test_preflight_reports_missing_entries(self):
        with tempfile.TemporaryDirectory() as directory:
            args = make_args(directory)
            with self.assertRaisesRegex(FileNotFoundError, 'required cache entries'):
                check_released_cache(args)

    def test_preflight_reports_old_pointer(self):
        with tempfile.TemporaryDirectory() as directory:
            args = make_args(directory)
            path = Path(args.cache.path(args.cache.identity('extract', args, 1, 0, 'group_train')))
            path.parent.mkdir(parents=True)
            path.write_text('version https://git-lfs.github.com/spec/v1\n')
            with self.assertRaisesRegex(RuntimeError, 'git pull'):
                check_released_cache(args)

    def test_cached_splits_match_backbone_extraction(self):
        class Backbone(torch.nn.Module):
            def forward_with_features(self, x):
                return x, x[:, :10]

        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            args = make_args(directory)
            args.calibration_epochs = 1
            torch.manual_seed(17)
            x = torch.randn(40, 12)
            y = torch.tensor([0, 1] * 20)
            test = TensorDataset(x[:12], y[:12])
            device = torch.device('cpu')
            live = run_mtcc_task(Backbone(), x, y, test, args, device, 0, seed=1)
            splits = {split: args.cache.load(args.cache.identity('extract', args, 1, 0, split))
                      for split in ('group_train', 'cal', 'test')}
            # A cache-only call must never reach backbone extraction.
            with patch('methods.mtcc.extract_features_and_logits', side_effect=AssertionError):
                cached = run_mtcc_task(None, None, None, None, args, device, 0,
                                       seed=1, cached_splits=splits)
            torch.testing.assert_close(cached['probs'], live['probs'], rtol=0, atol=0)
            torch.testing.assert_close(cached['labels'], live['labels'])
            torch.testing.assert_close(cached['groups'], live['groups'])
            np.testing.assert_array_equal(cached['init_temps'], live['init_temps'])


if __name__ == '__main__':
    unittest.main()
