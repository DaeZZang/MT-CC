"""Calibration metrics computed from predicted probabilities.

ECE / AECE / SCE use ``n_bins`` bins over (lower, upper]; NLL and Brier are
bin-free. All functions accept torch tensors or numpy arrays.
"""

import numpy as np
import torch


def _to_numpy(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def ece(probs, labels, n_bins=10):
    probs, labels = _to_numpy(probs), _to_numpy(labels)
    conf = probs.max(axis=1)
    acc = (probs.argmax(axis=1) == labels)
    bounds = np.linspace(0, 1, n_bins + 1)
    out = 0.0
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        in_bin = (conf > lo) & (conf <= hi)
        prop = float(in_bin.mean())
        if prop > 0:
            out += abs(float(conf[in_bin].mean()) - float(acc[in_bin].mean())) * prop
    return out


def aece(probs, labels, n_bins=10):
    """Adaptive ECE: equal-mass bins (the last bin takes the remainder)."""
    probs, labels = _to_numpy(probs), _to_numpy(labels)
    conf = probs.max(axis=1)
    acc = (probs.argmax(axis=1) == labels)
    order = np.argsort(conf)
    conf, acc = conf[order], acc[order]
    n = len(conf)
    size = n // n_bins
    out = 0.0
    for i in range(n_bins):
        s, e = i * size, (n if i == n_bins - 1 else (i + 1) * size)
        if e > s:
            out += (e - s) / n * abs(float(acc[s:e].mean()) - float(conf[s:e].mean()))
    return out


def sce(probs, labels, n_bins=10):
    """Static calibration error: class-wise ECE averaged over all output columns."""
    probs, labels = _to_numpy(probs), _to_numpy(labels)
    bounds = np.linspace(0, 1, n_bins + 1)
    n, k = probs.shape
    out = 0.0
    for c in range(k):
        conf = probs[:, c]
        acc = (labels == c).astype('float')
        for lo, hi in zip(bounds[:-1], bounds[1:]):
            in_bin = (conf > lo) & (conf <= hi)
            prop = float(in_bin.mean())
            if prop > 0:
                out += prop * abs(float(conf[in_bin].mean()) - float(acc[in_bin].mean()))
    return out / k


def nll(probs, labels):
    probs = torch.as_tensor(_to_numpy(probs), dtype=torch.float32)
    labels = torch.as_tensor(_to_numpy(labels), dtype=torch.long)
    return torch.nn.functional.nll_loss(torch.log(probs + 1e-8), labels).item()


def brier(probs, labels):
    probs = torch.as_tensor(_to_numpy(probs), dtype=torch.float32)
    labels = torch.as_tensor(_to_numpy(labels), dtype=torch.long)
    onehot = torch.zeros_like(probs)
    onehot[torch.arange(len(labels)), labels] = 1.0
    return ((probs - onehot) ** 2).sum(dim=1).mean().item()


def accuracy(probs, labels):
    probs, labels = _to_numpy(probs), _to_numpy(labels)
    return float((probs.argmax(axis=1) == labels).mean())


def all_metrics(probs, labels, n_bins=10):
    return {
        'ece': ece(probs, labels, n_bins),
        'aece': aece(probs, labels, n_bins),
        'sce': sce(probs, labels, n_bins),
        'nll': nll(probs, labels),
        'brier': brier(probs, labels),
        'acc': accuracy(probs, labels),
    }


def format_metrics(m):
    return (f"ECE: {m['ece'] * 100:.2f}%, AECE: {m['aece'] * 100:.2f}%, "
            f"SCE: {m['sce'] * 1000:.2f}‰, NLL: {m['nll']:.4f}, "
            f"Brier: {m['brier']:.4f}, Acc: {m['acc'] * 100:.2f}%")
