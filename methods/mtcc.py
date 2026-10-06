"""MT-CC: group-wise temperature scaling with a learned grouping network (W-Net).

Per task:
  Stage 1  Train W-Net (a linear router over per-sample signals) jointly with one
           temperature per group by minimising the NLL of the temperature mixture,
           plus a group-wise DCA term that keeps every group's confidence close to
           its accuracy. The router and temperatures are carried over to the next task.
  Stage 2  Hard-assign calibration and test samples with the router, then fit one
           temperature per group with ordinary temperature scaling.
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader, TensorDataset

from cache import restore_rng, snapshot_rng
from methods.temp_scaling import calibrate as temp_scaling_calibrate

INPUT_PARTS = ('features', 'confidence', 'top1_confidence', 'logit_margin', 'entropy')


class WNet(torch.nn.Module):
    """Linear grouping network.

    ``input_mode`` joins any of the following with '+':
      features        penultimate features (z-scored)
      confidence      the full softmax vector
      top1_confidence the maximum probability
      logit_margin    top-1 minus top-2 logit (min-max normalised)
      entropy         predictive entropy (z-scored)
    With ``use_old_new_signal`` the vector [p_old, p_new] is appended: the
    probability mass on previously-seen vs. current-task classes (soft signal),
    mixed during training with the true membership from the labels (hard signal)
    as alpha * hard + (1 - alpha) * soft. On the first task, where no "new" classes
    exist yet, the signal is the constant [first_task_old_ratio, 1 - ratio].
    Normalisation statistics are fitted once, on the first data the router sees.
    """

    def __init__(self, feature_dim, num_classes, num_groups, input_mode='confidence+logit_margin',
                 normalize_inputs=True, use_old_new_signal=True, alpha=0.0,
                 first_task_old_ratio=1.0, bias=True):
        super().__init__()
        self.parts = input_mode.split('+')
        unknown = set(self.parts) - set(INPUT_PARTS)
        if unknown:
            raise ValueError(f'unknown W-Net input(s) {sorted(unknown)}; choose from {INPUT_PARTS}')
        dims = {'features': feature_dim, 'confidence': num_classes,
                'top1_confidence': 1, 'logit_margin': 1, 'entropy': 1}
        self.input_dim = sum(dims[p] for p in self.parts) + (2 if use_old_new_signal else 0)
        self.num_groups = num_groups
        self.normalize_inputs = normalize_inputs
        self.use_old_new_signal = use_old_new_signal
        self.alpha = alpha
        self.first_task_old_ratio = first_task_old_ratio
        self.old_classes, self.new_classes = [], []
        self.stats = None
        self.model = torch.nn.Sequential(torch.nn.Linear(self.input_dim, num_groups, bias=bias))

    def set_old_new_classes(self, old_classes, new_classes):
        self.old_classes, self.new_classes = list(old_classes), list(new_classes)

    @staticmethod
    def _signals(logits):
        probs = F.softmax(logits, dim=1)
        top2 = torch.topk(logits, k=min(2, logits.shape[1]), dim=1)[0]
        if logits.shape[1] >= 2:
            margin = top2[:, 0:1] - top2[:, 1:2]
        else:
            margin = torch.zeros_like(top2[:, :1])
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1, keepdim=True)
        return probs, margin, entropy

    def fit_stats(self, features, logits):
        _, margin, entropy = self._signals(logits)
        self.stats = {
            'f_mean': features.mean(0, keepdim=True), 'f_std': features.std(0, keepdim=True),
            'm_min': margin.min(0, keepdim=True)[0], 'm_max': margin.max(0, keepdim=True)[0],
            'e_mean': entropy.mean(0, keepdim=True), 'e_std': entropy.std(0, keepdim=True),
        }

    def _old_new_soft(self, probs):
        n, device = probs.shape[0], probs.device
        if self.old_classes and not self.new_classes:  # first task
            p_old = torch.full((n, 1), self.first_task_old_ratio, device=device)
            p_new = torch.full((n, 1), 1.0 - self.first_task_old_ratio, device=device)
        else:
            p_old = probs[:, self.old_classes].sum(1, keepdim=True) if self.old_classes else torch.zeros(n, 1, device=device)
            p_new = probs[:, self.new_classes].sum(1, keepdim=True) if self.new_classes else torch.zeros(n, 1, device=device)
        return torch.cat([p_old, p_new], dim=1)

    def _old_new_hard(self, labels):
        n, device = labels.shape[0], labels.device
        if self.old_classes and not self.new_classes:  # first task
            is_old = torch.full((n,), self.first_task_old_ratio, device=device, dtype=torch.float)
            is_new = torch.full((n,), 1.0 - self.first_task_old_ratio, device=device, dtype=torch.float)
        else:
            def member(classes):
                if not classes:
                    return torch.zeros(n, device=device, dtype=torch.float)
                return torch.isin(labels, torch.tensor(classes, device=device, dtype=labels.dtype)).float()
            is_old, is_new = member(self.old_classes), member(self.new_classes)
        return torch.stack([is_old, is_new], dim=1)

    def build_input(self, features, logits, labels=None, is_training=False):
        if self.normalize_inputs and self.stats is None:
            self.fit_stats(features, logits)
        probs, margin, entropy = self._signals(logits)
        s = {k: v.to(logits.device) for k, v in self.stats.items()} if self.normalize_inputs else None
        parts = []
        for p in self.parts:
            if p == 'features':
                parts.append((features - s['f_mean']) / (s['f_std'] + 1e-8) if s else features)
            elif p == 'confidence':
                parts.append(probs)
            elif p == 'top1_confidence':
                parts.append(probs.max(dim=1, keepdim=True)[0])
            elif p == 'logit_margin':
                parts.append((margin - s['m_min']) / (s['m_max'] - s['m_min'] + 1e-8) if s else margin)
            elif p == 'entropy':
                parts.append((entropy - s['e_mean']) / (s['e_std'] + 1e-8) if s else entropy)
        if self.use_old_new_signal:
            soft = self._old_new_soft(probs)
            if is_training and labels is not None:
                parts.append(self.alpha * self._old_new_hard(labels) + (1.0 - self.alpha) * soft)
            else:
                parts.append(soft)
        return torch.cat(parts, dim=1)

    def forward(self, features, logits, labels=None, is_training=False):
        x = self.build_input(features, logits, labels, is_training)
        if next(self.model.parameters()).device != x.device:
            self.model.to(x.device)
        return self.model(x)


def mixture_log_probs(logits, tau, group_logits):
    """log of sum_g softmax(router)_g * softmax(logits / tau_g): the soft-grouping
    calibrated log-probabilities optimised in Stage 1."""
    n, c = logits.shape
    g = tau.numel()
    log_w = F.log_softmax(group_logits, dim=1).view(n, g, 1).expand(n, g, c)
    log_p = F.log_softmax(logits.view(n, 1, c) / tau.view(1, g, 1), dim=2)
    return torch.logsumexp(log_w + log_p, dim=1)


def groupwise_dca(calibrated_logits, labels, group_logits):
    """Group-wise difference between average confidence and accuracy, weighted by
    sqrt(group size); groups are the router's hard assignments."""
    probs = F.softmax(calibrated_logits, dim=1)
    conf = probs.max(dim=1)[0]
    acc = (probs.argmax(dim=1) == labels).to(conf.dtype)
    onehot = F.one_hot(group_logits.argmax(dim=1), num_classes=group_logits.shape[1]).to(conf.dtype)
    counts = onehot.sum(dim=0)
    nonempty = counts > 0
    safe = torch.where(nonempty, counts, torch.ones_like(counts))
    conf_avg = torch.where(nonempty, (conf.unsqueeze(0) @ onehot).squeeze(0) / safe, torch.zeros_like(counts))
    acc_avg = torch.where(nonempty, (acc.unsqueeze(0) @ onehot).squeeze(0) / safe, torch.zeros_like(counts))
    weights = torch.where(nonempty, torch.sqrt(counts), torch.zeros_like(counts))[nonempty]
    dca = torch.abs(conf_avg - acc_avg)[nonempty]
    if weights.numel() == 0:
        return torch.zeros((), device=conf.device, dtype=conf.dtype)
    return (weights * dca).sum() / (weights.sum() + 1e-8)


def train_grouping_network(features, logits, labels, w_net, initial_temps, args):
    """Stage 1. Returns (tau, w_net) with ``tau`` on CPU and ``w_net`` moved to CPU."""
    device = features.device
    w_net.to(device)
    if w_net.normalize_inputs and w_net.stats is None:
        w_net.fit_stats(features, logits)
    log_tau = torch.nn.Parameter(torch.log(initial_temps.to(device)).clone().detach())
    params = [log_tau] + list(w_net.parameters())
    if args.group_optimizer == 'adam':
        optimizer = torch.optim.Adam(params, lr=args.group_learning_rate)
    elif args.group_optimizer == 'sgd':
        optimizer = torch.optim.SGD(params, lr=args.group_learning_rate)
    else:
        optimizer = torch.optim.SGD(params, lr=args.group_learning_rate, momentum=0.9)
    epochs = args.group_epochs
    scheduler = MultiStepLR(optimizer, milestones=[epochs // 2], gamma=0.1)
    n = features.shape[0]
    batch_size = min(args.group_batch_size or n, n)

    for _ in range(epochs):
        perm = torch.randperm(n)
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            bf, bl, by = features[idx], logits[idx], labels[idx]
            optimizer.zero_grad()
            tau = torch.exp(log_tau)
            reg = 0
            for name, p in w_net.named_parameters():
                if 'weight' in name:
                    reg = reg + torch.mean(p ** 2)
            reg = reg * args.group_weight_decay
            group_logits = w_net(bf, bl, by, is_training=True)
            calibrated = mixture_log_probs(bl, tau, group_logits)
            loss = F.cross_entropy(calibrated, by) + reg
            if args.groupwise_dca_weight > 0:
                loss = loss + args.groupwise_dca_weight * groupwise_dca(calibrated, by, group_logits)
            loss.backward()
            optimizer.step()
        scheduler.step()
    return torch.exp(log_tau).detach().cpu(), w_net.cpu()


def group_temperature_scaling(w_net, tau, cal, test, args):
    """Stage 2. Returns (probs, group ids, temperatures) with group ids relabelled in
    ascending order of temperature, plus the temperatures in the router's own order."""
    cal_features, cal_logits, cal_labels = cal
    test_features, test_logits, _ = test
    device = cal_logits.device
    w_net.to(device)
    tau = tau.to(device)
    with torch.no_grad():
        cal_groups = w_net(cal_features, cal_logits, cal_labels, is_training=True).argmax(dim=1)
        test_groups = w_net(test_features, test_logits, None, is_training=False).argmax(dim=1)

    probs = torch.zeros_like(test_logits)
    temps = tau.cpu().numpy().tolist()
    for g in range(w_net.num_groups):
        test_mask = test_groups == g
        if test_mask.sum() == 0:
            continue
        cal_mask = cal_groups == g
        if cal_mask.sum() > 0:
            group_logits, group_labels = cal_logits[cal_mask], cal_labels[cal_mask]
        else:  # empty calibration group: fall back to all calibration data
            group_logits, group_labels = cal_logits, cal_labels
        result = temp_scaling_calibrate(
            group_logits, group_labels, test_logits[test_mask],
            epochs=args.calibration_epochs, batch_size=args.calibration_batch_size,
            initial_tau=tau[g].item() if args.use_stage1_tau_init else 1.0,
            lr=args.temperature_scaling_lr)
        probs[test_mask] = torch.softmax(result['logits'], dim=1)
        temps[g] = result['tau']

    order = sorted(range(w_net.num_groups), key=lambda g: temps[g])
    remapped = test_groups.clone()
    for new_id, old_id in enumerate(order):
        remapped[test_groups == old_id] = new_id
    return probs, remapped, [temps[g] for g in order], temps


@torch.no_grad()
def extract_features_and_logits(model, dataset, device, batch_size):
    model.eval()
    feats, logits, labels = [], [], []
    for x, y in DataLoader(dataset, batch_size=batch_size, shuffle=False):
        f, l = model.forward_with_features(x.to(device))
        feats.append(f)
        logits.append(l)
        labels.append(y.to(device))
    return torch.cat(feats), torch.cat(logits), torch.cat(labels)


def old_new_classes(args, task):
    if task == 0:
        return list(range(args.base_classes)), []
    start = args.base_classes + (task - 1) * args.new_classes_per_task
    return list(range(start)), list(range(start, start + args.new_classes_per_task))


def run_mtcc_task(model, cal_x, cal_y, test_dataset, args, device, task,
                  previous_temps=None, previous_w_nets=None, seed=None, cached_splits=None):
    """One task of MT-CC.

    ``cal_x, cal_y`` is the calibration set (T-CIL adversarial samples by default). It
    is split per class: the first ``group_data_ratio`` of every class trains the router
    (Stage 1), the rest fits the group temperatures (Stage 2).

    ``cached_splits`` supplies group_train/cal/test features, logits and labels
    directly; model, calibration images and test_dataset can then be None.

    Returns a dict with ``probs`` / ``labels`` (CPU), ``groups`` (test assignments),
    ``temperatures`` (per partition, ascending), and ``init_temps`` / ``w_nets`` to
    pass as ``previous_*`` for the next task.
    """
    old_classes, new_classes = old_new_classes(args, task)

    bs = args.calibration_batch_size
    cache = getattr(args, 'cache', None)

    def extract(split, dataset):
        ident = cache.identity('extract', args, seed, task, split) if cache is not None else None
        hit = cache.load(ident) if ident is not None else None
        if hit is not None:
            return tuple(hit[k].to(device) for k in ('features', 'logits', 'labels'))
        out = extract_features_and_logits(model, dataset, device, bs)
        if ident is not None:
            cache.save(ident, dict(zip(('features', 'logits', 'labels'), out)))
        return out

    if cached_splits is None:
        group_idx, cal_idx = [], []
        for c in torch.unique(cal_y):
            idx = torch.where(cal_y == c)[0].tolist()
            split = int(len(idx) * args.group_data_ratio)
            group_idx += idx[:split]
            cal_idx += idx[split:]
        group = extract('group_train', TensorDataset(cal_x[group_idx], cal_y[group_idx]))
        cal = extract('cal', TensorDataset(cal_x[cal_idx], cal_y[cal_idx]))
        test = extract('test', test_dataset)
    else:
        group, cal, test = (tuple(cached_splits[split][k].to(device)
                                  for k in ('features', 'logits', 'labels'))
                            for split in ('group_train', 'cal', 'test'))

    # A cache hit skipped the forward passes / adversarial generation that a miss run
    # performs, so align the RNG stream here: router init and mini-batch order below
    # are then identical for hit and miss runs.
    if cache is not None and cache.enabled:
        rng_ident = cache.identity('rng', args, seed, task)
        state = cache.load(rng_ident)
        if state is not None:
            restore_rng(state)
        else:
            cache.save(rng_ident, snapshot_rng())

    bias = args.w_net_bias if args.w_net_bias is not None else args.dataset != 'cifar10'
    stage1 = []
    for p in range(args.num_partitions):
        if previous_temps is None:
            values = list(args.temp_init_values)
            values += [values[-1]] * (args.num_groups - len(values))
            initial_temps = torch.tensor(values[:args.num_groups], device=device)
        else:
            initial_temps = torch.tensor(previous_temps[min(p, len(previous_temps) - 1)], device=device)
        if previous_w_nets is not None and p < len(previous_w_nets):
            w_net = previous_w_nets[p]
        else:
            w_net = WNet(group[0].shape[1], args.num_classes, args.num_groups,
                         input_mode=args.w_net_input_mode,
                         normalize_inputs=args.normalize_w_net_inputs,
                         use_old_new_signal=args.use_old_new_signal, alpha=args.alpha,
                         first_task_old_ratio=args.first_task_old_ratio, bias=bias)
        w_net.set_old_new_classes(old_classes, new_classes)
        stage1.append(train_grouping_network(group[0], group[1], group[2], w_net, initial_temps, args))

    probs_list, groups_list, temps_list, pre_remap = [], [], [], []
    for tau, w_net in stage1:
        probs, groups, temps, raw_temps = group_temperature_scaling(w_net, tau, cal, test, args)
        probs_list.append(probs.cpu())
        groups_list.append(groups.cpu())
        temps_list.append(temps)
        pre_remap.append(raw_temps)

    if len(probs_list) > 1:
        probs = torch.stack(probs_list).mean(0)
        groups = torch.mode(torch.stack(groups_list), dim=0)[0]
    else:
        probs, groups = probs_list[0], groups_list[0]

    # Next-task initialisation: average the sorted temperatures over partitions, then
    # hand each partition the averages in its own group order.
    pre = np.array(pre_remap, dtype=float)
    avg_sorted = np.sort(pre, axis=1).mean(axis=0)
    init_temps = np.empty_like(pre)
    for j in range(pre.shape[0]):
        init_temps[j, np.argsort(pre[j])] = avg_sorted

    return {
        'probs': probs, 'labels': test[2].cpu(), 'groups': groups,
        'temperatures': temps_list, 'init_temps': init_temps,
        'w_nets': [w for _, w in stage1],
    }
