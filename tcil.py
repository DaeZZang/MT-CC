"""T-CIL: adversarial calibration samples for class-incremental learning.

Replay-buffer images are perturbed (FGSM) so that temperature scaling on them
mimics the temperature that the current task's validation data would give. The
perturbation size is found by bisection (``find_optimal_epsilon``).
"""

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader, TensorDataset


class AdversarialTrainer:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.model.eval()

    @torch.no_grad()
    def _class_means(self, loader):
        sums, counts = {}, {}
        for data, labels in loader:
            emb = self.model.get_features(data.to(self.device)).cpu()
            for e, y in zip(emb, labels.tolist()):
                if y in sums:
                    sums[y] += e
                    counts[y] += 1
                else:
                    sums[y] = e.clone()
                    counts[y] = 1
        return {y: sums[y] / counts[y] for y in sums}

    @torch.no_grad()
    def _target_labels(self, inputs, labels, num_classes, means, num_class_per_task):
        """Per sample: new-task classes -> most similar other class; old classes -> most
        dissimilar class (distance to the class mean embeddings)."""
        classes = sorted(means)
        mean_tensor = torch.stack([means[c] for c in classes]).to(self.device)
        threshold = num_classes - num_class_per_task
        emb = self.model.get_features(inputs)
        out = torch.zeros_like(labels)
        for i, (e, y) in enumerate(zip(emb, labels.tolist())):
            dist = torch.norm(mean_tensor - e.unsqueeze(0), dim=1)
            if y not in classes:
                out[i] = classes[dist.argmin().item()]
                continue
            j = classes.index(y)
            if y >= threshold:
                dist[j] = float('-inf')
                out[i] = classes[dist.argmax().item()]
            else:
                dist[j] = float('inf')
                out[i] = classes[dist.argmin().item()]
        return out

    def generate_adversarial_data(self, buffer_data, target_data, num_classes, batch_size,
                                  epsilon, num_class_per_task):
        """FGSM step of size ``epsilon`` on every sample of ``target_data``; the class
        mean embeddings come from ``buffer_data``. Returns CPU tensors."""
        buffer_loader = DataLoader(buffer_data, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        target_loader = DataLoader(target_data, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        means = self._class_means(buffer_loader)
        xs, ys = [], []
        for data, labels in target_loader:
            data = data.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            x = data.clone().detach().requires_grad_(True)
            target = self._target_labels(x, labels, num_classes, means, num_class_per_task)
            loss = -F.cross_entropy(self.model(x)[:, :num_classes], target)
            loss.backward()
            with torch.no_grad():
                x = x + epsilon * x.grad.sign()
            xs.append(x.detach().cpu())
            ys.append(labels.cpu())
        return torch.cat(xs), torch.cat(ys)

    def get_temperature(self, loader, num_classes, epochs, batch_size, lr=0.1):
        """Single temperature fitted by SGD on the cross-entropy of ``loader``."""
        self.model.eval()
        outputs, labels = [], []
        with torch.no_grad():
            for data, y in loader:
                outputs.append(self.model(data.to(self.device))[:, :num_classes].cpu())
                labels.append(y.cpu())
        outputs, labels = torch.cat(outputs), torch.cat(labels)

        temperature = torch.ones(1, requires_grad=True)
        optimizer = optim.SGD([temperature], lr=lr)
        scheduler = MultiStepLR(optimizer, milestones=[epochs // 2], gamma=0.1)
        fit_loader = DataLoader(TensorDataset(outputs, labels), batch_size=batch_size, shuffle=True)
        for _ in range(epochs):
            for out, y in fit_loader:
                loss = F.cross_entropy(out / temperature, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step()
        return temperature


def find_optimal_epsilon(trainer, buffer_data, new_task_data, target_temp, num_classes,
                         num_tasks, epochs, batch_size, lr=0.1, tolerance=1e-3):
    """Bisection on epsilon in [0, 1] until the temperature fitted on the perturbed
    new-task buffer samples matches ``target_temp`` (the validation-set temperature)."""
    lo, hi = 0.0, 1.0
    num_class_per_task = num_classes // num_tasks
    while hi - lo > tolerance:
        eps = (lo + hi) / 2.0
        adv, labels = trainer.generate_adversarial_data(
            buffer_data, new_task_data, num_classes, batch_size, eps, num_class_per_task)
        loader = DataLoader(TensorDataset(adv, labels), batch_size=batch_size, shuffle=True)
        temperature = trainer.get_temperature(loader, num_classes, epochs, batch_size, lr).item()
        if temperature < target_temp:
            lo = eps
        else:
            hi = eps
    return (lo + hi) / 2.0
