"""Single-temperature scaling fitted with SGD (T-CIL style)."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader, TensorDataset


def calibrate(train_logits, train_labels, test_logits, epochs=100, batch_size=128,
              initial_tau=1.0, lr=0.1):
    """Fit one temperature on ``train_logits`` and return ``{'tau', 'logits'}`` with the
    scaled ``test_logits``. Everything stays on the device of the inputs."""
    device = train_logits.device
    temperature = torch.tensor([initial_tau], requires_grad=True, device=device)
    optimizer = optim.SGD([temperature], lr=lr)
    scheduler = MultiStepLR(optimizer, milestones=[epochs // 2], gamma=0.1)
    loader = DataLoader(TensorDataset(train_logits, train_labels), batch_size=batch_size, shuffle=True)
    for _ in range(epochs):
        for out, y in loader:
            loss = F.cross_entropy(out / temperature, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
    tau = temperature.item()
    if tau != tau:  # NaN guard
        tau = 1.0
    return {'tau': tau, 'logits': test_logits / tau}


def validation_temperature(model, valid_data, num_classes, epochs, batch_size, device):
    """Temperature of the current task's validation set (the T-CIL target), fitted on
    ``log(tau)`` so that it stays positive."""
    model.eval()
    loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    outputs, labels = [], []
    with torch.no_grad():
        for data, y in loader:
            outputs.append(model(data.to(device, non_blocking=True))[:, :num_classes].cpu())
            labels.append(y)
    outputs, labels = torch.cat(outputs), torch.cat(labels)

    log_temperature = torch.tensor([0.0], requires_grad=True)
    optimizer = optim.SGD([log_temperature], lr=0.1)
    scheduler = MultiStepLR(optimizer, milestones=[50], gamma=0.1)
    criterion = nn.CrossEntropyLoss()
    fit_loader = DataLoader(TensorDataset(outputs, labels), batch_size=batch_size, shuffle=True)
    for _ in range(epochs):
        for out, y in fit_loader:
            loss = criterion(out / torch.exp(log_temperature), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
    return torch.exp(log_temperature).item()
