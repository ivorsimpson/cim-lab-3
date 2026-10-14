from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import FashionMNIST
from torchvision.transforms import v2


def _limit(dataset, maximum):
    if maximum is None or maximum >= len(dataset):
        return dataset
    return Subset(dataset, range(int(maximum)))


def build_loaders(data_dir, batch_size, num_workers, max_train=None, max_test=None):
    transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
    train = FashionMNIST(data_dir, train=True, download=True, transform=transform)
    test = FashionMNIST(data_dir, train=False, download=True, transform=transform)
    train = _limit(train, max_train)
    test = _limit(test, max_test)
    kwargs = {
        "batch_size": int(batch_size),
        "num_workers": int(num_workers),
        "pin_memory": True,
        "persistent_workers": int(num_workers) > 0,
    }
    return (
        DataLoader(train, shuffle=True, **kwargs),
        DataLoader(test, shuffle=False, **kwargs),
    )
