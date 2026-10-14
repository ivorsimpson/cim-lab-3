from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.utils import make_grid, save_image


def save_reconstructions(model, images, path: Path, n=16):
    model.eval()
    with torch.inference_mode():
        logits, _, _ = model(images[:n])
        recon = torch.sigmoid(logits)
    comparison = torch.cat([images[:n], recon])
    save_image(comparison, path, nrow=n, padding=2)
    return path


def save_samples(model, path: Path, device, n=16):
    model.eval()
    with torch.inference_mode():
        samples = model.sample(n, device)
    save_image(samples, path, nrow=int(np.sqrt(n)), padding=2)
    return path


def rasterise_latent_grid(model, path: Path, device, points=20, extent=3.0):
    if model.latent_dim != 2:
        raise ValueError("Latent rasterisation requires model.latent_dim == 2")
    coordinates = torch.linspace(-extent, extent, points, device=device)
    # Top row should correspond to positive vertical latent values.
    yy, xx = torch.meshgrid(coordinates.flip(0), coordinates, indexing="ij")
    z = torch.stack([xx.flatten(), yy.flatten()], dim=1)
    # turn off the gradiennts
    model.eval()
    with torch.inference_mode():
        images = torch.sigmoid(model.decode(z))
    grid = make_grid(images, nrow=points, padding=1, pad_value=1.0)
    save_image(grid, path)
    return path
