from __future__ import annotations

import torch
from torch import nn


class BaseVAE(nn.Module):
    def reparameterise(self, mu, logvar):
        if not self.training:
            return mu
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterise(mu, logvar)
        return self.decode(z), mu, logvar

    def sample(self, n, device):
        z = torch.randn(n, self.latent_dim, device=device)
        return torch.sigmoid(self.decode(z))


class MLPVAE(BaseVAE):
    def __init__(self, latent_dim: int, hidden_dims: list[int]):
        super().__init__()
        self.latent_dim = latent_dim
        h1, h2 = hidden_dims

        # STUDENT TASK 1: add your own MLP encoder,
        # posterior heads, and mirrored decoder.
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear,
            nn.ReLU(),
            nn.Linear,
            nn.ReLU(),
        )
        self.mu = nn.Linear(h2, latent_dim)
        self.logvar = nn.Linear(h2, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, h2),
            nn.ReLU(),
            nn.Linear,
            nn.ReLU(),
            nn.Linear(, 28 * 28),
        )

    def encode(self, x):
        features = self.encoder(x)
        return self.mu(features), self.logvar(features)

    def decode(self, z):
        return self.decoder(z).view(-1, 1, 28, 28)


class ConvVAE(BaseVAE):
    def __init__(self, latent_dim: int, channels: list[int]):
        super().__init__()
        self.latent_dim = latent_dim
        c1, c2 = channels
        self.encoder = nn.Sequential(
            #   STUDENT TASK 1: add your own conv encoder,
            # posterior heads, and mirrored decoder.
            nn.Conv2d(?, ?, kernel_size=?, stride=2, padding=1),  # 28 -> 14
            nn.ReLU(),
            nn.Conv2d(?, stride=2, padding=1),  # 14 -> 7
            nn.ReLU(),
            nn.Flatten(),
        )
        self.mu = nn.Linear(c2 * 7 * 7, latent_dim)
        self.logvar = nn.Linear(c2 * 7 * 7, latent_dim)
        self.decoder_input = nn.Linear(latent_dim, c2 * 7 * 7)
        self.decoder = nn.Sequential(
            # ConvTranspose2d learns an upsampling operation. Output size depends
            # on kernel, stride, padding and output_padding, so check shapes carefully.
            nn.ConvTranspose2d(c2, ?, kernel_size=4, stride=2, padding=1),  # 7 -> 14
            nn.ReLU(),
            nn.ConvTranspose2d(c1, 1, kernel_size=4, stride=2, padding=1),  # 14 -> 28
        )
        self.c2 = c2

    def encode(self, x):
        features = self.encoder(x)
        return self.mu(features), self.logvar(features)

    def decode(self, z):
        features = self.decoder_input(z).view(-1, self.c2, 7, 7)
        return self.decoder(features)


def build_model(cfg):
    if cfg.type == "mlp":
        return MLPVAE(int(cfg.latent_dim), list(cfg.hidden_dims))
    if cfg.type == "conv":
        return ConvVAE(int(cfg.latent_dim), list(cfg.channels))
    raise ValueError(f"Unknown model type: {cfg.type}")
