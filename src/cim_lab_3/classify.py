from __future__ import annotations

from pathlib import Path
import argparse
import torch
from torch import nn

from .data import build_loaders
from .models import build_model
from .run_io import load_run


def encode_dataset(encoder, loader, device):
    features, labels = [], []
    encoder.eval()
    with torch.inference_mode():
        for images, targets in loader:
            mu, _ = encoder.encode(images.to(device))
            features.append(mu.cpu())
            labels.append(targets)
    return torch.cat(features), torch.cat(labels)


def train_classifier(features, labels, latent_dim, device, epochs=50):
    classifier = nn.Linear(latent_dim, 10).to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-2)
    dataset = torch.utils.data.TensorDataset(features, labels)
    loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(classifier(x), y)
            loss.backward()
            optimizer.step()
    return classifier


def accuracy(classifier, features, labels, device):
    with torch.inference_mode():
        predictions = classifier(features.to(device)).argmax(1).cpu()
    return (predictions == labels).float().mean().item()


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args(argv)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device(device)
    cfg, checkpoint = load_run(args.run_dir, device)
    vae = build_model(cfg.model).to(device)
    vae.load_state_dict(checkpoint["model_state"])
    train_loader, test_loader = build_loaders(
        cfg.data_dir, 256, cfg.training.num_workers, None, None
    )

    
    vae.requires_grad_(False)
    ?, train_labels = encode_dataset(vae, train_loader, device)
     = encode_dataset(vae, test_loader, device)

    # Train the linear classifier on the frozen latent features. 
    # #Extension: you could save them out to make it faster to run multiple experiments.
    classifier = train_classifier(
        train_features, train_labels, int(cfg.model.latent_dim), device, args.epochs
    )

    # Calculate the train and test accuracy
    result = accuracy(classifier, ?, ?, device)

    # Extension: sampling from q(z|x) could yield augmented feature vectors.
    print(f"Frozen-encoder linear-classifier test accuracy: {result:.4f}")


if __name__ == "__main__":
    main()
