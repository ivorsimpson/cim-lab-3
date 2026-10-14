from __future__ import annotations

from pathlib import Path
import random

import hydra
from hydra.core.hydra_config import HydraConfig
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from .data import build_loaders
from .losses import vae_loss
from .models import build_model
from .run_io import save_run
from .visuals import save_reconstructions, save_samples


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def epoch(model, loader, device, cfg, optimizer=None):
    training = optimizer is not None
    model.train(training)
    totals = np.zeros(3, dtype=float)
    examples = 0
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for images, _ in loader:
            images = images.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits, mu, logvar = model(images)
            total, reconstruction, kl = vae_loss(
                logits, images, mu, logvar, cfg.beta, cfg.likelihood
            )
            if training:
                total.backward()
                optimizer.step()
            batch = images.size(0)
            totals += np.array([total.item(), reconstruction.item(), kl.item()]) * batch
            examples += batch
    return dict(zip(("loss", "reconstruction", "kl"), (totals / examples).tolist()))


@hydra.main(version_base=None, config_path="../../conf", config_name="mlp")
def main(cfg: DictConfig):
    set_seed(int(cfg.seed))
    run_dir = Path(HydraConfig.get().runtime.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device(device)
    train_loader, test_loader = build_loaders(
        cfg.data_dir,
        cfg.training.batch_size,
        cfg.training.num_workers,
        cfg.training.max_train_examples,
        cfg.training.max_test_examples,
    )
    model = build_model(cfg.model).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.training.learning_rate))

    wb = None
    if cfg.logging.use_wandb:
        import wandb
        wb = wandb.init(
            project=cfg.logging.project,
            entity=cfg.logging.entity,
            name=cfg.experiment_name,
            config=OmegaConf.to_container(cfg, resolve=True),
            dir=str(run_dir),
        )

    fixed_images, _ = next(iter(test_loader))
    fixed_images = fixed_images.to(device)
    latest = None
    for number in range(1, int(cfg.training.epochs) + 1):
        train_metrics = epoch(model, train_loader, device, cfg.training, optimizer)
        test_metrics = epoch(model, test_loader, device, cfg.training)
        latest = {f"train/{k}": v for k, v in train_metrics.items()}
        latest.update({f"test/{k}": v for k, v in test_metrics.items()})
        latest["epoch"] = number
        print(latest)

        if number % int(cfg.logging.image_every) == 0 or number == int(cfg.training.epochs):
            reconstruction_path = run_dir / f"reconstructions_epoch_{number:03d}.png"
            sample_path = run_dir / f"samples_epoch_{number:03d}.png"
            save_reconstructions(model, fixed_images, reconstruction_path, cfg.logging.n_images)
            save_samples(model, sample_path, device, cfg.logging.n_images)
            if wb is not None:
                import wandb
                media = {
                    "reconstructions": wandb.Image(str(reconstruction_path)),
                    "samples": wandb.Image(str(sample_path)),
                }
                wb.log({**latest, **media})
                continue
        if wb is not None:
            wb.log(latest)

    save_run(run_dir, cfg, model, int(cfg.training.epochs), latest)
    if wb is not None:
        import wandb
        artifact = wandb.Artifact(f"{cfg.experiment_name}-{wb.id}", type="model")
        artifact.add_file(str(run_dir / "checkpoint.pt"))
        artifact.add_file(str(run_dir / "resolved_config.yaml"))
        wb.log_artifact(artifact)
        wb.finish()
    print(f"Saved run to {run_dir}")


if __name__ == "__main__":
    main()
