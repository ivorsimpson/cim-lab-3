from __future__ import annotations

from pathlib import Path
import argparse
import torch

from .models import build_model
from .run_io import load_run
from .visuals import rasterise_latent_grid


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("--points", type=int, default=20)
    parser.add_argument("--extent", type=float, default=3.0)
    args = parser.parse_args(argv)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device(device)
    cfg, checkpoint = load_run(args.run_dir, device)
    model = build_model(cfg.model).to(device)
    model.load_state_dict(checkpoint["model_state"])

    path = rasterise_latent_grid(
        model, Path(args.run_dir) / "latent_grid.png", device, args.points, args.extent
    )
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
