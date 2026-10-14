from __future__ import annotations

import json
from pathlib import Path

import torch
from omegaconf import OmegaConf


def save_run(run_dir, cfg, model, epoch, metrics):
    run_dir = Path(run_dir)
    OmegaConf.save(cfg, run_dir / "resolved_config.yaml", resolve=True)
    torch.save(
        {"model_state": model.state_dict(), "epoch": epoch, "metrics": metrics},
        run_dir / "checkpoint.pt",
    )
    (run_dir / "summary.json").write_text(json.dumps(metrics, indent=2) + "\n")


def load_run(run_dir, device):
    run_dir = Path(run_dir)
    cfg = OmegaConf.load(run_dir / "resolved_config.yaml")
    checkpoint = torch.load(run_dir / "checkpoint.pt", map_location=device)
    return cfg, checkpoint
