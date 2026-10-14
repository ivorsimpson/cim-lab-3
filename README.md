# CIM Lab 3: Variational Autoencoders

Train and explore two variational autoencoders on FashionMNIST:

1. an MLP encoder-decoder;
2. a convolutional encoder-decoder.

The repository deliberately uses one complete Hydra file per experiment. Select it at startup with `--config-name`.

## Guidance

- [Lab 0 setup and WSL guidance](https://github.com/ivorsimpson/cim-lab-0/blob/main/README.md)
- [Lab 0 GitHub support](https://github.com/ivorsimpson/cim-lab-0/blob/main/docs/github-support.md)
- [Previous lab pattern](https://github.com/ivorsimpson/cim-lab-2/)
- [Torchvision FashionMNIST documentation](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.FashionMNIST.html)
- [Hydra command-line flags](https://hydra.cc/docs/advanced/hydra-command-line-flags/)

## Setup

```bash
uv sync
uv run pytest -q
```

W&B is enabled in the supplied experiment files. Log in with:

```bash
uv run wandb login
```

To run without W&B:

```bash
uv run python -m cim_lab_3.train --config-name mlp logging.use_wandb=false
```

## Task 1: MLP and Conv VAE

Open `src/cim_lab_3/models.py` and find **STUDENT TASK 1**. Complete a mirrored MLP and a Convolution encoder-decoder. The encoder must produce both:

```text
mu:     [batch, latent_dim]
logvar: [batch, latent_dim]
```

Run:

```bash
uv run python -m cim_lab_3.train --config-name mlp
```

```bash
uv run python -m cim_lab_3.train --config-name conv
```

## Task 2: choose the likelihood

In `conf/mlp.yaml` or `conf/conv.yaml`, select one of:

```text
bernoulli
gaussian
categorical_cross_entropy
student_t
```

Use the decoder outputs and pixel representation to decide which is appropriate. Inspect `reconstruction_nll()` in `losses.py`.

## Task 3: train and inspect

The run directory contains reconstruction and prior-sample grids for every epoch, plus the complete configuration and checkpoint. If W&B is enabled, loss curves, reconstructions, samples, and the final checkpoint are also logged.

Inspect:

- total loss;
- reconstruction negative log likelihood;
- KL divergence;
- reconstruction quality;
- prior sample quality.

Repeat using the convolutional VAE:

```bash
uv run python -m cim_lab_3.train --config-name conv
```

The channel pattern is already mirrored. Read the comments around `ConvTranspose2d` and verify every intermediate spatial shape.

## Task 4: a two-dimensional latent space

Override the latent dimension:

```bash
uv run python -m cim_lab_3.train --config-name conv \
  model.latent_dim=2 experiment_name=conv_vae_2d
```

Then rasterise a regular grid through the decoder:

```bash
uv run python -m cim_lab_3.latent_grid \
  outputs/conv_vae_2d/<timestamp>
```

Discuss which visual properties vary smoothly across the grid and whether the two coordinates have clear interpretations.

## Task 5: representation learning

Use the posterior mean as a frozen feature vector and train a linear classifier:

```bash
uv run python -m cim_lab_3.classify \
  outputs/conv_vae/<timestamp>
```

The encoder remains frozen. This tests whether the representation learned for reconstruction also supports class discrimination.

Extension: instead of using only the posterior mean, draw multiple samples from `q(z|x)` and investigate whether this acts as useful feature augmentation.

## Experiment configuration

Each YAML includes:

- `beta` for the beta-VAE objective;
- Adam learning rate;
- batch size;
- latent dimension;
- MLP hidden dimensions or convolutional channels;
- likelihood choice;
- epoch count;
- W&B settings.

Override any value directly, for example:

```bash
uv run python -m cim_lab_3.train --config-name conv \
  training.beta=4.0 model.latent_dim=8 experiment_name=conv_beta4
```

## Fast debug run

```bash
uv run python -m cim_lab_3.train --config-name mlp \
  training.epochs=1 \
  training.max_train_examples=512 \
  training.max_test_examples=128 \
  logging.use_wandb=false \
  experiment_name=debug
```

## Minimum completion

- [ ] MLP architecture completed and shape-tested.
- [ ] Likelihood selected and justified.
- [ ] MLP and convolutional VAEs trained.
- [ ] Reconstructions and prior samples inspected.
- [ ] Two-dimensional latent grid produced.
- [ ] Frozen latent representation evaluated with a linear classifier.
