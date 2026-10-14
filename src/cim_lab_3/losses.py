from __future__ import annotations

import torch
import torch.nn.functional as F

VALID_LIKELIHOODS = {
    "bernoulli",
    "gaussian",
    "categorical_cross_entropy",
    "student_t",
}


def reconstruction_nll(predictions, targets, likelihood: str):
    """Negative log likelihood, summed per image then averaged over the batch."""
    if likelihood not in VALID_LIKELIHOODS:
        raise ValueError(f"Choose one of {sorted(VALID_LIKELIHOODS)}")

    # STUDENT TASK 2: decide which likelihood is appropriate for grayscale
    # FashionMNIST pixels scaled to [0, 1]. 
    if likelihood == "bernoulli":
        return F.binary_cross_entropy_with_logits(predictions, targets, reduction="sum") / targets.size(0)
    if likelihood == "gaussian":
        return F.gaussian_nll_loss(predictions, targets, var=1.0, reduction="sum") / targets.size(0)

    if likelihood == "categorical_cross_entropy":
        raise ValueError(
            "Categorical cross entropy models mutually exclusive classes, not continuous pixel intensities."
        )
    if likelihood == "student_t":
        raise NotImplementedError(
            "A Student-t likelihood needs a location, scale and degrees of freedom; this decoder only predicts logits."
        )
    raise AssertionError("unreachable")


def kl_divergence(mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu.square() - logvar.exp()) / mu.size(0)


def vae_loss(logits, targets, mu, logvar, beta, likelihood):
    reconstruction = reconstruction_nll(logits, targets, likelihood)
    kl = kl_divergence(mu, logvar)
    return reconstruction + float(beta) * kl, reconstruction, kl
