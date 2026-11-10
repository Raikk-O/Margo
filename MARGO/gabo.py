"""Phase 1: GABO - IQR-based shrink-ratio computation and per-layer bound initialization.

Core formulas (Section 3.2.1):
    rho_shrink = clip( rho_max - (rho_max - rho_min) * clip(IQR/a_max, 0, 1)
                      - kappa_gamma * tanh(|gamma|/2)
                      - kappa_H * min(H/H0, 1),  rho_min, rho_max )
    U_init = rho_shrink * a_max
    lambda_init = min(U_init, a_max)
    [lambda_min, lambda_max] = [U_bar - sigma, min(U_bar + sigma, a_max)]
"""
from __future__ import annotations

import numpy as np
import torch

from config import GABOConfig


def positive_activations(acts: torch.Tensor) -> torch.Tensor:
    """Return strictly positive activations, excluding the ReLU zero mass."""
    return acts[acts > 0]


def feature_entropy(acts: torch.Tensor, n_bins: int) -> float:
    """Feature entropy H (normalized to [0,1] then binned)."""
    a = acts.detach().cpu().numpy().astype(np.float64)
    a_min, a_max = a.min(), a.max()
    a = (a - a_min) / (a_max - a_min + 1e-9)
    hist, _ = np.histogram(a, bins=n_bins, range=(0, 1))
    p = hist / hist.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p + 1e-12)).sum())


def compute_shrink_ratio(acts: torch.Tensor, cfg: GABOConfig) -> float:
    """Compute the shrink ratio rho_shrink from one layer's activation distribution."""
    ap = positive_activations(acts)
    a_max = float(acts.max())
    ap_np = ap.detach().cpu().numpy().astype(np.float64)
    iqr = float(np.percentile(ap_np, 75) - np.percentile(ap_np, 25))
    mean = float(acts.mean())
    std = float(acts.std()) + 1e-9
    gamma = float(((acts - mean) ** 3).mean() / (std ** 3))   # skewness
    H = feature_entropy(acts, cfg.n_bins)

    base = cfg.rho_max - (cfg.rho_max - cfg.rho_min) * np.clip(iqr / a_max, 0.0, 1.0)
    corr = cfg.kappa_gamma * np.tanh(abs(gamma) / 2.0) + cfg.kappa_H * min(H / cfg.H0, 1.0)
    rho = np.clip(base - corr, cfg.rho_min, cfg.rho_max)
    return float(rho)


def init_layer_bound(acts: torch.Tensor, cfg: GABOConfig) -> tuple[float, float, float]:
    """Return (lambda_init, lambda_min, lambda_max) for one layer."""
    a_max = float(acts.max())
    sigma = float(acts.std())
    rho = compute_shrink_ratio(acts, cfg)
    U_bar = min(rho * a_max, a_max)
    lam_init = U_bar
    lam_min = U_bar - sigma
    lam_max = min(U_bar + sigma, a_max)
    return lam_init, lam_min, lam_max


def find_relu_layers(model) -> list:
    """Return a list of (name, relu_module) for every ReLU layer in the model.

    This discovers the protected layers generically: for CRIMLP / ClassificationMLP
    it finds the five self.relus entries, and for MaritimeTrajModel it finds
    relu_proj / relu_fusion / relu_head.
    """
    return [(name, m) for name, m in model.named_modules()
            if isinstance(m, torch.nn.ReLU)]


def collect_layer_activations(model, calibration_loader) -> list:
    """Forward the calibration set and return the per-layer ReLU activations.

    Returns a list of tensors, one per ReLU layer (in find_relu_layers order).
    Each tensor has shape (N_cal, ...) and is moved to CPU.
    """
    relu_layers = find_relu_layers(model)
    acc = [[] for _ in relu_layers]

    def make_hook(idx: int):
        def hook(module, inp, out):
            acc[idx].append(out.detach().cpu())
        return hook

    hooks = [relu.register_forward_hook(make_hook(i))
             for i, (_, relu) in enumerate(relu_layers)]

    model.eval()
    with torch.no_grad():
        for batch in calibration_loader:
            x = batch[0] if isinstance(batch, (tuple, list)) else batch
            model(x)

    for h in hooks:
        h.remove()

    return [torch.cat(a, dim=0) for a in acc]


def gabo_warm_start(model, calibration_loader, cfg: GABOConfig) -> dict:
    """Run GABO initialization for every ReLU layer.

    Collects per-layer activations over the calibration set via forward hooks, then
    calls init_layer_bound for each layer.

    Returns {layer_idx: (lambda_init, lambda_min, lambda_max)}.
    """
    per_layer_acts = collect_layer_activations(model, calibration_loader)

    bounds = {}
    for i, acts in enumerate(per_layer_acts):
        bounds[i] = init_layer_bound(acts, cfg)

    return bounds


def apply_bounds(model, bounds: dict) -> None:
    """Clamp each protected ReLU layer to its clipping bound (for evaluation).

    A wrapper module replaces the ReLU forward so that activations above the bound
    are clipped to the bound (upper-bound clipping). This mutates the model in place.
    """
    relu_layers = find_relu_layers(model)

    class ClampedReLU(torch.nn.Module):
        def __init__(self, bound: float):
            super().__init__()
            self.bound = bound

        def forward(self, x):
            return torch.clamp(x, min=0.0, max=self.bound)

    for i, (name, relu) in enumerate(relu_layers):
        bound = bounds[i][0] if isinstance(bounds[i], tuple) else bounds[i]
        parent_name, _, attr = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, attr, ClampedReLU(bound))
