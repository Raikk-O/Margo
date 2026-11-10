"""32-bit fixed-point fault injection (bit flips).

Fault model (paper Section 2.2):
  - Each parameter v is encoded in two's-complement fixed point:
        v = 2^{-16} * ( sum_{i=0}^{n-2} b_i * 2^i  -  b_{n-1} * 2^{n-1} )
  - Injection flips one bit: b'_k = 1 - b_k
  - Per layer l, the number of flipped bits is:
        N_fault^(l) = floor(|W_l| * n_bits * BER)
  - Parameter coordinates and bit positions are sampled uniformly at random
  - If N_fault^(l) == 0 (small layer at low BER), no fault is injected into that layer
  - Corrupted bits persist for one inference run, then are discarded
  - Fixed-point encoding never produces NaN / inf, so no special handling is needed
"""
from __future__ import annotations

import copy

import numpy as np
import torch

from config import FaultConfig


# ---------------------------------------------------------------- fixed-point encoding

def float_to_fixed(x: torch.Tensor, cfg: FaultConfig) -> np.ndarray:
    """Quantize a float tensor to Q15.16 fixed-point integers (numpy int32).

    The scale is 2^frac_bits; the result is the two's-complement integer whose
    bit pattern represents the fixed-point value.
    """
    scale = 1 << cfg.frac_bits
    return torch.round(x.detach().cpu() * scale).to(torch.int32).numpy()


def fixed_to_float(q: np.ndarray, cfg: FaultConfig) -> torch.Tensor:
    """Convert fixed-point integers back to a float tensor."""
    scale = 1 << cfg.frac_bits
    return torch.from_numpy(q.astype(np.float32) / scale)


def as_uint32(q: np.ndarray) -> np.ndarray:
    """Reinterpret an int32 array as raw uint32 bits (for bit-level inspection)."""
    return q.view(np.uint32)


def as_int32(bits: np.ndarray) -> np.ndarray:
    """Reinterpret raw uint32 bits as int32 (two's-complement)."""
    return bits.view(np.int32)


# ---------------------------------------------------------------- fault count & sampling

def compute_fault_count(n_params: int, cfg: FaultConfig) -> int:
    """Expected number of flipped bits for a tensor with n_params parameters."""
    return int(np.floor(n_params * cfg.total_bits * cfg.ber))


def sample_bit_positions(n_params: int, n_faults: int,
                         cfg: FaultConfig, rng: np.random.Generator):
    """Sample (parameter index, bit index) pairs uniformly at random."""
    param_idx = rng.integers(0, n_params, size=n_faults)
    bit_idx = rng.integers(0, cfg.total_bits, size=n_faults)
    return param_idx, bit_idx


def flip_bits(q: np.ndarray, param_idx: np.ndarray, bit_idx: np.ndarray) -> np.ndarray:
    """Flip the selected bits of the quantized tensor; returns a new tensor.

    Flipping bit k inverts it via XOR with 2^k, which is valid for two's-complement
    (the sign bit, bit 31, is handled identically).
    """
    flat = q.ravel().copy().astype(np.int32)
    mask = (1 << bit_idx).astype(np.int32)   # bit 31 wraps to the sign bit correctly
    flat[param_idx] ^= mask
    return flat.reshape(q.shape)


# ---------------------------------------------------------------- main injection

def inject_bitflips(state_dict: dict, cfg: FaultConfig, seed=None):
    """Apply random bit flips to model parameters; return a new state_dict.

    The original state_dict is left untouched (deep copy), so the pristine model
    (unprotected baseline) stays intact. Returns the injected state_dict.
    """
    rng = np.random.default_rng(seed)
    injected = copy.deepcopy(state_dict)
    stats = {}

    for name, param in injected.items():
        if not param.dtype.is_floating_point:
            continue

        q = float_to_fixed(param, cfg)              # int32 numpy
        n_params = q.size
        n_faults = compute_fault_count(n_params, cfg)
        if n_faults == 0:
            continue

        param_idx, bit_idx = sample_bit_positions(n_params, n_faults, cfg, rng)
        q_flipped = flip_bits(q, param_idx, bit_idx)
        injected[name] = fixed_to_float(q_flipped, cfg)
        stats[name] = n_faults

    return injected


# ---------------------------------------------------------------- convenience wrapper

class FaultInjector:
    """Convenience wrapper around inject_bitflips that carries a FaultConfig.

    Usage:
        injector = FaultInjector(FaultConfig(ber=5e-5))
        faulty_sd = injector(model.state_dict(), seed=0)
    """
    def __init__(self, cfg: FaultConfig):
        self.cfg = cfg

    def __call__(self, state_dict: dict, seed=None) -> dict:
        return inject_bitflips(state_dict, self.cfg, seed=seed)

    def inject_model(self, model, seed=None) -> dict:
        """Return a fault-injected copy of the model's state_dict."""
        return inject_bitflips(model.state_dict(), self.cfg, seed=seed)


# ---------------------------------------------------------------- minimal self-check

if __name__ == "__main__":
    cfg = FaultConfig(ber=1e-4)
    w = torch.randn(64, 32)
    q = float_to_fixed(w, cfg)
    n_faults = compute_fault_count(q.size, cfg)
    print(f"params={q.size}, faults={n_faults}")
    print(f"quantized range: [{q.min()}, {q.max()}]  (int32, no NaN/inf by construction)")
    print(f"bit-width check: total_bits={cfg.total_bits}, frac_bits={cfg.frac_bits}")
