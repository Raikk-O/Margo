"""MARGO hyperparameters.

All values follow the appendix tables of the paper
(GABO parameters / Implementation settings).
"""
from dataclasses import dataclass


@dataclass
class GABOConfig:
    rho_min: float = 0.7        # lower bound of the shrink ratio
    rho_max: float = 0.95       # upper bound of the shrink ratio
    kappa_gamma: float = 0.10   # skewness correction coefficient tanh(|gamma|/2)
    kappa_H: float = 0.10       # entropy correction coefficient
    H0: float = 8.0             # entropy normalization constant (= log2 256)
    n_bins: int = 256           # number of histogram bins for feature entropy


@dataclass
class SACConfig:
    lr: float = 1e-3            # SAC learning rate
    episodes: int = 50          # number of training episodes
    beta: float = 0.5           # action scaling factor
    w_pen: float = 1.0          # clean-performance penalty weight
    n_injections: int = 3       # fault injections per reward evaluation (tunable)
    tau: float = 0.05           # clean tolerance


@dataclass
class FaultConfig:
    total_bits: int = 32        # fixed-point bit width
    frac_bits: int = 16         # fractional bits
    int_bits: int = 15          # integer bits
    ber: float = 5e-5           # target bit error rate


@dataclass
class DataConfig:
    calibration_samples: int = 2048   # calibration samples for GABO stats / SAC reward
    random_seed: int = 0              # fixed seed (data split / injection / SAC training)
