# MARGO

**Maritime Automatic Range Generation for Self-Correcting DNNs under Transient Faults**

MARGO is a two-phase automatic clipping-bound generation framework for self-correcting DNNs under transient hardware faults:

- **Phase 1 (GABO)**: derives a per-layer shrink ratio from the IQR dispersion of each ReLU layer's activation distribution, refined by distribution-aware statistics (skewness and entropy), producing a warm-start clipping bound and a safe search interval per layer.
- **Phase 2 (SAC)**: optimizes all layer-wise bounds jointly as a single decision vector using Soft Actor-Critic under fault injection, capturing inter-layer coupling.

## Repository layout

| File | Description |
|------|-------------|
| `config.py` | Hyperparameters (GABO / SAC / fault injection / data) |
| `fault_inject.py` | 32-bit fixed-point bit-flip fault injection |
| `gabo.py` | Phase 1: shrink-ratio computation and per-layer bound initialization |
| `sac.py` | Phase 2: SAC joint bound optimization (state/action/reward) |
| `model.py` | Three task models (trajectory / CRI / classification) |
| `requirements.txt` | Python dependencies |

## Datasets

This work uses three public AIS datasets:

- VTD (Vessel Traffic Data) - public AIS database maintained by the US Coast Guard
- DMA (Danish Maritime Authority) AIS dataset
- HawaiiCoast - curated by Sandia National Laboratories

## Requirements

- Python **3.9+** (3.10 recommended, as used in the paper)
- PyTorch 2.x (`pip install torch`)

## Installation

```bash
pip install -r requirements.txt
```

The core components (`config.py`, `fault_inject.py`, `gabo.py`, `sac.py`, `model.py`) can be imported directly; each module's docstring documents the core formulas.

## Citation

If this work is useful to you, please cite:

```bibtex
@article{...,
  title     = {MARGO: Maritime Automatic Range Generation for Self-Correcting DNNs under Transient Faults},
  author    = {...},
  ...
}
```

## License

MIT
