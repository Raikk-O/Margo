# DNN-SPECTRE-SDN / MARGO — Public Reproducibility Release

This repository releases the implementation required to reproduce the fault-tolerance
experiments reported in the paper. It contains the fixed-point fault-injection procedure,
the GABO bound update, the SAC reward calculation, and the scripts used for the main
comparisons.

## Directory layout

```
release_public/
├── src/rrelu/                              # core implementation
│   ├── setup.py                            #   replace_act (used by the main scripts)
│   ├── pytorchfi/                          #   fixed-point fault injection (BFWP)
│   ├── relu_bound/                         #   bounded ReLU modules
│   └── search_bound/
│       ├── ranger.py                       #   Ranger baseline
│       ├── gabo_reg.py                     #   GABO bound update
│       ├── RL-SAC/gabo_DMA_SAC.py          #   SAC reward + two-phase optimisation (DMA)
│       ├── RL-SAC/gabo_reg_sac.py          #   SAC reward (multi-dataset regression variant)
│       ├── GABO_XR_Shrink_iqr/
│       │   └── gabo_dma_sac_ablation.py    #   GABO-SAC (DMA) — used by --search_bound gabo_sac
├── experiments/
│   ├── trajectory/
│   │   ├── fault_test_brelu_traj_dma_gabo_XR.py   # DMA trajectory main experiment
│   │   ├── fault_test_brelu_traj_gabo_XR.py       # multi-dataset trajectory experiment
│   │   └── ProcessAisDMA.py                        # DMA data preprocessing
│   └── multiclass/                         #   anomaly-classification main experiment
├── config/repro_config.yaml                #   seeds / injection counts / budgets / BER / samples
├── docs/                                   #   reviewer mapping, appendix pseudocode
├── requirements.txt
├── pyproject.toml
├── LICENSE
└── .gitignore
```

## Environment

Python 3.10–3.12 is required (do not use 3.13/3.15; `zstandard` has no wheel there).

```bash
uv venv --python 3.12
uv pip install -r requirements.txt
```

## Mapping to the reviewer's requirements

| Requirement | Implementation | Entry point |
|---|---|---|
| Fixed-point fault injection | `src/rrelu/pytorchfi/{core,weight_error_models}.py` | `src/rrelu/pytorchfi/` |
| GABO bound update | `src/rrelu/search_bound/gabo_reg.py` | `--search_bound gabo` |
| SAC reward calculation | `src/rrelu/search_bound/GABO_XR_Shrink_iqr/gabo_dma_sac_ablation.py` | `--search_bound gabo_sac` |
| Ranger baseline | `src/rrelu/search_bound/ranger.py` | `--search_bound ranger` |
| Main comparison scripts | `experiments/trajectory/`, `experiments/multiclass/` | `fault_test_brelu_traj_dma_gabo_XR.py` |

All hyper-parameters (random seeds, fault-injection counts, optimization budgets, BER,
calibration sample counts) are consolidated in `config/repro_config.yaml` with source
line references. Detailed pseudocode is in `docs/APPENDIX_ALGORITHMS.md`; the mapping
from each reviewer comment to the code is in `docs/REVIEWER_MAPPING.md`.
