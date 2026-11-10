"""Three task models for MARGO.

- MaritimeTrajModel  : trajectory prediction (multi-scale attention network)
- CRIMLP             : CRI prediction (fully-connected MLP)
- ClassificationMLP  : vessel anomaly type classification (fully-connected MLP)

Protected ReLU layers:
  trajectory model: 3 (relu_proj / relu_fusion / relu_head)
  CRI and classification: 5 each (self.relus)
MARGO clips these ReLU layers per layer.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


# ---------------------------------------------------------------- basic modules

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class ScaleAttention(nn.Module):
    """A single-scale attention module.

    `scale` is the temporal downsampling stride: the sequence is chunked/aggregated by
    this stride before self-attention, modeling a coarser time resolution.
    """
    def __init__(self, d_model: int, n_heads: int, scale: int):
        super().__init__()
        self.scale = scale
        self.down = nn.Conv1d(d_model, d_model, kernel_size=scale, stride=scale, padding=0)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.up = nn.Upsample(scale_factor=scale, mode="nearest")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        x_t = x.transpose(1, 2)                 # (B, D, T)
        x_t = self.down(x_t)                    # (B, D, T//scale)
        x_t = x_t.transpose(1, 2)               # (B, T//scale, D)
        attn_out, _ = self.attn(x_t, x_t, x_t)  # self-attention
        attn_out = attn_out.transpose(1, 2)     # (B, D, T//scale)
        attn_out = self.up(attn_out)[:, :, :T]  # upsample back to (B, D, T)
        return attn_out.transpose(1, 2)         # (B, T, D)


# ---------------------------------------------------------------- trajectory model

class MaritimeTrajModel(nn.Module):
    def __init__(
        self,
        input_dim: int = 4,          # SOG, COG, dlat, dlon
        hist_steps: int = 18,        # historical steps
        pred_steps: int = 24,        # future steps (4 hours)
        d_model: int = 256,
        n_heads: int = 8,
        ff_dim: int = 1024,
        n_enc_layers: int = 2,
        scales: tuple = (1, 4, 16, 32),
        pred_hidden: int = 256,
    ):
        super().__init__()
        self.d_model = d_model
        self.pred_steps = pred_steps

        # encoder input projection (1st protected ReLU: projection)
        self.input_proj = nn.Linear(input_dim, d_model)
        self.relu_proj = nn.ReLU()

        # 4 scale-specific modules
        self.pos_encs = nn.ModuleList([PositionalEncoding(d_model) for _ in scales])
        self.scale_attns = nn.ModuleList([ScaleAttention(d_model, n_heads, s) for s in scales])

        # cross-attention fusion (2nd protected ReLU: fusion)
        self.fusion_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.fusion_mlp = nn.Linear(d_model * len(scales), d_model)
        self.relu_fusion = nn.ReLU()

        # 2-layer Transformer encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=ff_dim,
            batch_first=True, activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_enc_layers)

        # prediction head (3rd protected ReLU: prediction-head MLP)
        self.head_mlp = nn.Linear(d_model, pred_hidden)
        self.relu_head = nn.ReLU()
        self.output = nn.Linear(pred_hidden, pred_steps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, hist_steps, input_dim) -> (B, pred_steps)"""
        h = self.relu_proj(self.input_proj(x))       # (B, T, d_model)

        scale_feats = []
        for pos_enc, scale_attn in zip(self.pos_encs, self.scale_attns):
            scale_feats.append(scale_attn(pos_enc(h)))

        cat = torch.cat(scale_feats, dim=-1)              # (B, T, d_model*4)
        fused = self.relu_fusion(self.fusion_mlp(cat))    # (B, T, d_model)

        enc = self.transformer(fused)                     # (B, T, d_model)

        pooled = enc.mean(dim=1)                          # (B, d_model)
        out = self.relu_head(self.head_mlp(pooled))
        return self.output(out)                           # (B, pred_steps)


# ---------------------------------------------------------------- CRI model

class CRIMLP(nn.Module):
    """CRI prediction model: fully-connected MLP.

    Paper appendix: 7 input features, hidden layers 256 -> 128 -> 64 -> 32 -> 16,
    scalar output. 5 protected ReLU layers, 45,825 parameters.
    """
    def __init__(self, input_dim: int = 7, hidden_dims: tuple = (256, 128, 64, 32, 16)):
        super().__init__()
        self.relus = nn.ModuleList()   # 5 protected ReLU layers
        self.linears = nn.ModuleList()
        in_dim = input_dim
        for h in hidden_dims:
            self.linears.append(nn.Linear(in_dim, h))
            self.relus.append(nn.ReLU())
            in_dim = h
        self.output = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, input_dim) -> (B, 1)"""
        for lin, relu in zip(self.linears, self.relus):
            x = relu(lin(x))
        return self.output(x)


# ---------------------------------------------------------------- classification model

class ClassificationMLP(nn.Module):
    """Vessel anomaly type classification model: fully-connected MLP.

    Paper appendix: hidden layers [256, 192, 128, 64, 32], 4-class output.
    5 protected ReLU layers. Set input_dim to your actual feature dimension.
    """
    def __init__(self, input_dim: int = 8, hidden_dims: tuple = (256, 192, 128, 64, 32),
                 n_classes: int = 4):
        super().__init__()
        self.relus = nn.ModuleList()   # 5 protected ReLU layers
        self.linears = nn.ModuleList()
        in_dim = input_dim
        for h in hidden_dims:
            self.linears.append(nn.Linear(in_dim, h))
            self.relus.append(nn.ReLU())
            in_dim = h
        self.output = nn.Linear(in_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, input_dim) -> (B, n_classes)"""
        for lin, relu in zip(self.linears, self.relus):
            x = relu(lin(x))
        return self.output(x)


# ---------------------------------------------------------------- usage example

if __name__ == "__main__":
    traj = MaritimeTrajModel()
    x = torch.randn(4, 18, 4)
    print("Trajectory input:", x.shape, "-> output:", traj(x).shape)  # (4, 24)
    print("Trajectory params: %.2f M" % (sum(p.numel() for p in traj.parameters()) / 1e6))

    cri = CRIMLP()
    xc = torch.randn(4, 7)
    print("CRI input:", xc.shape, "-> output:", cri(xc).shape)  # (4, 1)
    print("CRI params:", sum(p.numel() for p in cri.parameters()))  # 45825

    cls = ClassificationMLP()
    xk = torch.randn(4, 8)
    print("Classification input:", xk.shape, "-> output:", cls(xk).shape)  # (4, 4)
    print("Classification params: %.0f K" % (sum(p.numel() for p in cls.parameters()) / 1e3))
