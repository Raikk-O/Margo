
import os
import sys
import math
import copy
import random
import argparse
import warnings
import csv
import time
import json

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, '..', '..'))
for _p in (_PROJECT_ROOT, os.path.join(_PROJECT_ROOT, 'src')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import importlib.util
import ast
from typing import Dict

warnings.filterwarnings("ignore")

import numpy as np
if not hasattr(np, 'trapz'):
    np.trapz = np.trapezoid
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset, TensorDataset
from tqdm import tqdm
from fxpmath import Fxp

from src.rrelu.pytorchfi.core import FaultInjection
from src.rrelu.pytorchfi.weight_error_models import (
    multi_weight_inj_fixed, multi_weight_inj_float, multi_weight_inj_int
)
from src.rrelu.relu_bound.bound_zero import bounded_relu_zero
from src.rrelu.relu_bound.bound_tresh import bounded_relu_tresh
from src.rrelu.relu_bound.bound_proact import bounded_hyrelu_proact

Relu_bound = next(c for c in bounded_relu_zero.__mro__ if c.__name__ == 'Relu_bound')


def _load_rrelu_module(mod_name, rel_path):
    
    file_path = os.path.join(_PROJECT_ROOT, 'src', 'rrelu', rel_path)
    spec = importlib.util.spec_from_file_location(mod_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


Ranger_bounds = _load_rrelu_module('rrelu_ranger', 'search_bound/ranger.py').Ranger_bounds
GABO_reg_bounds = _load_rrelu_module('rrelu_gabo_reg', 'search_bound/gabo_reg.py').GABO_bounds
GABO_reg_ablation_bounds = _load_rrelu_module('rrelu_gabo_reg_ablation',
                                              'search_bound/GABO_XR/gabo_reg_ablation.py').GABO_bounds
SAC_bounds = _load_rrelu_module('rrelu_gabo_reg_sac', 'search_bound/RL-SAC/gabo_reg_sac.py').SAC_bounds
SAC_ablation_bounds = _load_rrelu_module('rrelu_gabo_reg_sac_ablation',
                                         'search_bound/GABO_XR/gabo_reg_sac_ablation.py').SAC_bounds

USE_CUDA = torch.cuda.is_available()
if not USE_CUDA:
    torch.Tensor.cuda = lambda self, *a, **k: self
    torch.nn.Module.cuda = lambda self, *a, **k: self
    _orig_randint = torch.randint

    def _randint_cpu(*args, **kwargs):
        if kwargs.get('device') == 'cuda':
            kwargs['device'] = 'cpu'
        return _orig_randint(*args, **kwargs)

    torch.randint = _randint_cpu

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DATASETS = {
    "dma": {
        "csv": os.path.join(SCRIPT_DIR, '..', 'DMA', 'aisdk-2026-08-26', 'windows.csv'),
        "model": os.path.join(SCRIPT_DIR, 'best_trajectory_model.pth'),
        "outdir": "results",
        "desc": "DMA (600s 采样, 预测 4 小时)",
    },
    "dma_new": {
        "csv": os.path.join(SCRIPT_DIR, 'data', 'DMA', 'windows_dma_600s.csv'),
        "model": os.path.join(SCRIPT_DIR, 'best_trajectory_model_dma_new.pth'),
        "outdir": "results_dma_new",
        "desc": "DMA 新方案 (相对位移 dLat/dLon, 600s, 3天)",
    },
    "pireaus": {
        "csv": os.path.join(SCRIPT_DIR, 'data', 'PireausAIS', 'windows_pireaus_60s.csv'),
        "model": os.path.join(SCRIPT_DIR, 'best_trajectory_model_pireaus.pth'),
        "outdir": "results_pireaus",
        "desc": "PireausAIS 清洗版 (20s 采样, 预测 8 分钟, 已过滤停船)",
    },
    "american": {
        "csv": os.path.join(SCRIPT_DIR, 'data', 'AmericanAis', 'windows_america_600s.csv'),
        "model": os.path.join(SCRIPT_DIR, 'best_trajectory_model_american.pth'),
        "outdir": "results_american",
        "desc": "AmericanAIS 600s (10分钟采样, 预测 4 小时, 全15天)",
    },
    "houston": {
        "csv": os.path.join(SCRIPT_DIR, 'data', 'AmericanAis', 'houston_600s.csv'),
        "model": os.path.join(SCRIPT_DIR, 'best_trajectory_model_houston.pth'),
        "outdir": "results_houston",
        "desc": "AmericanAIS 休斯顿港框定 (600s 采样, 预测 4 小时, 全15天)",
    },
    "losangeles": {
        "csv": os.path.join(SCRIPT_DIR, 'data', 'AmericanAis', 'losangeles_600s.csv'),
        "model": os.path.join(SCRIPT_DIR, 'best_trajectory_model_losangeles.pth'),
        "outdir": "results_losangeles",
        "desc": "AmericanAIS 洛杉矶/长滩港框定 (600s 采样, 预测 4 小时, 全15天)",
    },
    "newyork": {
        "csv": os.path.join(SCRIPT_DIR, 'data', 'AmericanAis', 'newyork_600s.csv'),
        "model": os.path.join(SCRIPT_DIR, 'best_trajectory_model_newyork.pth'),
        "outdir": "results_newyork",
        "desc": "AmericanAIS 纽约/新泽西港框定 (600s 采样, 预测 4 小时, 全15天)",
    },
    "neworleans": {
        "csv": os.path.join(SCRIPT_DIR, 'data', 'AmericanAis', 'neworleans_600s.csv'),
        "model": os.path.join(SCRIPT_DIR, 'best_trajectory_model_neworleans.pth'),
        "outdir": "results_neworleans",
        "desc": "AmericanAIS 新奥尔良港框定 (600s 采样, 预测 4 小时, 全15天)",
    },
}
DEFAULT_DATASET = "dma"

WINDOWS_CSV = DATASETS[DEFAULT_DATASET]["csv"]
MODEL_PATH = DATASETS[DEFAULT_DATASET]["model"]


class Config:
    HIST_LEN = 18
    PRED_LEN = 24
    TOTAL_LEN = HIST_LEN + PRED_LEN
    FEAT_DIM = 4
    MAX_SOG = 30.0
    EARTH_RADIUS_NM = 3440.065
    FEAT_MIN = None
    FEAT_MAX = None
    D_MODEL = 256
    N_HEADS = 8
    N_LAYERS = 2
    SCALES = [1, 4, 16, 32]
    BATCH_SIZE = 64
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


METRIC_KEYS = ["ADE_nm", "FDE_nm", "MSE_SOG", "MAE_SOG", "MSE_COG", "MAE_COG"]


class TrajectoryDataset(Dataset):
    def __init__(self, csv_path):
        print(f"[Data] loading: {csv_path}")
        self.df = pd.read_csv(csv_path)
        num_samples = len(self.df) // Config.TOTAL_LEN
        self.df = self.df.iloc[:num_samples * Config.TOTAL_LEN].reset_index(drop=True)
        self.num_samples = num_samples
        self.seg_ids = self.df['segment_id'].values[::Config.TOTAL_LEN]
        norm_json = os.path.splitext(csv_path)[0] + "_norm.json"
        with open(norm_json, "r", encoding="utf-8") as f:
            stats = json.load(f)
        cols = ["dLat", "dLon", "SOG", "COG"]
        Config.FEAT_MIN = np.array([stats[c]["min"] for c in cols], dtype=np.float32)
        Config.FEAT_MAX = np.array([stats[c]["max"] for c in cols], dtype=np.float32)
        print(f"[Data] loaded {self.num_samples} trajectory samples ({Config.TOTAL_LEN} steps/sample)")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        start = idx * Config.TOTAL_LEN
        window = self.df.iloc[start: start + Config.TOTAL_LEN]
        norm = window[["dLat_norm", "dLon_norm", "SOG_norm", "COG_norm"]].to_numpy(dtype=np.float32)
        X = norm[:Config.HIST_LEN]
        Y = norm[Config.HIST_LEN:]
        dlat = window["dLat"].values.astype(np.float32)[Config.HIST_LEN:]
        dlon = window["dLon"].values.astype(np.float32)[Config.HIST_LEN:]
        sog = window["SOG"].values.astype(np.float32)[Config.HIST_LEN:]
        cog = window["COG"].values.astype(np.float32)[Config.HIST_LEN:]
        Y_raw = np.stack([dlat, dlon, sog, cog], axis=-1)
        anchor_lat = window["anchor_lat"].iloc[0]
        anchor_lon = window["anchor_lon"].iloc[0]

        return (
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(Y, dtype=torch.float32),
            torch.tensor(Y_raw, dtype=torch.float32),
            torch.tensor(anchor_lat, dtype=torch.float32),
            torch.tensor(anchor_lon, dtype=torch.float32)
        )

    def get_train_val_split(self, val_ratio=0.2, seed=0):
        unique_segs = np.unique(self.seg_ids)
        np.random.seed(seed)
        np.random.shuffle(unique_segs)
        val_segs = set(unique_segs[:int(len(unique_segs) * val_ratio)])
        train_indices = [i for i in range(self.num_samples) if self.seg_ids[i] not in val_segs]
        val_indices = [i for i in range(self.num_samples) if self.seg_ids[i] in val_segs]
        print(f"[Data] split: train {len(train_indices)}, val {len(val_indices)}")
        return train_indices, val_indices


class ScaleSpecificAttention(nn.Module):
    def __init__(self, d_model, scale):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1))
        self.d_k = d_model

    def forward(self, Q, K, V):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        return (F.softmax(scores, dim=-1) @ V) * self.alpha


class ScaleSpecificPE(nn.Module):
    def __init__(self, d_model, scale, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div * scale)
        pe[:, 1::2] = torch.cos(pos * div * scale)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(1)].unsqueeze(0)


class MultiScaleEncoder(nn.Module):
    def __init__(self, n_vars=4, d_model=256, n_heads=8, n_layers=2, seq_len=18, scales=None, dropout=0.1):
        super().__init__()
        self.scales = scales or [1, 4, 16, 32]
        self.proj = nn.Sequential(nn.Linear(seq_len, d_model), nn.ReLU())
        self.scale_atts = nn.ModuleList([ScaleSpecificAttention(d_model, s) for s in self.scales])
        self.scale_pes = nn.ModuleList([ScaleSpecificPE(d_model, s) for s in self.scales])
        layer = nn.TransformerEncoderLayer(d_model, n_heads, d_model * 4, dropout, norm_first=True, batch_first=True)
        self.tf = nn.TransformerEncoder(layer, n_layers)
        self.cross = nn.MultiheadAttention(d_model * len(self.scales), n_heads, batch_first=True)
        self.fuse = nn.Sequential(nn.Linear(d_model * len(self.scales), d_model), nn.ReLU())

    def forward(self, x):
        h = self.proj(x.transpose(1, 2))
        feats = []
        for i in range(len(self.scales)):
            p = self.scale_pes[i](h)
            att_out = self.scale_atts[i](p, p, p)
            feats.append(att_out)
        concat = torch.cat(feats, dim=-1)
        fused, _ = self.cross(concat, concat, concat)
        fused = self.fuse(fused)
        out = self.tf(fused + h)
        return out


class MaritimeTrajModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.d_model = config.D_MODEL
        self.encoder = MultiScaleEncoder(
            n_vars=config.FEAT_DIM, d_model=config.D_MODEL, n_heads=config.N_HEADS,
            n_layers=config.N_LAYERS, seq_len=config.HIST_LEN, scales=config.SCALES
        )
        self.traj_head = nn.Sequential(
            nn.Linear(config.D_MODEL, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, config.PRED_LEN * config.FEAT_DIM)
        )

    def forward(self, x):
        h = self.encoder(x)
        pooled = h.mean(dim=1)
        out = self.traj_head(pooled)
        return out.view(-1, Config.PRED_LEN, Config.FEAT_DIM)


def haversine_nm(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(torch.deg2rad, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = torch.sin(dlat / 2) ** 2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2) ** 2
    c = 2 * torch.asin(torch.sqrt(a))
    return Config.EARTH_RADIUS_NM * c


def evaluate_metrics(y_pred_norm, y_true_raw, anchor_lats, anchor_lons):
    feat_min = torch.tensor(Config.FEAT_MIN, device=y_pred_norm.device)
    feat_max = torch.tensor(Config.FEAT_MAX, device=y_pred_norm.device)
    y_pred = y_pred_norm * (feat_max - feat_min) + feat_min
    pred_dlat = y_pred[:, :, 0]   
    pred_dlon = y_pred[:, :, 1]   

    true_dlat = y_true_raw[:, :, 0]
    true_dlon = y_true_raw[:, :, 1]

    anchor_lats = anchor_lats.view(-1, 1)
    anchor_lons = anchor_lons.view(-1, 1)
    pred_lat = anchor_lats + pred_dlat
    pred_lon = anchor_lons + pred_dlon
    true_lat = anchor_lats + true_dlat
    true_lon = anchor_lons + true_dlon

    dists = haversine_nm(pred_lat, pred_lon, true_lat, true_lon)

    pred_sog = y_pred[:, :, 2]   
    pred_cog = y_pred[:, :, 3]   
    true_sog = y_true_raw[:, :, 2]
    true_cog = y_true_raw[:, :, 3]

    cog_diff = torch.abs(pred_cog - true_cog) % 360.0
    cog_diff = torch.minimum(cog_diff, 360.0 - cog_diff)

    return {
        "ADE_nm": torch.mean(dists).item(),
        "FDE_nm": torch.mean(dists[:, -1]).item(),
        "MSE_SOG": torch.mean((pred_sog - true_sog) ** 2).item(),
        "MAE_SOG": torch.mean(torch.abs(pred_sog - true_sog)).item(),
        "MSE_COG": torch.mean(cog_diff ** 2).item(),
        "MAE_COG": torch.mean(cog_diff).item(),
    }


@torch.no_grad()
def _eval_metrics(model, loader, device):
    model.eval()
    accum = {k: 0.0 for k in METRIC_KEYS}
    total = 0
    for batch in loader:
        X, _, Y_raw, anchor_lat, anchor_lon = batch
        X = X.to(device)
        Y_raw = Y_raw.to(device)
        anchor_lat = anchor_lat.to(device)
        anchor_lon = anchor_lon.to(device)
        pred = model(X)
        m = evaluate_metrics(pred, Y_raw, anchor_lat, anchor_lon)
        n = X.size(0)
        for k in METRIC_KEYS:
            accum[k] += m[k] * n
        total += n
    for k in METRIC_KEYS:
        accum[k] /= total
    return accum





class PairDataset(Dataset):
    
    def __init__(self, base, indices):
        self.base = base
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        X, Y, _, _, _ = self.base[self.indices[i]]
        return X, Y


def _convert_gabo_keys(results, tresh, alpha=None):
    
    def convert(key):
        parts = key.split('.')
        return parts[-1] + ''.join(parts[:-1])
    return ({convert(k): v for k, v in results.items()},
            {convert(k): v for k, v in tresh.items()},
            None)


_sac_episodes = 10
_sac_steps = 20
_protect_ratio = 0.85
_sac_reward_metric = 'fde'
_sac_fault_rate = 1e-6
_sac_max_samples = 2048
_sac_trials = 5
_sac_reduce = 'mean'
_sac_tolerance = 0.02
_sac_penalty_weight = 10.0
_sac_protect_bonus_weight = 0.0

_gabo_protect_ratio = 0.5    
_gabo_fde_weight = 3.0       
_gabo_max_samples = 2048     
_gabo_max_iter = 100         
_gabo_ablation = "full"      
_gabo_correct_threshold = 0.1  
_gabo_bound_method = "shrink_iqr_adaptive"    


def find_bounds(model, data_loader, name, bound_type, bitflip):
    
    if name == 'gabo':
        results, tresh, alpha = GABO_reg_ablation_bounds(
            model, data_loader, device=Config.DEVICE, bound_type=bound_type, bitflip=bitflip,
            protect_ratio=_gabo_protect_ratio, fde_weight=_gabo_fde_weight,
            max_samples=_gabo_max_samples, max_iter=_gabo_max_iter,
            ablation=_gabo_ablation, correct_threshold=_gabo_correct_threshold,
            bound_method=_gabo_bound_method,
            feat_min=Config.FEAT_MIN, feat_max=Config.FEAT_MAX)
        return _convert_gabo_keys(results, tresh, alpha)
    if name == 'gabo_sac':
        sac_loader_dict = dict(data_loader)
        sac_loader_dict['sub_train'] = sac_loader_dict['sub_train_full']
        results, tresh, alpha = SAC_ablation_bounds(model, sac_loader_dict, device=Config.DEVICE,
                                                   bound_type=bound_type, bitflip=bitflip,
                                                   episodes=_sac_episodes, steps_per_episode=_sac_steps,
                                                   protect_ratio=_protect_ratio,
                                                   reward_metric=_sac_reward_metric,
                                                   fault_rate=_sac_fault_rate,
                                                   eval_samples=_sac_max_samples,
                                                   n_fault_trials=_sac_trials,
                                                   reduce=_sac_reduce,
                                                   tolerance=_sac_tolerance,
                                                   penalty_weight=_sac_penalty_weight,
                                                   protect_bonus_weight=_sac_protect_bonus_weight,
                                                   feat_min=Config.FEAT_MIN, feat_max=Config.FEAT_MAX,
                                                   ablation=_gabo_ablation,
                                                   bound_method=_gabo_bound_method)
        return _convert_gabo_keys(results, tresh, alpha)
    if name != 'ranger':
        print(f"[warn] search_bound='{name}' 的现有实现依赖缺失(gymnasium), 回退到 'ranger'")
    return Ranger_bounds(model, data_loader, device=Config.DEVICE,
                         bound_type=bound_type, bitflip=bitflip)


def replace_act_all(model, relu_bound_fn, bounds, tresh, alpha=None, name=''):
    
    for name1, layer in model.named_children():
        if list(layer.children()) == []:
            if isinstance(layer, nn.ReLU) and 'last' not in name1:
                name_ = name1 + name
                if alpha is None:
                    model._modules[name1] = relu_bound_fn(bounds[name_].detach(), tresh[name_].detach(), alpha)
                else:
                    model._modules[name1] = relu_bound_fn(bounds[name_].detach(), tresh[name_].detach(),
                                                          alpha[name_].detach())
            elif isinstance(layer, nn.ReLU) and 'last' in name1:
                name_ = name1 + name
                model._modules[name1] = relu_bound_fn(bounds[name_].detach(), tresh[name_].detach(), None, k=-20.0)
        else:
            replace_act_all(layer, relu_bound_fn, bounds, tresh, alpha, name + name1)
    return model


def _extract_functions(py_path, func_names):
    
    with open(py_path, 'r', encoding='utf-8') as f:
        src = f.read()
    lines = src.splitlines()
    chunks = []
    for node in ast.parse(src).body:
        if isinstance(node, ast.FunctionDef) and node.name in func_names:
            chunks.append('\n'.join(lines[node.lineno - 1:node.end_lineno]))
    return '\n\n'.join(chunks)


_brelu_ns = {
    'copy': copy, 'nn': nn, 'torch': torch, 'print': print,
    'bounded_relu_zero': bounded_relu_zero,
    'bounded_relu_tresh': bounded_relu_tresh,
    'bounded_relu_fitact': bounded_hyrelu_proact,
    'bounded_hyrelu_proact': bounded_hyrelu_proact,
    'bounded_hyrelu_rlact': bounded_hyrelu_proact,
    'find_bounds': find_bounds,
    'replace_act_all': replace_act_all,
}
exec(_extract_functions(os.path.join(_PROJECT_ROOT, 'src', 'rrelu', 'setup.py'), {'replace_act'}), _brelu_ns)
replace_act = _brelu_ns['replace_act']


def eval(model, data_loader_dict):
    
    device = next(model.parameters()).device
    metrics = _eval_metrics(model, data_loader_dict["val"], device)
    metrics["val_loss"] = metrics["ADE_nm"]                 
    metrics["val_acc"] = 1.0 / (1.0 + metrics["ADE_nm"])     
    return metrics


def eval_fault(model: nn.Module, data_loader_dict, fault_rate, iterations=1000,
               bitflip="fixed", total_bits=32, n_frac=16, n_int=15) -> Dict:
    
    device = next(model.parameters()).device
    inputs = next(iter(data_loader_dict['val']))[0]   

    if bitflip == 'fixed':
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param is not None:
                    param.copy_(torch.tensor(
                        Fxp(param.clone().cpu().numpy(), True,
                            n_word=total_bits, n_frac=n_frac, n_int=n_int).get_val()))

    pfi_model = FaultInjection(
        model,
        inputs.shape[0],
        input_shape=[inputs.shape[1], inputs.shape[2]],
        layer_types=[torch.nn.Linear, Relu_bound],
        total_bits=total_bits,
        n_frac=n_frac,
        n_int=n_int,
        use_cuda=USE_CUDA,
    )

    agg = {k: [] for k in METRIC_KEYS}
    for i in tqdm(range(iterations), desc="Fault rate {:.0e}".format(fault_rate)):
        if bitflip == 'float':
            corrupted_model = multi_weight_inj_float(pfi_model, fault_rate)
        elif bitflip == 'fixed':
            corrupted_model = multi_weight_inj_fixed(pfi_model, fault_rate)
        elif bitflip == 'int':
            corrupted_model = multi_weight_inj_int(pfi_model, fault_rate)
        else:
            corrupted_model = pfi_model.original_model

        if i == 0:
            diff = sum(int((v1 != v2).sum())
                       for (_, v1), (_, v2) in zip(pfi_model.original_model.state_dict().items(),
                                                   corrupted_model.state_dict().items()))
            types = [str(t).split('.')[-1].split("'")[0] for t in pfi_model.layers_type]
            all_lin = sum(1 for m in model.modules() if isinstance(m, torch.nn.Linear))
            all_rb = sum(1 for m in model.modules() if isinstance(m, Relu_bound))
            print("    [diag] fault_rate={:.0e}: pfi识别层数={}, 改动权重数={}".format(
                fault_rate, pfi_model.get_total_layers(), diff))
            print("    [diag] 模型实际 Linear={}, Relu_bound={}; pfi识别类型: {}".format(
                all_lin, all_rb, types))
            with torch.no_grad():
                x = next(iter(data_loader_dict['val']))[0][:8].to(device)
                o1 = pfi_model.original_model(x)
                o2 = corrupted_model(x)
                print("    [diag] 输出对比: original[0,0,:2]={}, corrupted[0,0,:2]={}, 最大差={:.4f}".format(
                    [round(v, 4) for v in o1[0, 0, :2].tolist()],
                    [round(v, 4) for v in o2[0, 0, :2].tolist()],
                    float((o1 - o2).abs().max())))

        m = _eval_metrics(corrupted_model, data_loader_dict["val"], device)
        for k in METRIC_KEYS:
            agg[k].append(m[k])

    mean = {k: float(np.mean(agg[k])) for k in METRIC_KEYS}
    val_results = {
        "val_loss": mean["ADE_nm"],                    
        "val_acc": 1.0 / (1.0 + mean["ADE_nm"]),       
        "fault_rate": fault_rate,
    }
    val_results.update(mean)
    return val_results



def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="neworleans",
                        help="按数据集名一键切换数据/模型/产物目录 (默认 %s)\n" % DEFAULT_DATASET
                        + "\n".join("  %s: %s -> %s" % (k, v["desc"], v["outdir"])
                                    for k, v in DATASETS.items()))
    parser.add_argument("--data", default=None,
                        help="数据 CSV (默认取 --dataset 对应的 windows 文件)")
    parser.add_argument("--model", default=None,
                        help="模型权重 (默认取 --dataset 对应的 .pth)")
    parser.add_argument("--batch_size", type=int, default=Config.BATCH_SIZE)
    parser.add_argument("--iterations", type=int, default=100, help="每个故障率注入次数")
    parser.add_argument("--seed", type=int, default=0, help="随机种子(数据划分/故障注入)")

    parser.add_argument("--n_word", type=int, default=32)
    parser.add_argument("--n_frac", type=int, default=16)
    parser.add_argument("--n_int", type=int, default=15)
    parser.add_argument("--name_relu_bound", type=str, default="tresh",
                        choices=["zero", "tresh", "proact", "fitact"])
    parser.add_argument("--name_serach_bound", type=str, default="gabo_sac",
                        choices=["ranger", "gabo", "gabo_sac", "proact"])
    parser.add_argument("--bounds_type", type=str, default="layer", choices=["layer", "neuron"])
    parser.add_argument("--bitflip", type=str, default="fixed", choices=["fixed", "float", "int"])
    parser.add_argument("--fault_rates", type=str, default="1e-7,3e-7,5e-7,7e-7,1e-6")
    parser.add_argument("--sac_episodes", type=int, default=30, help="SAC 训练 episode 数")
    parser.add_argument("--sac_steps", type=int, default=20, help="SAC 每 episode 步数")
    parser.add_argument("--sac_reward_metric", type=str, default="ade", choices=["ade", "fde", "ade_fde"],
                        help="SAC reward 最小化的故障指标 (ade/fde/ade_fde)")
    parser.add_argument("--sac_fault_rate", type=float, default=1e-6,
                        help="SAC 训练时的权重位翻转率(应与 --fault_rates 量级一致)")
    parser.add_argument("--sac_max_samples", type=int, default=2048, help="SAC 评估样本数")
    parser.add_argument("--sac_trials", type=int, default=5,
                        help="SAC 每次故障评估的独立注入次数(取平均, 抑制单次位翻转的方差)")
    parser.add_argument("--sac_reduce", type=str, default="median", choices=["mean", "median"],
                        help="多次注入的聚合方式: mean=期望退化, median=抗重尾更稳")
    parser.add_argument("--sac_tolerance", type=float, default=0.05,
                        help="clean FDE 允许相对基线的退化比例(0.02=2%%, 0.05=5%%)")
    parser.add_argument("--sac_penalty_weight", type=float, default=0.0,
                        help="精度越界惩罚权重(reward 里的 10)")
    parser.add_argument("--sac_protect_bonus_weight", type=float, default=0.0,
                        help="保护比例奖励权重(默认 0 关闭: protect_ratio 与 FDE 同量级会污染 reward)")
    parser.add_argument("--protect_ratio", type=float, default=0.0,
                        help="边界搜索下限比例 λ>=max_act*ratio(回归任务收缩过度会崩 clean ADE/FDE)")


    parser.add_argument("--gabo_protect_ratio", type=float, default=0.0,
                        help="GABO 边界下限比例 max_act*ratio: 越低边界越紧、容错越强, 但 clean 损失越大")
    parser.add_argument("--gabo_fde_weight", type=float, default=3.0, help="GABO FDE 终点惩罚权重")
    parser.add_argument("--gabo_max_samples", type=int, default=2048, help="GABO 统计+优化样本数")
    parser.add_argument("--gabo_max_iter", type=int, default=100, help="GABO 梯度上升迭代次数")
    parser.add_argument("--gabo-ablation", type=str, default="full",
                        choices=["full", "no_entropy", "no_skew", "fixed_iqr"],
                        help="GABO 初始化统计量消融: full / no_entropy / no_skew / fixed_iqr")
    parser.add_argument("--gabo-correct-threshold", type=float, default=0.0,
                        help="θthr 修正阈值: up_iqr < ratio*max_act 时才重置为 0.9*max_act (默认 0.1, 原版 0.5)")
    parser.add_argument("--bound-method", type=str, default="shrink_iqr_adaptive",
                        choices=["shrink_iqr_adaptive"],
                        help="上界公式: shrink_iqr_adaptive(统计量调制分位点的自适应IQR)")
    parser.add_argument("--max_samples", type=int, default=1024, help="验证集评估样本数")
    parser.add_argument("--sub_train", type=int, default=2048, help="边界统计子训练集样本数")
    args = parser.parse_args()

    ds_name = args.dataset or DEFAULT_DATASET
    if ds_name not in DATASETS:
        raise SystemExit("未知数据集: %s (可选: %s)" % (ds_name, sorted(DATASETS)))
    ds = DATASETS[ds_name]
    data_path = args.data or ds["csv"]
    model_path = args.model or ds["model"]
    out_dir = os.path.join(SCRIPT_DIR, ds["outdir"])
    print("[Dataset] %s | data=%s" % (ds_name, data_path))
    print("[Dataset] model=%s" % model_path)

    global _sac_episodes, _sac_steps, _protect_ratio, _sac_reward_metric, _sac_fault_rate, _sac_max_samples, _sac_trials, _sac_reduce, _sac_tolerance, _sac_penalty_weight, _sac_protect_bonus_weight
    global _gabo_protect_ratio, _gabo_fde_weight, _gabo_max_samples, _gabo_max_iter, _gabo_ablation, _gabo_correct_threshold, _gabo_bound_method
    _sac_episodes = args.sac_episodes
    _sac_steps = args.sac_steps
    _protect_ratio = args.protect_ratio
    _sac_reward_metric = args.sac_reward_metric
    _sac_fault_rate = args.sac_fault_rate
    _sac_max_samples = args.sac_max_samples
    _sac_trials = args.sac_trials
    _sac_reduce = args.sac_reduce
    _sac_tolerance = args.sac_tolerance
    _sac_penalty_weight = args.sac_penalty_weight
    _sac_protect_bonus_weight = args.sac_protect_bonus_weight
    _gabo_protect_ratio = args.gabo_protect_ratio
    _gabo_fde_weight = args.gabo_fde_weight
    _gabo_max_samples = args.gabo_max_samples
    _gabo_max_iter = args.gabo_max_iter
    _gabo_ablation = args.gabo_ablation
    _gabo_correct_threshold = args.gabo_correct_threshold
    _gabo_bound_method = args.bound_method

    fault_rates = [float(x) for x in args.fault_rates.split(",") if x.strip()]

    device = Config.DEVICE
    print(f"[Config] device={device}, relu_bound={args.name_relu_bound}, "
          f"search_bound={args.name_serach_bound}, bounds_type={args.bounds_type}, "
          f"bitflip={args.bitflip}, iterations={args.iterations}, fault_rates={fault_rates}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dataset = TrajectoryDataset(data_path)
    train_idx, val_idx = dataset.get_train_val_split(val_ratio=0.2, seed=args.seed)

    sub_train_idx = train_idx[:min(args.sub_train, len(train_idx))]
    eval_idx = val_idx[:min(args.max_samples, len(val_idx))]

    data_loader_dict = {
        "train": DataLoader(Subset(dataset, train_idx), batch_size=args.batch_size, shuffle=True),
        "val": DataLoader(Subset(dataset, eval_idx), batch_size=args.batch_size, shuffle=False),
        "sub_train": DataLoader(PairDataset(dataset, sub_train_idx), batch_size=args.batch_size, shuffle=False),
        "sub_train_full": DataLoader(Subset(dataset, sub_train_idx), batch_size=args.batch_size, shuffle=False),
    }
    print(f"[Data] sub_train={len(sub_train_idx)}, val(eval)={len(eval_idx)}")

    model = MaritimeTrajModel(Config).to(device)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    print(f"[Model] loaded {model_path}, params={sum(p.numel() for p in model.parameters()):,}")

    if args.bitflip == 'fixed':
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param != None:
                    param.copy_(torch.tensor(
                        Fxp(param.clone().cpu().numpy(), True, n_word=args.n_word, n_frac=args.n_frac,
                            n_int=args.n_int).get_val()))

    base_metrics = eval(model, data_loader_dict)
    print("\n[1] 基线性能 (原始 ReLU, 无故障):")
    for k in METRIC_KEYS:
        print(f"    {k:10s}: {base_metrics[k]:.4f}")


    print("\n[2] 无容错故障注入 (原始 ReLU + 故障):")

    orig_fault = {}

    print("\n[边界优化] 开始边界搜索 (search_bound={}, bounds_type={}) ...".format(
        args.name_serach_bound, args.bounds_type))
    t_bound_start = time.time()
    model = replace_act(model, args.name_relu_bound, args.name_serach_bound,
                        data_loader_dict, args.bounds_type, args.bitflip)
    bound_time = time.time() - t_bound_start
    print("[边界优化] 边界搜索耗时: {:.2f} s ({:.4f} min)".format(bound_time, bound_time / 60.0))

    print("\n[3] model accuracy in {} format = {} after replace ReLU activcation functions".format(
        args.bitflip, eval(model, data_loader_dict)))

    print("\n[4] ranger 容错故障注入 (有界 ReLU + 故障):")
    brelu_fault = {}
    for fault_rate in fault_rates:
        val_results_fault = eval_fault(model, data_loader_dict, fault_rate, args.iterations,
                                       args.bitflip, args.n_word, args.n_frac, args.n_int)
        brelu_fault[fault_rate] = val_results_fault
        print("val_loss = {} ,  val_acc = {} , fault_rate = {} ".format(
            val_results_fault['val_loss'],
            val_results_fault['val_acc'],
            val_results_fault['fault_rate']))
        print(f"    -> ADE={val_results_fault['ADE_nm']:.4f}, FDE={val_results_fault['FDE_nm']:.4f}, "
              f"MSE_SOG={val_results_fault['MSE_SOG']:.4f}")

    metrics_to_report = ["ADE_nm", "FDE_nm", "MSE_SOG", "MAE_SOG", "MSE_COG", "MAE_COG"]

    lines = []   
    rows = []    

    title = "=" * 100 + "\n性能对比汇总 (指标越小越好)\n" + "=" * 100
    print("\n" + title)
    lines.append(title)

    has_orig = bool(orig_fault)

    for metric in metrics_to_report:
        head_title = "\n【" + metric + "】"
        if has_orig:
            head_col = "{:>10s} | {:>13s} | {:>20s} | {:>13s} | {:>24s}".format(
                "fault_rate", "基线(无故障)", "无容错", "ranger容错", "退化对比")
        else:
            head_col = "{:>10s} | {:>13s} | {:>13s} | {:>16s}".format(
                "fault_rate", "基线(无故障)", "ranger容错", "退化")
        head_sep = "-" * 84
        print(head_title)
        print(head_col)
        print(head_sep)
        lines.extend([head_title, head_col, head_sep])

        base_v = base_metrics[metric]
        for fr in fault_rates:
            b = brelu_fault[fr][metric]
            deg_b = (b - base_v) / base_v * 100.0 if base_v > 0 else 0.0   
            if has_orig:
                o = orig_fault[fr][metric]
                deg_o = (o - base_v) / base_v * 100.0 if base_v > 0 else 0.0   
                row_txt = "{:>10.1e} | {:>13.4f} | {:>20.4f} | {:>13.4f} | {:>+8.1f}% -> {:>+7.1f}%".format(
                    fr, base_v, o, b, deg_o, deg_b)
                no_tol = o
                deg_o_val = deg_o
            else:
                row_txt = "{:>10.1e} | {:>13.4f} | {:>13.4f} | {:>+15.1f}%".format(
                    fr, base_v, b, deg_b)
                no_tol = ''
                deg_o_val = ''
            print(row_txt)
            lines.append(row_txt)
            rows.append({
                'metric': metric,
                'fault_rate': fr,
                'baseline': base_v,
                'no_tolerance': no_tol,
                'ranger': b,
                'deg_no_tolerance_pct': deg_o_val,
                'deg_ranger_pct': deg_b,
                'n_changed_weights': brelu_fault[fr].get('n_changed_weights', ''),
            })

    os.makedirs(out_dir, exist_ok=True)
    run_time = time.strftime('%Y-%m-%d %H:%M:%S')
    csv_path = os.path.join(out_dir, 'fault_compare.csv')   
    txt_path = os.path.join(out_dir, 'fault_compare.txt')   

    csv_fieldnames = ['run_time', 'relu_bound', 'search_bound', 'bitflip', 'iterations',
                      'max_samples', 'metric', 'fault_rate', 'baseline', 'no_tolerance',
                      'ranger', 'deg_no_tolerance_pct', 'deg_ranger_pct', 'n_changed_weights',
                      'bound_time_s']
    csv_exists = os.path.exists(csv_path)
    with open(csv_path, 'a', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=csv_fieldnames)
        if not csv_exists:
            writer.writeheader()
        for r in rows:
            row = {
                'run_time': run_time,
                'relu_bound': args.name_relu_bound,
                'search_bound': args.name_serach_bound,
                'bitflip': args.bitflip,
                'iterations': args.iterations,
                'max_samples': len(eval_idx),
                'bound_time_s': round(bound_time, 2),
            }
            row.update(r)
            writer.writerow(row)

    with open(txt_path, 'a', encoding='utf-8') as f:
        f.write("\n" + "=" * 100 + "\n")
        f.write("生成时间: {}\n".format(run_time))
        f.write("配置: relu_bound={}, search_bound={}, bounds_type={}, bitflip={}, "
                "iterations={}, fault_rates={}\n".format(
                    args.name_relu_bound, args.name_serach_bound, args.bounds_type,
                    args.bitflip, args.iterations, fault_rates))
        f.write("评估样本数: {}, 边界统计(sub_train)样本数: {}, 模型参数量: {:,}\n".format(
            len(eval_idx), len(sub_train_idx), sum(p.numel() for p in model.parameters())))
        f.write("边界优化耗时: {:.2f} s ({:.4f} min)\n".format(bound_time, bound_time / 60.0))
        f.write("\n基线(原始 ReLU, 无故障) 指标:\n")
        for k in METRIC_KEYS:
            f.write("    {:10s}: {:.4f}\n".format(k, base_metrics[k]))
        f.write("\n")
        f.write("\n".join(lines))
        f.write("\n")

    print("\n结果已追加保存:")
    print("  CSV: {}".format(csv_path))
    print("  TXT: {}".format(txt_path))

    print("\nDone.")


if __name__ == "__main__":
    main()
