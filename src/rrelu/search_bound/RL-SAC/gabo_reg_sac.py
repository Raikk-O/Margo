# -*- coding: utf-8 -*-
"""
SAC_bounds — 轨迹回归版 SAC 边界优化 (两阶段, 严格对齐老设计 gabo_SAC.py)
========================================================================
Phase 1 (GABO) : 逐层梯度上升 → 初始 λ + 搜索空间锚点
Phase 2 (SAC)  : 全局联合微调(固定状态 bandit + 直接双向 λ 动作)

与老设计(gabo_SAC.py)对齐的要点:
  1. 两阶段: GABO 出初始 λ, SAC 微调。
  2. 动作直接映射 λ = max_act * exp(action * scale), 可收缩也可放大(clamp_to_ranger 可选)。
  3. 固定上下文 state(单步 MDP, critic 不自举)。
  4. 奖励 = -故障指标 - penalty_weight*精度惩罚 + protect_bonus_weight*保护比例(稀疏引导)。

保留(与老设计不同但必要的改造):
  * 故障模型 = BFWP 权重位翻转(向量化, 快照还原, 多次注入聚合 mean/median), 与 eval_fault 对齐。
  * 边界施加用模块替换(BoundedReLU)而非 hook, 便于配合 BFWP 快照还原。
  * reward 指标可选 ade / fde / ade_fde(轨迹回归)。

对外接口(与 Ranger_bounds 同契约):
    results_DMA, tresh, alpha = SAC_bounds(model, data_loader_dict, device, ...)
    results_DMA/tresh 的 key 用 Ranger 风格('1encoderproj' 等), 与 replace_act_all 一致。

依赖: torch, numpy
"""
import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Part 0: 统计辅助 (from Args_ana.py)
# =============================================================================
def _feature_entropy(x, n_bins=256):
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size == 0: return 0.0
    mn = x.min()
    if mn < 0: x = x - mn
    mx = x.max()
    if mx != 0: x = x / mx
    q = (x * 255).astype(np.int64)
    counts = np.bincount(q + 1, minlength=n_bins).astype(np.float64)
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def _skewness(x):
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size == 0: return 0.0
    s = x.std()
    if s == 0: return 0.0
    return float(((x - x.mean()) ** 3).mean() / (s ** 3))


def iqr_method(features, custom_mean, custom_std, custom_entropy, custom_skewness,
               base_factor=1.5, base_lower_percentile=10, base_upper_percentile=90):
    features = np.asarray(features, dtype=np.float64).ravel()
    n = len(features)
    if n == 0: return 0.0, 0.0
    lower_percentile = max(1.0, base_lower_percentile - np.log(n) * 0.5 - abs(custom_skewness))
    upper_percentile = min(99.0, base_upper_percentile + np.log(n) * 0.5
                           + abs(custom_skewness) + custom_entropy * 0.05)
    std_mean_ratio = custom_std / custom_mean if custom_mean != 0 else base_factor
    std_mean_ratio = float(np.clip(std_mean_ratio, 0, 0.1))
    factor = base_factor + std_mean_ratio
    q1 = np.percentile(features, lower_percentile)
    q3 = np.percentile(features, upper_percentile)
    iqr = q3 - q1
    return float(q1 - factor * iqr), float(q3 + factor * iqr)


def _dynamic_upper_bound_range(final_upper_bound, custom_std, factor=1.0):
    adjustment = custom_std * factor
    return float(final_upper_bound - adjustment), float(final_upper_bound + adjustment)


def _dynamically_adjust_upper_bound(final_upper_bound, custom_max_value,
                                    threshold_ratio=0.5, correction_factor=0.9):
    if final_upper_bound < custom_max_value * threshold_ratio:
        return float(custom_max_value * correction_factor)
    return float(final_upper_bound)


def _ranger_key(module_path):
    """named_modules 路径 'encoder.proj.1' -> Ranger key '1encoderproj'。"""
    parts = module_path.split('.')
    return parts[-1] + ''.join(parts[:-1])


# =============================================================================
# Part 1: Hook 辅助 (供 GABO Phase 收集激活)
# =============================================================================
def _register_relu_hooks(model, activations, handles, name=''):
    for name1, layer in model.named_children():
        if len(list(layer.children())) == 0:
            if isinstance(layer, nn.ReLU):
                name_ = name1 + name

                def _make_hook(key):
                    def _hook(module, inp, out):
                        out.retain_grad()
                        activations[key] = out
                    return _hook

                handles.append(layer.register_forward_hook(_make_hook(name_)))
        else:
            _register_relu_hooks(layer, activations, handles, name + name1)


# =============================================================================
# Part 2: 自适应学习率 (from gabo_SAC.py)
# =============================================================================
def _adjust_learning_rate(grad, prev_grad, lr, initial_lr_factor=0.2,
                          scale_factor=2, patience_counter=None, increase_patience=2):
    grad_abs = grad.abs().clamp_min(1e-12)
    max_lr = initial_lr_factor / grad_abs
    if patience_counter is None:
        patience_counter = torch.zeros_like(grad)
    if prev_grad is None:
        lr = (initial_lr_factor / grad_abs).clamp(1e-4, 1e9)
        return torch.round(lr * 1e5) / 1e5, patience_counter
    sign_change = torch.sign(grad) != torch.sign(prev_grad)
    lr = torch.where(sign_change, lr / scale_factor, lr)
    inc = ~sign_change
    patience_counter = patience_counter + inc.to(patience_counter.dtype)
    bump = inc & (patience_counter >= increase_patience)
    lr = torch.where(bump, lr * scale_factor, lr)
    patience_counter = torch.where(bump, torch.zeros_like(patience_counter), patience_counter)
    lr = torch.minimum(lr, max_lr)
    return torch.round(lr * 1e5) / 1e5, patience_counter


# =============================================================================
# Part 3: GABO 核心 (回归版梯度上升, 同 gabo_SAC.py)
# =============================================================================
def _gradient_ascent_bound_regression(
        residual0, grad_act, act, init_lambda,
        lower_limit, upper_limit, per_neuron,
        mse_clean_baseline, tolerance=0.05, penalty_weight=10.0,
        max_iter=50, patience=6,
        initial_lr_factor=0.2, scale_factor=2,
        device="cpu", verbose=False):
    dtype = act.dtype
    if per_neuron:
        lam = torch.full((act.shape[1],), float(init_lambda), device=device, dtype=dtype, requires_grad=True)
        lower_limit = torch.as_tensor(lower_limit, device=device, dtype=dtype)
        upper_limit = torch.as_tensor(upper_limit, device=device, dtype=dtype)
    else:
        lam = torch.tensor(float(init_lambda), device=device, dtype=dtype, requires_grad=True)
        lower_limit = torch.as_tensor(float(lower_limit), device=device, dtype=dtype)
        upper_limit = torch.as_tensor(float(upper_limit), device=device, dtype=dtype)

    lr = None; prev_grad = None; patience_counter = None
    best_obj = -float("inf"); best_lam = lam.detach().clone()
    obj_stack = []; delta = 1e-8

    for _ in range(max_iter):
        deviation = torch.clamp(lam - act, max=0.0)  # min(λ-a, 0): 低于阈值偏差为 0
        perturb = (grad_act * deviation).sum(dim=1)
        residual_perturbed = residual0 + perturb
        mse_fault = residual_perturbed.mean()  # residual 已是 MSE, 直接取均值
        penalty = F.relu(mse_fault - mse_clean_baseline * (1.0 + tolerance))
        objective = -mse_fault - penalty_weight * penalty
        obj_val = float(objective.detach())
        obj_stack.append(obj_val)

        if obj_val > best_obj and float(lam.detach().min()) > 0:
            best_obj = obj_val; best_lam = lam.detach().clone()

        if len(obj_stack) > patience:
            recent = obj_stack[-patience:]
            changes = [abs(recent[i] - recent[i - 1]) for i in range(1, len(recent))]
            trend = [recent[i] <= recent[i - 1] for i in range(1, len(recent))]
            if all(c < delta for c in changes) and all(trend): break

        grad = torch.autograd.grad(objective, lam, retain_graph=False)[0]
        if grad is None or not torch.isfinite(grad).all() or float(grad.abs().max()) == 0: break

        lr, patience_counter = _adjust_learning_rate(grad, prev_grad, lr, initial_lr_factor, scale_factor, patience_counter)
        prev_grad = grad.detach()
        if not torch.isfinite(lr).all() or float(lr.abs().max()) == 0: break

        with torch.no_grad():
            lam_new = lam + lr * grad
            lam_new = torch.round(lam_new * 1e4) / 1e4
            lam.data.copy_(lam_new)

    return best_lam.detach().clamp(lower_limit, upper_limit)


# =============================================================================
# Part 4: GABO Phase (轨迹回归: 输出 (B, T, 4) -> 用"平均输出误差"作标量 residual)
# =============================================================================
def gabo_phase(model, train_loader, device, max_samples=2048, protect_ratio=0.95,
               tolerance=0.05, penalty_weight=10.0,
               max_iter=50, patience=6, initial_lr_factor=0.2, scale_factor=2,
               dynamic_range_factor=1.0, verbose=False):
    model.eval(); model = model.to(device)
    activations = {}; handles = []
    _register_relu_hooks(model, activations, handles, name='')
    loader = train_loader['sub_train']
    stats = {}; all_residuals = []; n_seen = 0

    for batch in loader:
        # 轨迹 5 元组 (X, Y, Y_raw, ref_lat, ref_lon)
        data = batch[0].to(device)
        label = batch[1].to(device)                 # Y, (B, PRED_LEN, FEAT_DIM)
        activations.clear()
        out = model(data)                           # (B, PRED_LEN, FEAT_DIM)
        # 标量化: 每个样本的 MSE(先平方再平均), 避免正负误差相互抵消
        residual = ((out - label) ** 2).reshape(out.shape[0], -1).mean(dim=1)  # (B,)

        for key, act in activations.items():
            grad_act = torch.autograd.grad(
                residual, act, grad_outputs=torch.ones_like(residual),
                retain_graph=True, allow_unused=True)[0]
            if grad_act is None: continue
            st = stats.setdefault(key, {"act": [], "grad": []})
            # 轨迹模型的部分 ReLU 激活是 3 维 (B, *, D), GABO 核心要求 2 维 (N, D), 这里按样本展平
            st["act"].append(act.detach().reshape(act.shape[0], -1))
            st["grad"].append(grad_act.detach().reshape(grad_act.shape[0], -1))
        all_residuals.append(residual.detach())
        n_seen += data.shape[0]
        if n_seen >= max_samples: break

    for h in handles: h.remove()
    if not stats: return {}, {}, {}, {}, 0.0, {}

    residual0 = torch.cat(all_residuals, dim=0)          # (N,)
    mse_clean = float(residual0.mean())  # residual 已是 MSE, 直接取均值
    init_lambdas, bounds_dict, tresh_dict, layer_stats, max_acts = {}, {}, {}, {}, {}

    for key, st in stats.items():
        act = torch.cat(st["act"], dim=0)                # (N, D)
        grad_act = torch.cat(st["grad"], dim=0)          # (N, D)
        act_np = act.detach().cpu().numpy().astype(np.float64)

        batch_means = [a.mean(dim=0) for a in st["act"]]
        mins = batch_means[0]
        for m in batch_means[1:]: mins = torch.minimum(mins, m)
        tresh_dict[key] = torch.min(mins)

        flat = act_np.ravel()
        lo_iqr, up_iqr = iqr_method(flat, float(flat.mean()), float(flat.std()),
                                    _feature_entropy(flat), _skewness(flat))
        max_act = float(act_np.max()); std_act = float(act_np.std()); mean_grad = float(grad_act.abs().mean())
        up_iqr = _dynamically_adjust_upper_bound(up_iqr, max_act)
        up_iqr = min(up_iqr, max_act)
        lower_limit, upper_limit = _dynamic_upper_bound_range(up_iqr, std_act, dynamic_range_factor)
        upper_limit = min(upper_limit, max_act)
        if protect_ratio > 0: lower_limit = max(lower_limit, max_act * protect_ratio)
        upper_limit = max(upper_limit, lower_limit)

        bound = _gradient_ascent_bound_regression(
            residual0=residual0, grad_act=grad_act, act=act, init_lambda=float(up_iqr),
            lower_limit=lower_limit, upper_limit=upper_limit, per_neuron=False,
            mse_clean_baseline=mse_clean, tolerance=tolerance, penalty_weight=penalty_weight,
            max_iter=max_iter, patience=patience, initial_lr_factor=initial_lr_factor,
            scale_factor=scale_factor, device=device, verbose=verbose)

        init_lambdas[key] = bound
        bounds_dict[key] = (float(lower_limit), float(upper_limit))
        max_acts[key] = max_act
        layer_stats[key] = {"max_act": max_act, "std_act": std_act, "mean_grad": mean_grad,
                            "iqr_range": up_iqr - lo_iqr}

    return init_lambdas, bounds_dict, tresh_dict, layer_stats, mse_clean, max_acts


# =============================================================================
# Part 5: 有界 ReLU + 向量化位翻转 (BFWP)
# =============================================================================
class BoundedReLU(nn.Module):
    def __init__(self, upper_bound):
        super().__init__()
        self.register_buffer('upper_bound', torch.tensor(float(upper_bound), dtype=torch.float32))

    def forward(self, x):
        return torch.clamp(torch.relu(x), max=self.upper_bound)


def _get_module_by_path(model, name):
    parts = name.split('.')
    m = model
    for p in parts:
        m = getattr(m, p)
    return m


def _flip_bits_fixed(t, fault_rate, total_bits=32, n_frac=16, n_int=15):
    n_fault = int(t.numel() * total_bits * fault_rate)
    if n_fault <= 0:
        return
    dev = t.device
    scale = 2 ** n_frac
    q = torch.round(t.detach().double() * scale).to(torch.int64).view(-1)
    idx = torch.randint(0, t.numel(), (n_fault,), device=dev)
    b = torch.randint(0, total_bits, (n_fault,), device=dev)
    full = (1 << total_bits) - 1
    sign = 1 << (total_bits - 1)
    u_new = (q & full)[idx] ^ torch.bitwise_left_shift(torch.ones_like(b), b)
    q_new = (u_new ^ sign) - sign
    t.data.view(-1)[idx] = (q_new.to(torch.float64) / scale).to(t.dtype)


def _flip_bits_float(t, fault_rate, total_bits=32):
    n_fault = int(t.numel() * total_bits * fault_rate)
    if n_fault <= 0 or t.dtype != torch.float32:
        return
    dev = t.device
    int_view = t.detach().contiguous().view(torch.int32).view(-1)
    idx = torch.randint(0, int_view.numel(), (n_fault,), device=dev)
    b = torch.randint(0, total_bits, (n_fault,), device=dev).to(torch.int32)
    int_view[idx] ^= torch.bitwise_left_shift(torch.ones_like(b), b)


# =============================================================================
# Part 6: SAC 网络 (同 gabo_SAC.py)
# =============================================================================
class SACActor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.mu_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = self.net(state)
        mu = self.mu_head(x)
        log_std = self.log_std_head(x).clamp(-20, 2)
        return mu, log_std

    def sample(self, state):
        mu, log_std = self.forward(state)
        std = log_std.exp()
        dist = torch.distributions.Normal(mu, std)
        x_t = dist.rsample()
        action = torch.tanh(x_t)
        log_prob = dist.log_prob(x_t) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob


class SACCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.q1 = nn.Sequential(nn.Linear(state_dim + action_dim, hidden_dim), nn.ReLU(),
                                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
        self.q2 = nn.Sequential(nn.Linear(state_dim + action_dim, hidden_dim), nn.ReLU(),
                                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa), self.q2(sa)


# =============================================================================
# Part 7: 环境 (固定状态 bandit + 直接双向 λ 动作 + BFWP + protect_ratio 奖励)
# =============================================================================
class BoundEvalEnv:
    def __init__(self, model, data_loader, max_acts, layer_keys, relu_paths, device,
                 fault_rate, bitflip, total_bits, n_frac, n_int, reward_metric='fde',
                 tolerance=0.05, penalty_weight=10.0, protect_bonus_weight=0.0,
                 n_fault_trials=5, action_scale=0.5, reduce='mean',
                 clamp_to_ranger=True, eval_samples=2048, protect_ratio=0.85,
                 feat_min=None, feat_max=None):
        self.protect_ratio = protect_ratio
        self.device = device
        self.max_acts = max_acts              # dict: key -> max_act (Ranger 上界)
        self.layer_keys = layer_keys          # 有序 key 列表
        self.relu_paths = relu_paths          # 与 layer_keys 对齐的模块路径
        self.action_dim = len(layer_keys)
        self.reward_metric = reward_metric
        self.tolerance = tolerance
        self.penalty_weight = penalty_weight
        self.protect_bonus_weight = protect_bonus_weight
        self.fault_rate = fault_rate
        self.bitflip = bitflip
        self.total_bits = total_bits
        self.n_frac = n_frac
        self.n_int = n_int
        self.n_fault_trials = n_fault_trials
        self.action_scale = action_scale
        self.reduce = reduce
        self.clamp_to_ranger = clamp_to_ranger
        self.feat_min = feat_min
        self.feat_max = feat_max

        # 工作模型(只深拷贝一次) + 权重快照 + 要翻转的 Linear 层
        self._work_model = copy.deepcopy(model).to(device).eval()
        self._lin_names = self._collect_linear_names(self._work_model)
        self._weight_snapshot = {}
        for n in self._lin_names:
            mod = _get_module_by_path(self._work_model, n)
            self._weight_snapshot[n + '.weight'] = mod.weight.detach().clone()
            if mod.bias is not None:
                self._weight_snapshot[n + '.bias'] = mod.bias.detach().clone()

        # 评估数据(5 元组)缓存
        self.X_val, self.Y_raw_val, self.ref_lats_val, self.ref_lons_val = self._prepare_data(data_loader, eval_samples)

        # 固定上下文 state (老设计)
        state_parts = []
        for key in layer_keys:
            m = max_acts[key]
            state_parts.extend([m, m * 0.1, m * 0.5, m * 2.0])
        self.state = torch.tensor(state_parts, dtype=torch.float32, device=device).unsqueeze(0)

        self.baseline_metrics = self._evaluate(self._lambda_max_array(), inject_fault=False)

    @property
    def state_dim(self):
        return self.state.shape[1]

    @staticmethod
    def _collect_linear_names(model):
        skip = set()
        for m in model.modules():
            if isinstance(m, nn.MultiheadAttention):
                for c in m.children():
                    if isinstance(c, nn.Linear):
                        skip.add(id(c))
        return [n for n, m in model.named_modules() if isinstance(m, nn.Linear) and id(m) not in skip]

    def _prepare_data(self, data_loader, eval_samples):
        X, Yr, lat, lon = [], [], [], []
        n = 0
        for batch in data_loader:
            X.append(batch[0]); Yr.append(batch[2]); lat.append(batch[3]); lon.append(batch[4])
            n += batch[0].size(0)
            if n >= eval_samples: break
        return (torch.cat(X).to(self.device), torch.cat(Yr).to(self.device),
                torch.cat(lat).to(self.device), torch.cat(lon).to(self.device))

    def _lambda_max_array(self):
        return np.array([self.max_acts[k] for k in self.layer_keys], dtype=np.float32)

    def _action_to_lambda(self, action):
        """老设计: λ = max_act * exp(action * scale), 可收缩/放大, clamp_to_ranger 裁掉放大侧。"""
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()
        lambdas = []
        for i, key in enumerate(self.layer_keys):
            m = self.max_acts[key]
            lam = m * math.exp(float(action[i]) * self.action_scale)
            if self.clamp_to_ranger:
                lam = min(lam, m)
            # 下界约束: λ >= max_act * protect_ratio, 防止轨迹回归收缩过度导致 clean ADE/FDE 崩
            lam = max(lam, m * self.protect_ratio)
            lambdas.append(lam)
        return lambdas

    def _apply_bounds(self, lambdas):
        for i, path in enumerate(self.relu_paths):
            parts = path.split('.')
            parent = self._work_model
            for p in parts[:-1]:
                parent = getattr(parent, p)
            setattr(parent, parts[-1], BoundedReLU(lambdas[i]))
        self._work_model.to(self.device)

    def _restore_weights(self):
        for n in self._lin_names:
            mod = _get_module_by_path(self._work_model, n)
            mod.weight.data.copy_(self._weight_snapshot[n + '.weight'])
            if mod.bias is not None:
                mod.bias.data.copy_(self._weight_snapshot[n + '.bias'])

    def _inject_weight_faults(self):
        for n in self._lin_names:
            mod = _get_module_by_path(self._work_model, n)
            if self.bitflip == 'fixed':
                _flip_bits_fixed(mod.weight, self.fault_rate, self.total_bits, self.n_frac, self.n_int)
                if mod.bias is not None:
                    _flip_bits_fixed(mod.bias, self.fault_rate, self.total_bits, self.n_frac, self.n_int)
            else:
                _flip_bits_float(mod.weight, self.fault_rate, self.total_bits)
                if mod.bias is not None:
                    _flip_bits_float(mod.bias, self.fault_rate, self.total_bits)

    # ---- 单次前向 -> {ade, fde} ----
    def _forward_metrics(self):
        with torch.no_grad():
            pred = self._work_model(self.X_val)     # (B, T, 4) 归一化 [dLat, dLon, SOG, COG]
            pred = torch.nan_to_num(pred, nan=0.0, posinf=1e4, neginf=-1e4)
            # 反归一化: 归一化 -> 度(dLat/dLon)
            if self.feat_min is not None and self.feat_max is not None:
                feat_min = torch.as_tensor(self.feat_min, device=self.device, dtype=pred.dtype)
                feat_max = torch.as_tensor(self.feat_max, device=self.device, dtype=pred.dtype)
                pred = pred * (feat_max - feat_min) + feat_min
            pred_dlat = pred[:, :, 0]   # 预测相对锚点纬度位移(度)
            pred_dlon = pred[:, :, 1]   # 预测相对锚点经度位移(度)
            true_dlat = self.Y_raw_val[:, :, 0]   # 真实位移(度)
            true_dlon = self.Y_raw_val[:, :, 1]
            # 反算绝对位置 = 锚点 + 位移
            anchor_lat = self.ref_lats_val.unsqueeze(1)
            anchor_lon = self.ref_lons_val.unsqueeze(1)
            pred_lat = anchor_lat + pred_dlat
            pred_lon = anchor_lon + pred_dlon
            true_lat = anchor_lat + true_dlat
            true_lon = anchor_lon + true_dlon
            dlat = torch.deg2rad(pred_lat - true_lat)
            dlon = torch.deg2rad(pred_lon - true_lon)
            a = torch.sin(dlat / 2) ** 2 + torch.cos(torch.deg2rad(true_lat)) * torch.cos(
                torch.deg2rad(pred_lat)) * torch.sin(dlon / 2) ** 2
            dists = 3440.065 * 2 * torch.asin(torch.sqrt(a))
        return {'ade': float(dists.mean()), 'fde': float(dists[:, -1].mean())}

    def _evaluate(self, lambdas, inject_fault=False):
        self._apply_bounds(lambdas)
        if not inject_fault:
            self._restore_weights()
            return self._forward_metrics()
        mets = []
        for _ in range(self.n_fault_trials):
            self._restore_weights()
            self._inject_weight_faults()
            mets.append(self._forward_metrics())
        agg = {}
        for k in ('ade', 'fde'):
            vals = [m[k] for m in mets]
            if self.reduce == 'median':
                agg[k] = float(np.median(vals))
            else:
                agg[k] = float(np.mean(vals))
        return agg

    def _score(self, metrics):
        m = self.reward_metric
        if m == 'ade':
            return metrics['ade']
        if m == 'fde':
            return metrics['fde']
        return metrics['ade'] + metrics['fde']  # 'ade_fde'

    def evaluate(self, action):
        """评估一组 action, 返回 (reward, info)。老设计 reward。"""
        lambdas = self._action_to_lambda(action)
        lambdas_arr = np.array(lambdas, dtype=np.float32)

        clean_metrics = self._evaluate(lambdas_arr, inject_fault=False)
        fault_metrics = self._evaluate(lambdas_arr, inject_fault=True)

        fault_score = self._score(fault_metrics)
        clean_score = self._score(clean_metrics)
        base_score = self._score(self.baseline_metrics)

        # 归一化 reward: 用相对退化比 + log 压缩, 量级统一且跨数据集可比。
        #   fault_deg = fault/base (>=1, 无退化=1), clean_deg = clean/base (≈1)
        #   reward = -log(fault_deg) 量级 ~-3~-5, 对 critic 友好; 方向不变(退化越小越好)
        base = base_score if base_score > 1e-6 else 1e-6
        fault_deg = fault_score / base
        clean_deg = clean_score / base
        reward = -math.log(max(fault_deg, 1e-6))
        # 硬约束: clean 相对退化惩罚(超过 tolerance 才罚, 与主目标同量纲)
        penalty = max(0.0, clean_deg - (1.0 + self.tolerance))
        reward -= self.penalty_weight * penalty
        # 稀疏引导: 鼓励有效保护 (λ < max_act)
        n_protected = sum(1 for i, key in enumerate(self.layer_keys) if lambdas[i] < self.max_acts[key])
        protect_ratio = n_protected / self.action_dim
        reward += self.protect_bonus_weight * protect_ratio
        if not np.isfinite(reward):
            reward = -1e6

        info = {"clean_score": clean_score, "fault_score": fault_score,
                "penalty": penalty, "protect_ratio": protect_ratio, "lambdas": lambdas}
        return float(reward), info


# =============================================================================
# Part 8: Replay Buffer & SAC Agent (bandit, 同 gabo_SAC.py)
# =============================================================================
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = []; self.capacity = capacity; self.pos = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity: self.buffer.append(None)
        self.buffer[self.pos] = (state, action, reward, next_state, done)
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), min(batch_size, len(self.buffer)), replace=False)
        batch = [self.buffer[i] for i in indices]
        return (np.array([b[0] for b in batch], dtype=np.float32),
                np.array([b[1] for b in batch], dtype=np.float32),
                np.array([b[2] for b in batch], dtype=np.float32),
                np.array([b[3] for b in batch], dtype=np.float32),
                np.array([b[4] for b in batch], dtype=np.float32))

    def __len__(self): return len(self.buffer)


class SACAgent:
    def __init__(self, state_dim, action_dim, device, lr=3e-4, tau=0.005, alpha=0.2, auto_entropy=True, hidden_dim=128):
        self.device = device; self.tau = tau; self.alpha = alpha; self.auto_entropy = auto_entropy
        self.actor = SACActor(state_dim, action_dim, hidden_dim).to(device)
        self.critic = SACCritic(state_dim, action_dim, hidden_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic).to(device)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=lr)
        if auto_entropy:
            self.target_entropy = -action_dim
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=lr)
            self.alpha = self.log_alpha.exp().item()
        self.buffer = ReplayBuffer(10000)

    def select_action(self, state):
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad(): action, _ = self.actor.sample(state_t)
        return action.squeeze(0).cpu().numpy()

    def push(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)

    def update(self, batch_size=64):
        if len(self.buffer) < batch_size: return {}
        states, actions, rewards, next_states, dones = self.buffer.sample(batch_size)
        states_t = torch.tensor(states, device=self.device)
        actions_t = torch.tensor(actions, device=self.device)
        rewards_t = torch.tensor(rewards, device=self.device).unsqueeze(1)
        next_states_t = torch.tensor(next_states, device=self.device)

        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_states_t)
            q1_tgt, q2_tgt = self.critic_target(next_states_t, next_action)
            q_tgt = torch.min(q1_tgt, q2_tgt) - self.alpha * next_log_prob
            target = rewards_t  # 单步 MDP (bandit)

        q1, q2 = self.critic(states_t, actions_t)
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        self.critic_optim.zero_grad(); critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0); self.critic_optim.step()

        new_action, log_prob = self.actor.sample(states_t)
        q1_new, q2_new = self.critic(states_t, new_action)
        actor_loss = (self.alpha * log_prob - torch.min(q1_new, q2_new)).mean()
        self.actor_optim.zero_grad(); actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0); self.actor_optim.step()

        if self.auto_entropy:
            alpha_loss = -(self.log_alpha * (log_prob.detach() + self.target_entropy)).mean()
            self.alpha_optim.zero_grad(); alpha_loss.backward(); self.alpha_optim.step()
            self.alpha = self.log_alpha.exp().item()

        with torch.no_grad():
            for p, p_tgt in zip(self.critic.parameters(), self.critic_target.parameters()):
                p_tgt.data.mul_(1.0 - self.tau).add_(p.data * self.tau)

        return {"critic_loss": critic_loss.item(), "actor_loss": actor_loss.item(), "alpha": self.alpha}


# =============================================================================
# Part 9: 主入口 — 两阶段 SAC (对齐老设计)
# =============================================================================
def SAC_bounds(model, data_loader_dict, device, bound_type='layer', bitflip='fixed',
               reward_metric='fde', fault_rate=1e-6,
               gabo_max_samples=2048, gabo_protect_ratio=0.95,
               episodes=30, steps_per_episode=10, sac_lr=3e-4,
               sac_batch_size=64, sac_updates_per_step=2,
               tolerance=0.05, penalty_weight=10.0, protect_bonus_weight=0.0,
               action_scale=0.5, n_fault_trials=5, eval_samples=2048,
               reduce='mean', clamp_to_ranger=True, protect_ratio=0.85,
               feat_min=None, feat_max=None,
               total_bits=32, n_frac=16, n_int=15, verbose=True):
    """SAC 边界搜索主入口(轨迹回归, 两阶段, 对齐老设计)。

    ``reward_metric``        : 'ade' | 'fde' | 'ade_fde'。
    ``protect_bonus_weight`` : 老设计奖励里的稀疏保护比例奖励权重。
    ``action_scale``         : λ = max_act * exp(action * scale) 的双向搜索幅度。
    ``clamp_to_ranger``      : True 时 λ 不允许超过 Ranger(=max_act)。
    ``protect_ratio``        : 搜索下界 max_act * protect_ratio。
    """
    if 'sub_train' not in data_loader_dict:
        raise ValueError("SAC_bounds requires 'sub_train' key in data_loader_dict")

    print("=" * 70)
    print(" SAC: Two-phase Bound Optimisation (GABO init + bandit SAC)")
    print("=" * 70)

    # Phase 1: GABO
    print("\n[Phase 1] GABO — per-layer gradient ascent")
    init_lambdas, bounds, tresh, layer_stats, mse_clean, max_acts = gabo_phase(
        model=model, train_loader=data_loader_dict, device=device,
        max_samples=gabo_max_samples, protect_ratio=gabo_protect_ratio,
        tolerance=tolerance, penalty_weight=penalty_weight, verbose=verbose)

    # ReLU key / path 按模块顺序对齐
    relu_paths = [n for n, m in model.named_modules() if isinstance(m, nn.ReLU)]
    layer_keys = [_ranger_key(p) for p in relu_paths]
    layer_keys = [k for k in layer_keys if k in max_acts]
    relu_paths = [p for p in relu_paths if _ranger_key(p) in max_acts]
    action_dim = len(layer_keys)
    if action_dim == 0:
        return {}, {}, None

    print(f"\n[GABO] clean MSE baseline : {mse_clean:.6f}")
    print(f"[GABO] layers to optimise : {action_dim}")

    # Phase 2: SAC (bandit, 固定 state, 直接双向 λ)
    print(f"\n[Phase 2] SAC — bandit fine-tuning "
          f"({'Shrink-only, λ ≤ Ranger' if clamp_to_ranger else 'Bidirectional'})")
    env = BoundEvalEnv(
        model=model, data_loader=data_loader_dict['sub_train'], max_acts=max_acts,
        layer_keys=layer_keys, relu_paths=relu_paths, device=device,
        fault_rate=fault_rate, bitflip=bitflip, total_bits=total_bits, n_frac=n_frac, n_int=n_int,
        reward_metric=reward_metric,
        tolerance=tolerance, penalty_weight=penalty_weight, protect_bonus_weight=protect_bonus_weight,
        n_fault_trials=n_fault_trials, action_scale=action_scale, reduce=reduce,
        clamp_to_ranger=clamp_to_ranger, eval_samples=eval_samples,
        protect_ratio=protect_ratio, feat_min=feat_min, feat_max=feat_max)

    agent = SACAgent(state_dim=env.state_dim, action_dim=action_dim, device=device, lr=sac_lr)
    state_np = env.state.squeeze(0).cpu().numpy()

    # 热启动: GABO 结果反推 action
    gabo_action = []
    for key in layer_keys:
        m = max_acts[key]
        lam_gabo = float(init_lambdas[key])
        act_val = math.log(max(lam_gabo, 1e-8) / max(m, 1e-8)) / action_scale
        gabo_action.append(np.clip(act_val, -1.0, 1.0))
    gabo_action = np.array(gabo_action, dtype=np.float32)

    reward_gabo, info_gabo = env.evaluate(gabo_action)
    agent.push(state_np, gabo_action, reward_gabo, state_np, True)
    print(f"[SAC] GABO init → reward={reward_gabo:.6f}  clean={info_gabo['clean_score']:.4f}  "
          f"fault={info_gabo['fault_score']:.4f}")

    # 预热 buffer
    for _ in range(20):
        noise = np.random.normal(0, 0.1, size=action_dim).astype(np.float32)
        noisy_action = np.clip(gabo_action + noise, -1.0, 1.0)
        r, _ = env.evaluate(noisy_action)
        agent.push(state_np, noisy_action, r, state_np, True)

    best_reward = reward_gabo; best_action = gabo_action.copy(); best_info = info_gabo.copy()

    for ep in range(episodes):
        ep_rewards = []
        for _ in range(steps_per_episode):
            action = agent.select_action(state_np)
            action = np.clip(action, -1.0, 1.0)
            reward, info = env.evaluate(action)
            agent.push(state_np, action, reward, state_np, True)
            ep_rewards.append(reward)
            if reward > best_reward:
                best_reward = reward; best_action = action.copy(); best_info = info.copy()
            for _ in range(sac_updates_per_step):
                agent.update(batch_size=sac_batch_size)
        if (ep + 1) % 10 == 0 or ep == 0:
            mean_r = np.mean(ep_rewards) if ep_rewards else 0.0
            print(f"  [SAC] Ep {ep + 1:3d}/{episodes}  mean_r={mean_r:.4f}  best_r={best_reward:.6f}  "
                  f"α={agent.alpha:.4f}  clean={best_info['clean_score']:.4f}  "
                  f"fault={best_info['fault_score']:.4f}  prot={best_info['protect_ratio']:.2f}")

    # 最终结果
    results = {}
    final_lambdas = env._action_to_lambda(best_action)
    for i, key in enumerate(layer_keys):
        results[key] = torch.tensor(final_lambdas[i], device=device)

    print(f"\n{'=' * 70}\n SAC Complete!\n{'=' * 70}")
    print(f"  Best reward      : {best_reward:.6f}")
    print(f"  GABO init reward : {reward_gabo:.6f}")
    print(f"  Improvement      : {best_reward - reward_gabo:+.6f}")
    print(f"\n  Final bounds (λ_sac vs Ranger / GABO):")
    for i, key in enumerate(layer_keys):
        m = max_acts[key]
        lam_sac = final_lambdas[i]
        lam_gabo = float(init_lambdas[key])
        status = "SHRINK" if lam_sac < m else "EXPAND" if lam_sac > m else "=RANGER"
        print(f"    {key}: λ_sac={lam_sac:.4f}  ranger={m:.4f}  λ/ranger={lam_sac / max(m, 1e-12):.3f}  "
              f"(gabo={lam_gabo:.4f}) [{status}]")
    print(f"{'=' * 70}")

    return results, tresh, None
