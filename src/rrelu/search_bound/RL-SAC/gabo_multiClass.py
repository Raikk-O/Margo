# -*- coding: utf-8 -*-
"""
GABO-SAC: Gradient Ascent Bound + Soft Actor-Critic (Bidirectional Search)
==========================================================================
Phase 1 (GABO): Per-layer gradient ascent → initial λ + search space anchor
Phase 2 (SAC) : Global joint fine-tuning with bidirectional bound search
                (can both SHRINK and EXPAND bounds beyond GABO/Ranger limits)

Public API:
    results_DMA, tresh, alpha = GABO_SAC_bounds(model, train_loader, device, ...)
"""

import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam


# =============================================================================
# Part 0: Statistics helpers (from Args_ana.py)
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


# =============================================================================
# Part 1: Hook helpers
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


def _ordered_relu_keys(model):
    """按 ``named_modules()`` 的 DFS 顺序返回各 ReLU 的 hook key。

    ★ 为什么不能用 ``sorted(max_acts.keys())``:
      key 形如 ``'2net'`` / ``'10net'``, 字典序(``10net < 2net``)与模块顺序
      (net.2, net.6, net.10 ...) 不一致。而 ``BoundEvalEnv.evaluate`` 是按
      ``named_modules()`` 顺序把 action 维度绑到 ReLU 上的, 排序不一致会让
      第 k 个 action 的 λ_k 落到错误的层上 —— SAC 能跑, 但优化的是噪声。
    """
    keys = []

    def _walk(module, prefix=''):
        for name1, layer in module.named_children():
            if len(list(layer.children())) == 0:
                if isinstance(layer, nn.ReLU):
                    keys.append(name1 + prefix)
            else:
                _walk(layer, prefix + name1)

    _walk(model, '')
    return keys


# =============================================================================
# Part 2: Adaptive learning rate
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
# Part 3: GABO core (gradient ascent for regression)
# =============================================================================
def _gradient_ascent_bound_regression(
        margin0, grad_act, act, init_lambda,
        lower_limit, upper_limit, per_neuron,
        margin_clean_baseline, tolerance=0.05, penalty_weight=10.0,
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

    for it in range(max_iter):
        deviation = torch.clamp(lam - act, max=0.0)  # min(λ-a, 0): 低于阈值偏差为 0
        perturb = (grad_act * deviation).sum(dim=1)
        margin_perturbed = margin0 + perturb      # 正确类 logit 的一阶外推
        margin_mean = margin_perturbed.mean()
        penalty = F.relu(margin_clean_baseline * (1.0 - tolerance) - margin_mean)
        objective = margin_mean - penalty_weight * penalty
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
# Part 4: GABO Phase
# =============================================================================
def gabo_phase(model, train_loader, device, max_samples=2048, protect_ratio=0.95,
               tolerance=0.05, penalty_weight=10.0,
               max_iter=50, patience=6, initial_lr_factor=0.2, scale_factor=2,
               dynamic_range_factor=1.0, verbose=False):
    model.eval(); model = model.to(device)
    activations = {}; handles = []
    _register_relu_hooks(model, activations, handles, name='')
    loader = train_loader['sub_train']
    stats = {}; all_preds = []; all_labels = []; n_seen = 0

    for data, label in loader:
        data, label = data.to(device), label.to(device)
        activations.clear()
        out = model(data)                                             # [B, C] 多分类 logits
        correct_logit = out.gather(1, label.view(-1, 1).long()).sum()  # 正确类 logit 之和(求梯度)
        for key, act in activations.items():
            grad_act = torch.autograd.grad(correct_logit, act, retain_graph=True, allow_unused=True)[0]
            if grad_act is None: continue
            st = stats.setdefault(key, {"act": [], "grad": []})
            st["act"].append(act.detach()); st["grad"].append(grad_act.detach())
        all_preds.append(out.detach()); all_labels.append(label.detach().long().view(-1))
        n_seen += data.shape[0]
        if n_seen >= max_samples: break

    for h in handles: h.remove()
    if not stats: return {}, {}, {}, {}, 0.0, {}

    logits = torch.cat(all_preds, dim=0); labels = torch.cat(all_labels, dim=0)
    correct = logits.gather(1, labels.unsqueeze(1)).squeeze(1)    # [N] 正确类 logit
    margin0 = correct                                              # [N] 越大越好
    clean_acc = float((logits.argmax(1) == labels).float().mean())  # clean 准确率(Phase 2 保护用)
    margin_clean = float(margin0.mean())
    init_lambdas, bounds_dict, tresh_dict, layer_stats, max_acts = {}, {}, {}, {}, {}

    for key, st in stats.items():
        act = torch.cat(st["act"], dim=0); grad_act = torch.cat(st["grad"], dim=0)
        act_np = act.detach().cpu().numpy().astype(np.float64)
        batch_means = [a.mean(dim=0) for a in st["act"]]
        mins = batch_means[0]
        for m in batch_means[1:]: mins = torch.minimum(mins, m)
        tresh_dict[key] = torch.min(mins)

        flat = act_np.ravel()
        lo_iqr, up_iqr = iqr_method(flat, float(flat.mean()), float(flat.std()), _feature_entropy(flat), _skewness(flat))
        max_act = float(act_np.max()); std_act = float(act_np.std()); mean_grad = float(grad_act.abs().mean())
        up_iqr = _dynamically_adjust_upper_bound(up_iqr, max_act)
        up_iqr = min(up_iqr, max_act)
        lower_limit, upper_limit = _dynamic_upper_bound_range(up_iqr, std_act, dynamic_range_factor)
        upper_limit = min(upper_limit, max_act)
        if protect_ratio > 0: lower_limit = max(lower_limit, max_act * protect_ratio)
        upper_limit = max(upper_limit, lower_limit)

        bound = _gradient_ascent_bound_regression(
            margin0=margin0, grad_act=grad_act, act=act, init_lambda=float(up_iqr),
            lower_limit=lower_limit, upper_limit=upper_limit, per_neuron=False,
            margin_clean_baseline=margin_clean, tolerance=tolerance, penalty_weight=penalty_weight,
            max_iter=max_iter, patience=patience, initial_lr_factor=initial_lr_factor,
            scale_factor=scale_factor, device=device, verbose=verbose)

        init_lambdas[key] = bound
        bounds_dict[key] = (float(lower_limit), float(upper_limit))
        max_acts[key] = max_act
        layer_stats[key] = {"max_act": max_act, "std_act": std_act, "mean_grad": mean_grad, "iqr_range": up_iqr - lo_iqr}

    return init_lambdas, bounds_dict, tresh_dict, layer_stats, clean_acc, max_acts


# =============================================================================
# Part 5: SAC Networks
# =============================================================================
class SACActor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
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
        return action, log_prob  # action in [-1, 1]

    def deterministic(self, state):
        mu, _ = self.forward(state)
        return torch.tanh(mu)


class SACCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.q1 = nn.Sequential(nn.Linear(state_dim + action_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
        self.q2 = nn.Sequential(nn.Linear(state_dim + action_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa), self.q2(sa)


# =============================================================================
# Part 6: Bidirectional Bound Evaluation Environment
# =============================================================================
class BoundEvalEnv:
    """
    评估环境: 接收 action ∈ [-1, 1]^L，映射到 λ_k = max_act_k * exp(action_k * scale)
    支持边界收缩 (action < 0) 和 上升 (action > 0)。

    ★ clamp_to_ranger=True (默认) 时, λ 会被裁剪到不超过 max_act(=Ranger 上界):
      λ > max_act 意味着该层在训练集分布上永远不会触发裁剪 —— 保护被静默关闭,
      只有 val/test 上出现比训练集更大的激活时才会偶尔生效。容错场景下应关闭放大。
    """
    def __init__(self, model, data_loader, max_acts, layer_keys, device,
                 mse_clean_baseline, fault_injector=None,
                 tolerance=0.05, penalty_weight=10.0, protect_bonus_weight=1.0,
                 eval_samples=2048, n_fault_trials=5, action_scale=0.5,
                 clamp_to_ranger=True):
        self.model_template = model
        self.device = device
        self.max_acts = max_acts  # dict: key -> max_act (Ranger bound)
        self.layer_keys = layer_keys
        self.action_dim = len(layer_keys)
        self.mse_clean = mse_clean_baseline
        # 模块名 -> hook key ('net.2' -> '2net'), 与 _register_relu_hooks 的命名一致。
        # evaluate() 按名字查 λ, 彻底摆脱对遍历顺序的依赖。
        self.relu_key_by_module_name = {}
        for mname, module in model.named_modules():
            if isinstance(module, nn.ReLU):
                parts = mname.split('.')
                self.relu_key_by_module_name[mname] = parts[-1] + ''.join(parts[:-1])
        self.tolerance = tolerance
        self.penalty_weight = penalty_weight
        self.protect_bonus_weight = protect_bonus_weight
        self.fault_injector = fault_injector
        self.n_fault_trials = n_fault_trials
        self.action_scale = action_scale  # 控制搜索幅度，0.5 意味着 exp(±0.5) ≈ [0.6, 1.65]
        self.clamp_to_ranger = clamp_to_ranger  # True: λ 不允许超过 Ranger(=max_act)

        # 预加载评估数据(sub_train, 与 Phase1 激活统计口径一致; 不能用 val 防泄漏)
        self.eval_data = []
        n = 0
        for data, label in data_loader['sub_train']:
            self.eval_data.append((data.to(device), label.to(device)))
            n += data.shape[0]
            if n >= eval_samples: break

        # 构建 state 向量 (固定上下文)
        state_parts = []
        for key in layer_keys:
            m = max_acts[key]
            state_parts.extend([m, m * 0.1, m * 0.5, m * 2.0])  # 给 SAC 提供量级感知
        self.state = torch.tensor(state_parts, dtype=torch.float32, device=device).unsqueeze(0)

    @property
    def state_dim(self): return self.state.shape[1]

    def _action_to_lambda(self, action):
        """将 [-1, 1] 的 action 映射到实际 λ

        λ = max_act * exp(action * scale)，可收缩(action<0)也可放大(action>0)；
        clamp_to_ranger=True 时把放大侧裁掉，保证 λ <= max_act。
        """
        lambdas = []
        for i, key in enumerate(self.layer_keys):
            m = self.max_acts[key]
            # λ = max_act * exp(action * scale)
            lam = m * math.exp(float(action[i]) * self.action_scale)
            if self.clamp_to_ranger:
                lam = min(lam, m)
            lambdas.append(lam)
        return lambdas

    def _make_bound_hook(self, lam_val):
        def hook(module, inp):
            return (inp[0].clamp(max=lam_val),)
        return hook

    def evaluate(self, action):
        """评估一组 action ∈ [-1, 1]^L"""
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()

        lambdas_actual = self._action_to_lambda(action)

        # 1. 深拷贝模型 + 注册 bounded ReLU hooks
        model = copy.deepcopy(self.model_template).to(self.device).eval()
        # 按 (模块名 -> hook key) 精确绑定, 不依赖遍历顺序
        lam_by_key = {key: lambdas_actual[i] for i, key in enumerate(self.layer_keys)}
        hooks = []
        for mname, module in model.named_modules():
            key = self.relu_key_by_module_name.get(mname)
            if key is not None and key in lam_by_key:
                hooks.append(module.register_forward_pre_hook(self._make_bound_hook(lam_by_key[key])))

        # 2. Clean acc(多分类)
        clean_acc_sum = 0.0; n_total = 0
        for data, label in self.eval_data:
            with torch.no_grad():
                pred = model(data).argmax(1)
                lbl = label.long().view(-1)
            clean_acc_sum += (pred == lbl).sum().item()
            n_total += lbl.shape[0]
        clean_acc = clean_acc_sum / max(n_total, 1)

        # 3. Fault acc(用对齐的 fault_injector)
        fault_acc = clean_acc
        if self.fault_injector is not None:
            fault_acc_sum = 0.0
            for _ in range(self.n_fault_trials):
                for data, label in self.eval_data:
                    with torch.no_grad():
                        pred_fault = self.fault_injector(model, data)
                        if isinstance(pred_fault, tuple): pred_fault = pred_fault[0]
                        pred_fault = pred_fault.argmax(1)
                        lbl = label.long().view(-1)
                    fault_acc_sum += (pred_fault == lbl).sum().item()
            fault_acc = fault_acc_sum / max(n_total * self.n_fault_trials, 1)

        # 4. 清理
        for h in hooks: h.remove()
        del model

        # 5. 计算 Reward (核心设计)
        # 主目标: 最大化故障下的准确率
        reward = fault_acc

        # 硬约束: clean 精度退化惩罚(clean_acc 不能低于 baseline*(1-tolerance))
        penalty = max(0.0, self.mse_clean * (1.0 - self.tolerance) - clean_acc)
        reward -= self.penalty_weight * penalty

        # 稀疏引导: 仅在 clean 约束满足时, 才对"有效保护层数"给少量奖励。
        # ★ 关键: protect_ratio ∈ [0,1] 与 fault_acc 同量级, 权重过大时会把 SAC
        #   引向"全层收缩"(protect_ratio→1.0)而非"精准保护敏感层", 掩盖容错主信号。
        #   默认 protect_bonus_weight=0(关闭), 靠 clean penalty 自然约束过度收缩。
        n_protected = sum(1 for i, key in enumerate(self.layer_keys) if lambdas_actual[i] < self.max_acts[key])
        protect_ratio = n_protected / self.action_dim
        if penalty <= 0.0:
            reward += self.protect_bonus_weight * protect_ratio

        info = {
            "mse_clean": clean_acc,
            "mse_fault": fault_acc,
            "penalty": penalty,
            "protect_ratio": protect_ratio,
            "lambdas": lambdas_actual,
        }
        return float(reward), info


# =============================================================================
# Part 7: Replay Buffer & SAC Agent
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
        self.actor_optim = Adam(self.actor.parameters(), lr=lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=lr)
        if auto_entropy:
            self.target_entropy = -action_dim
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optim = Adam([self.log_alpha], lr=lr)
            self.alpha = self.log_alpha.exp().item()
        self.buffer = ReplayBuffer(10000)

    def select_action(self, state):
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad(): action, _ = self.actor.sample(state_t)
        return action.squeeze(0).cpu().numpy()

    def select_deterministic(self, state):
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad(): action = self.actor.deterministic(state_t)
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
            target = rewards_t  # 单步 MDP

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
# Part 8: 主入口 — GABO-SAC 联合优化 (Bidirectional)
# =============================================================================
def GABO_SAC_bounds(model, train_loader, device="cuda", fault_injector=None,
                     gabo_max_samples=2048, gabo_protect_ratio=0.95,
                     sac_episodes=200, sac_steps_per_episode=10, sac_lr=3e-4,
                     sac_batch_size=64, sac_updates_per_step=2,
                     tolerance=0.05, penalty_weight=10.0, protect_bonus_weight=1.0,
                     action_scale=0.5, n_fault_trials=5, eval_samples=2048,
                     clamp_to_ranger=True, verbose=False):
    """
    GABO-SAC 两阶段联合优化 (支持边界收缩与上升)。

    ``clamp_to_ranger`` (默认 True)
        把 λ 硬裁剪到不超过 Ranger 上界 (= 训练集最大激活 max_act)。
        关闭后允许放大到 max_act * exp(action_scale) ≈ 1.65 倍, 但被放大的那层
        在训练集分布上不会再触发裁剪, 等于关掉该层保护, 容错收益会归零。

    Returns: (results_DMA, tresh, alpha)
    """
    device = device if torch.cuda.is_available() or device == "cpu" else "cpu"
    print("=" * 70)
    print(" GABO-SAC: Bound Optimisation "
          f"({'Shrink-only, λ ≤ Ranger' if clamp_to_ranger else 'Bidirectional, Shrink & Expand'})")
    print("=" * 70)

    # Phase 1: GABO
    print("\n[Phase 1] GABO — Per-layer gradient ascent")
    print("-" * 70)
    init_lambdas, bounds, tresh, layer_stats, mse_clean, max_acts = gabo_phase(
        model=model, train_loader=train_loader, device=device,
        max_samples=gabo_max_samples, protect_ratio=gabo_protect_ratio,
        tolerance=tolerance, penalty_weight=penalty_weight, verbose=verbose)

    # ★ 必须按模块顺序排, 不能 sorted(): '2net'/'10net' 的字典序 != 模块顺序
    layer_keys = [k for k in _ordered_relu_keys(model) if k in max_acts]
    missing = [k for k in max_acts if k not in layer_keys]
    if missing:
        print(f"[warn] {len(missing)} 个 ReLU key 未匹配到模块顺序, 追加到末尾: {missing}")
        layer_keys.extend(sorted(missing))
    action_dim = len(layer_keys)
    if action_dim == 0: return {}, {}, None

    print(f"\n[GABO] Clean acc baseline : {mse_clean:.6f}")
    print(f"[GABO] Layers to optimise : {action_dim}")

    # Phase 2: SAC
    print(f"\n[Phase 2] SAC — Global joint fine-tuning "
          f"({'Shrink-only' if clamp_to_ranger else 'Bidirectional'})")
    print("-" * 70)

    env = BoundEvalEnv(
        model=model, data_loader=train_loader, max_acts=max_acts, layer_keys=layer_keys,
        device=device, mse_clean_baseline=mse_clean, fault_injector=fault_injector,
        tolerance=tolerance, penalty_weight=penalty_weight,
        protect_bonus_weight=protect_bonus_weight,
        eval_samples=eval_samples, n_fault_trials=n_fault_trials, action_scale=action_scale,
        clamp_to_ranger=clamp_to_ranger)

    agent = SACAgent(state_dim=env.state_dim, action_dim=action_dim, device=device, lr=sac_lr)
    state_np = env.state.squeeze(0).cpu().numpy()

    # 热启动: 用 GABO 结果反推 action
    gabo_action = []
    for key in layer_keys:
        m = max_acts[key]
        lam_gabo = float(init_lambdas[key])
        # action = log(lam / m) / scale
        act_val = math.log(max(lam_gabo, 1e-8) / max(m, 1e-8)) / action_scale
        gabo_action.append(np.clip(act_val, -1.0, 1.0))
    gabo_action = np.array(gabo_action, dtype=np.float32)

    reward_gabo, info_gabo = env.evaluate(gabo_action)
    agent.push(state_np, gabo_action, reward_gabo, state_np, True)
    print(f"[SAC] GABO init → reward={reward_gabo:.6f}  clean={info_gabo['mse_clean']:.6f}  fault={info_gabo['mse_fault']:.6f}")

    # 预热 buffer: 覆盖收缩区(GABO 起点≈不收缩, 若只围绕它探索, SAC 一开始
    # 根本见不到"收缩提升 fault"的信号, 收敛慢且随机性大)。
    # 一半围绕 GABO 加噪声, 一半均匀采样收缩区 action∈[-1,0]。
    rng_warm = np.random.default_rng(0)
    for i in range(40):
        if i % 2 == 0:
            noise = rng_warm.normal(0, 0.2, size=action_dim).astype(np.float32)
            a = np.clip(gabo_action + noise, -1.0, 1.0)
        else:
            a = rng_warm.uniform(-1.0, 0.0, size=action_dim).astype(np.float32)
        r, _ = env.evaluate(a)
        agent.push(state_np, a, r, state_np, True)

    best_reward = reward_gabo; best_action = gabo_action.copy(); best_info = info_gabo.copy()

    # SAC 训练循环
    for ep in range(sac_episodes):
        ep_rewards = []
        for step in range(sac_steps_per_episode):
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
            print(f"  [SAC] Ep {ep+1:3d}/{sac_episodes}  mean_r={mean_r:.4f}  best_r={best_reward:.6f}  "
                  f"α={agent.alpha:.4f}  clean={best_info['mse_clean']:.6f}  fault={best_info['mse_fault']:.6f}  "
                  f"prot={best_info['protect_ratio']:.2f}")

    # 构建最终结果
    results = {}
    final_lambdas = env._action_to_lambda(best_action)
    for i, key in enumerate(layer_keys):
        results[key] = torch.tensor(final_lambdas[i], device=device)

    print(f"\n{'='*70}")
    print(f" GABO-SAC Complete!")
    print(f"{'='*70}")
    print(f"  Best reward      : {best_reward:.6f}")
    print(f"  GABO init reward : {reward_gabo:.6f}")
    print(f"  Improvement      : {best_reward - reward_gabo:+.6f}")
    print(f"  Best clean MSE   : {best_info['mse_clean']:.6f}")
    print(f"  Best fault MSE   : {best_info['mse_fault']:.6f}")
    print(f"  Protect ratio    : {best_info['protect_ratio']:.2f}")
    print(f"\n  Final bounds (λ_sac vs Ranger / GABO):")
    for i, key in enumerate(layer_keys):
        m = max_acts[key]                       # Ranger 上界 = 训练集最大激活
        lam_sac = final_lambdas[i]
        lam_gabo = float(init_lambdas[key])
        delta = lam_sac - lam_gabo
        status = "SHRINK" if lam_sac < m else "EXPAND" if lam_sac > m else "=RANGER"
        print(f"    {key}: λ_sac={lam_sac:.4f}  ranger={m:.4f}  λ/ranger={lam_sac / max(m, 1e-12):.3f}  "
              f"(gabo={lam_gabo:.4f}, Δ={delta:+.4f}) [{status}]")
    print(f"{'='*70}")

    return results, tresh, None