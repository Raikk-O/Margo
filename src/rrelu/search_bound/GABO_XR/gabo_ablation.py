# -*- coding: utf-8 -*-
"""
GABO_bounds (ablation variant) — GABO 初始化统计量消融版
================================================================================
基于 ``gabo_multiclass.py`` 的多分类 GABO 移植, 额外支持对边界初始化阶段的
三个"分布感知统计量"做 only-one 消融(每次只保留单个统计量, 考察其独立贡献):

  * 特征熵   entropy   : ``custom_entropy * 0.05``  -> 上百分位数 P_high 的 β·H 项
  * 偏度     skewness  : ``abs(custom_skewness)``   -> 上下百分位数的 |γ| 项
  * 自适应IQR adaptive : ``np.log(n) * 0.5``       -> 样本量自适应项 α·lnN

消融模式(``ablation`` 参数):
  "full"          完整 GABO (熵 + 偏度 + 自适应IQR)
  "entropy_only"  只保留特征熵     (β·H ≠ 0, |γ| = 0, α·lnN = 0)
  "skew_only"     只保留偏度       (|γ| ≠ 0, β·H = 0, α·lnN = 0)
  "adaptive_only" 只保留自适应IQR  (α·lnN ≠ 0, |γ| = 0, β·H = 0)

对外契约与 Ranger_bounds / gabo_multiclass.GABO_bounds 完全一致:
    results, tresh, alpha = GABO_bounds(model, data_loader_dict, device, ...)

其余逻辑(梯度上升、分类 margin 目标、保护比例、收缩因子、自适应学习率)
与 gabo_multiclass.py 保持 1:1 一致, 仅初始化统计量可开关。
"""

import math

import numpy as np
import torch
import torch.nn as nn


# =============================================================================
# statistics helpers (ported from src/ARGS/Args_ana.py, pure numpy)
# =============================================================================
def _feature_entropy(x, n_bins=256):
    """Shannon entropy (bits) of the histogram of ``x``."""
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size == 0:
        return 0.0
    mn = x.min()
    if mn < 0:
        x = x - mn
    mx = x.max()
    if mx != 0:
        x = x / mx
    q = (x * 255).astype(np.int64)
    counts = np.bincount(q + 1, minlength=n_bins).astype(np.float64)
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def _skewness(x):
    """Sample skewness of ``x``."""
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size == 0:
        return 0.0
    s = x.std()
    if s == 0:
        return 0.0
    return float(((x - x.mean()) ** 3).mean() / (s ** 3))


def iqr_method(features, custom_mean, custom_std, custom_entropy, custom_skewness,
               use_entropy=True, use_skewness=True, use_adaptive=True,
               bound_method="shrink_iqr_adaptive"):
    """
    IQR based bounds with dynamic percentiles/factor.

    与 Args_ana.iqr_method 1:1 移植, 额外暴露三个自适应项的开关:

      use_entropy  : 是否启用特征熵项 ``custom_entropy * 0.05`` (β·H)
      use_skewness : 是否启用偏度项 ``abs(custom_skewness)`` (|γ|)
      use_adaptive : 是否启用样本量自适应项 ``np.log(n) * 0.5`` (α·lnN)

    bound_method:
      "shrink_iqr_adaptive" : 自适应 IQR 收缩 (统计量调制分位点的自适应 IQR)
    """
    features = np.asarray(features, dtype=np.float64).ravel()
    n = len(features)
    if n == 0:
        return 0.0, 0.0

    if bound_method == "shrink_iqr_adaptive":
        # 自适应 IQR 收缩 (v6): 参照 Tukey 的统计量调制分位点(样本量/偏度/熵),
        # 得到自适应 IQR, 但上界用线性归一化映射到 [0.7, 0.95](而非 q3+factor·IQR)。
        max_act = float(np.max(features))
        pos = features[features > 0]
        if pos.size == 0:
            return float(max_act * 0.60), float(max_act * 0.95)
        q1 = np.percentile(pos, 25.0)
        q3 = np.percentile(pos, 75.0)
        iqr = q3 - q1
        # IQR 主项: 相对离散度驱动的自适应收缩
        iqr_ratio = float(np.clip(iqr / max_act, 0.0, 1.0))
        base = 0.95 - 0.35 * iqr_ratio
        # 统计量修正项: 归一化后直接加减在收缩比例上(偏度 + 熵)
        skew_norm = float(np.tanh(abs(custom_skewness) / 2.0)) if use_skewness else 0.0
        entropy_norm = min(custom_entropy / 8.0, 1.0) if use_entropy else 0.0
        shrink_ratio = float(np.clip(base - 0.10 * skew_norm - 0.10 * entropy_norm, 0.60, 0.95))
        lower = float(max_act * 0.60)
        upper = float(max_act * shrink_ratio)
        print(f"    [shrink_iqr_adaptive] n={n} n_pos={pos.size}({100.0 * pos.size / n:.1f}%) "
              f"max_act={max_act:.4f}")
        print(f"        stats : mean={custom_mean:.4f} std={custom_std:.4f} "
              f"entropy={custom_entropy:.3f} skew={custom_skewness:.3f}")
        print(f"        terms: skew={skew_norm:.3f} ent={entropy_norm:.3f} "
              f"q1={q1:.4f} q3={q3:.4f} IQR={iqr:.4f} iqr/max={iqr / max_act:.4f}")
        print(f"        shrink_ratio={shrink_ratio:.4f} (range=0.7~0.95)")
        print(f"        -> lower={lower:.4f} upper={upper:.4f} upper/max={upper / max(max_act, 1e-12):.3f}")
        return lower, upper


def _dynamic_upper_bound_range(final_upper_bound, custom_std, factor=1.0):
    adjustment = custom_std * factor
    return float(final_upper_bound - adjustment), float(final_upper_bound + adjustment)


def _dynamically_adjust_upper_bound(final_upper_bound, custom_max_value,
                                    threshold_ratio=0.5, correction_factor=0.9):
    if final_upper_bound < custom_max_value * threshold_ratio:
        return float(custom_max_value * correction_factor)
    return float(final_upper_bound)


# =============================================================================
# hook helpers — key naming kept identical to ranger.py
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
# adaptive learning rate (ported from GABO_imp_test5_2.adjust_learning_rate)
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
# core: gradient ascent on the bound (与 gabo_multiclass.py 完全一致)
# =============================================================================
def _gradient_ascent_bound(margin0, sign_s, grad_act, act, init_lambda,
                           lower_limit, upper_limit, per_neuron,
                           max_iter=50, patience=6,
                           initial_lr_factor=0.2, scale_factor=2,
                           device="cpu", verbose=False):
    dtype = act.dtype
    if per_neuron:
        lam = torch.full((act.shape[1],), float(init_lambda),
                         device=device, dtype=dtype, requires_grad=True)
        lower_limit = torch.as_tensor(lower_limit, device=device, dtype=dtype)
        upper_limit = torch.as_tensor(upper_limit, device=device, dtype=dtype)
    else:
        lam = torch.tensor(float(init_lambda), device=device, dtype=dtype, requires_grad=True)
        lower_limit = torch.as_tensor(float(lower_limit), device=device, dtype=dtype)
        upper_limit = torch.as_tensor(float(upper_limit), device=device, dtype=dtype)

    lr = None
    prev_grad = None
    patience_counter = None
    best_obj = -float("inf")
    best_lam = lam.detach().clone()
    obj_stack = []
    delta = 1e-8

    for _ in range(max_iter):
        deviation = torch.clamp(lam - act, max=0.0)  # min(λ-a, 0): 低于阈值偏差为 0
        perturb = (grad_act * deviation).sum(dim=1)
        margin = margin0 + sign_s * perturb
        result = margin.clamp(-1.0, 1.0)

        log_sum = torch.log(result.abs().clamp_min(1e-12)).sum()
        sign_prod = torch.sign(result).prod()
        objective = log_sum * sign_prod

        obj_val = float(objective.detach())
        obj_stack.append(obj_val)

        if obj_val > best_obj and float(lam.detach().min()) > 0:
            best_obj = obj_val
            best_lam = lam.detach().clone()

        if verbose:
            print(f"      lambda={float(lam.detach().mean()):.4f}  obj={obj_val:.6f}  best={best_obj:.6f}")

        if len(obj_stack) > patience:
            recent = obj_stack[-patience:]
            changes = [abs(recent[i] - recent[i - 1]) for i in range(1, len(recent))]
            trend = [recent[i] <= recent[i - 1] for i in range(1, len(recent))]
            if all(c < delta for c in changes) and all(trend):
                break

        grad = torch.autograd.grad(objective, lam, retain_graph=False)[0]
        if grad is None or not torch.isfinite(grad).all() or float(grad.abs().max()) == 0:
            break

        lr, patience_counter = _adjust_learning_rate(
            grad, prev_grad, lr,
            initial_lr_factor=initial_lr_factor,
            scale_factor=scale_factor,
            patience_counter=patience_counter,
        )
        prev_grad = grad.detach()

        if not torch.isfinite(lr).all() or float(lr.abs().max()) == 0:
            break

        with torch.no_grad():
            lam_new = lam + lr * grad
            lam_new = torch.round(lam_new * 1e4) / 1e4
            lam.data.copy_(lam_new)

    if verbose:
        print(f"    -> searched lambda mean={float(best_lam.mean()):.4f} "
              f"clip range=[{float(lower_limit.mean()):.4f}, {float(upper_limit.mean()):.4f}]")

    return best_lam.detach().clamp(lower_limit, upper_limit)


# =============================================================================
# ablation 模式定义
# =============================================================================
ABLATION_MODES = {
    "full":          dict(use_entropy=True,  use_skewness=True,  use_adaptive=True),
    "entropy_only":  dict(use_entropy=True,  use_skewness=False, use_adaptive=False),
    "skew_only":     dict(use_entropy=False, use_skewness=True,  use_adaptive=False),
    "adaptive_only": dict(use_entropy=False, use_skewness=False, use_adaptive=True),
}


# =============================================================================
# public entry point — drop-in replacement of Ranger_bounds
# =============================================================================
def GABO_bounds(model, train_loader, device="cuda", bound_type='layer', bitflip='float',
                max_iter=100, patience=6, initial_lr_factor=0.2, scale_factor=2,
                max_samples=2048, protect_ratio=0.0, shrink_ratio=0.4,
                dynamic_range_factor=1.0, verbose=False, ablation="full",
                correct_threshold=0.1, bound_method="shrink_iqr_adaptive"):
    """
    Search ReLU activation bounds with GABO, 支持初始化统计量消融。

    与 gabo_multiclass.GABO_bounds 同契约, 额外参数:

    ``ablation``
        "full"          : 完整 GABO (熵 + 偏度 + 自适应IQR)
        "entropy_only"  : 只保留特征熵项 (β·H ≠ 0)
        "skew_only"     : 只保留偏度项 (|γ| ≠ 0)
        "adaptive_only" : 只保留自适应IQR项 (α·lnN ≠ 0)

    ``correct_threshold``
        θthr 修正阈值: 当 up_iqr < correct_threshold*max_act 时才重置为 0.9*max_act。
        默认放宽到 0.1(原版 0.5), 让更多层的统计量上界得以保留。

    Returns
    -------
    ``(results, tresh, None)`` — 与 Ranger_bounds 同布局。
    """
    if ablation not in ABLATION_MODES:
        raise ValueError(f"未知 ablation 模式: {ablation!r}, 可选 {sorted(ABLATION_MODES)}")
    abl = ABLATION_MODES[ablation]

    model.eval()
    model = model.to(device)

    per_neuron = (bound_type == "neuron")

    activations = {}
    handles = []
    _register_relu_hooks(model, activations, handles, name='')

    loader = train_loader['sub_train']
    stats = {}
    all_logits, all_labels = [], []
    n_seen = 0

    for data, label in loader:
        data = data.to(device)
        label = label.to(device)
        activations.clear()

        out = model(data)
        labels = label.view(-1, 1).long()
        correct = out.gather(1, labels)
        wrong_mask = torch.ones_like(out, dtype=torch.bool)
        wrong_mask.scatter_(1, labels, False)
        max_wrong = out.masked_fill(wrong_mask, -1e9).max(dim=1).values
        margin_sum = (correct.squeeze(1) - max_wrong).sum()

        for key, act in activations.items():
            grad_act = torch.autograd.grad(
                margin_sum, act,
                retain_graph=True, allow_unused=True,
            )[0]
            if grad_act is None:
                continue
            st = stats.setdefault(key, {"act": [], "grad": []})
            st["act"].append(act.detach())
            st["grad"].append(grad_act.detach())

        all_logits.append(out.detach())
        all_labels.append(label.detach().long().view(-1))

        n_seen += data.shape[0]
        if max_samples is not None and n_seen >= max_samples:
            break

    for h in handles:
        h.remove()

    if not stats:
        return {}, {}, None

    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0).long()
    correct = logits.gather(1, labels.unsqueeze(1)).squeeze(1)
    wrong_mask = torch.ones_like(logits, dtype=torch.bool)
    wrong_mask.scatter_(1, labels.unsqueeze(1), False)
    max_wrong = logits.masked_fill(wrong_mask, -1e9).max(dim=1).values
    sign_s = torch.ones_like(correct)
    margin0 = correct - max_wrong

    print(f"[GABO-ablation] mode={ablation}  "
          f"(entropy={abl['use_entropy']}, skew={abl['use_skewness']}, "
          f"adaptive={abl['use_adaptive']})")

    results, tresh = {}, {}

    for key, st in stats.items():
        act = torch.cat(st["act"], dim=0)
        grad_act = torch.cat(st["grad"], dim=0)

        batch_means = [a.mean(dim=0) for a in st["act"]]
        mins = batch_means[0]
        for m in batch_means[1:]:
            mins = torch.minimum(mins, m)
        tresh[key] = mins if per_neuron else torch.min(mins)

        act_np = act.detach().cpu().numpy().astype(np.float64)

        if per_neuron:
            D = act_np.shape[1]
            lowers, uppers = np.zeros(D), np.zeros(D)
            for j in range(D):
                col = act_np[:, j]
                lo, up = iqr_method(col, float(col.mean()), float(col.std()),
                                    _feature_entropy(col), _skewness(col),
                                    use_entropy=abl["use_entropy"],
                                    use_skewness=abl["use_skewness"],
                                    use_adaptive=abl["use_adaptive"],
                                    bound_method=bound_method)
                lowers[j], uppers[j] = lo, up
        else:
            flat = act_np.ravel()
            lo, up = iqr_method(flat, float(flat.mean()), float(flat.std()),
                                _feature_entropy(flat), _skewness(flat),
                                use_entropy=abl["use_entropy"],
                                use_skewness=abl["use_skewness"],
                                use_adaptive=abl["use_adaptive"],
                                bound_method=bound_method)
            lowers, uppers = np.array([lo]), np.array([up])

        max_act_np = float(act_np.max())
        std_np = float(act_np.std())

        uppers = np.array([_dynamically_adjust_upper_bound(u, max_act_np, threshold_ratio=correct_threshold)
                           for u in uppers])
        uppers = np.minimum(uppers, max_act_np)

        lower_limit = np.array([_dynamic_upper_bound_range(u, std_np, dynamic_range_factor)[0]
                                for u in uppers])
        upper_limit = np.array([_dynamic_upper_bound_range(u, std_np, dynamic_range_factor)[1]
                                for u in uppers])

        upper_limit = np.minimum(upper_limit, max_act_np)
        if protect_ratio > 0:
            lower_limit = np.maximum(lower_limit, max_act_np * protect_ratio)
        upper_limit = np.maximum(upper_limit, lower_limit)
        # (消融修复) 不再用 shrink_ratio 强制压上界: 它会把 uppers 的差异统一压到
        # shrink*max_act, 使熵/偏度/自适应IQR 的消融变体结果一致。让 upper_limit
        # 由统计量上界 (uppers + std) 决定, 统计量差异才能真正体现出来。
        # if shrink_ratio < 1.0:
        #     upper_limit = np.minimum(upper_limit, max_act_np * shrink_ratio)
        #     lower_limit = np.minimum(lower_limit, upper_limit)

        init_lambda = float(np.mean(uppers))

        print(f"[diag] {key}: up_iqr={float(np.mean(uppers)):.3f}  max_act={max_act_np:.3f}  "
              f"ratio={float(np.mean(uppers))/max_act_np:.3f}  "
              f"lower={float(np.mean(lower_limit)):.3f}  upper={float(np.mean(upper_limit)):.3f}  "
              f"init={init_lambda:.3f}")

        if verbose:
            print(f"[GABO] layer={key}  ranger_max={max_act_np:.4f}  "
                  f"init={init_lambda:.4f}  range=[{float(np.mean(lower_limit)):.4f},"
                  f" {float(np.mean(upper_limit)):.4f}]")

        bound = _gradient_ascent_bound(
            margin0=margin0,
            sign_s=sign_s,
            grad_act=grad_act,
            act=act,
            init_lambda=init_lambda,
            lower_limit=lower_limit if per_neuron else float(lower_limit[0]),
            upper_limit=upper_limit if per_neuron else float(upper_limit[0]),
            per_neuron=per_neuron,
            max_iter=max_iter,
            patience=patience,
            initial_lr_factor=initial_lr_factor,
            scale_factor=scale_factor,
            device=device,
            verbose=verbose,
        )

        results[key] = bound if per_neuron else bound.reshape(())
        print(f"[diag] {key}: final_bound={float(bound.mean()):.4f}  init={init_lambda:.4f}")

    return results, tresh, None
