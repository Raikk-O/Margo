# -*- coding: utf-8 -*-
"""
GABO_bounds — PyTorch port of the boundary optimisation in ``src/ARGS/GABO_imp_test5_2.py``
============================================================================================

Original algorithm (TensorFlow 1.x / Keras, CIFAR-10 + VGG16)
-------------------------------------------------------------
``GABO_imp_test5_2.py`` searches, for every ReLU layer *k*, an activation upper
bound ``lambda_k`` by **gradient ascent**::

    deviation   = lambda_k - linear_output_k                 # 边界与实际激活的偏差
    fc_affected = fc + <d fc / d out_k , deviation>          # 一阶外推：正确类分数
    fj_affected = max_j ( fj + <d fj / d out_k , deviation> )# 一阶外推：最高错误类分数
    result_i    = clip(fc_affected_i - fj_affected_max_i, -1, 1)
    objective   = prod_i result_i       # 对 lambda_k 做梯度上升，最大化该乘积

    # 早停 + 自适应学习率 (adjust_learning_rate)
    # 最终结果裁剪到 [lower_limit_std, upper_limit_std]

``[lower_limit_std, upper_limit_std]`` in the original comes from ``Args_ana.py``,
which transfers IQR statistics from a **pre-trained VGG16** onto the target model.

Why this port exists
--------------------
1. ``src/ARGS/*`` is TF-1.x (``tf.Session`` / ``tf.placeholder`` / ``tf.gradients``)
   and is hard-wired to Keras models + CIFAR-10 + VGG16 layer matching.
   It **cannot** be executed on a PyTorch MLP over tabular data.
2. The VGG16 statistic transfer has no counterpart for AIS tabular features,
   so here the initial bound / admissible range are computed from the **target
   model's own activations**, using the very same IQR rule (``iqr_method`` is
   ported 1:1 from ``Args_ana.py``).

Public contract — identical to ``Ranger_bounds``
------------------------------------------------
::

    results_DMA, tresh, alpha = GABO_bounds(model, data_loader_dict,
                                        device, bound_type, bitflip)

so it can be swapped in without touching the rest of the rrelu pipeline.

Difference w.r.t. Ranger
------------------------
* ``Ranger``  : bound = max activation seen on the training set (post-ReLU).
* ``GABO``    : bound = a *searched* value that maximises the worst-case
  classification margin under the bound perturbation, clipped into an IQR
  derived admissible range.  Tighter bound -> better bit-flip resilience.

Notes
-----
* Gradients are taken w.r.t. the ReLU **output**; this equals the gradient w.r.t.
  the pre-activation whenever the unit is active (a > 0), which is exactly the
  region an upper bound operates on.
* ``tresh`` (the replacement value used by ``bounded_relu_tresh``) is not
  optimised by GABO, so Ranger's definition is reused.
"""

import math

import numpy as np
import torch
import torch.nn as nn


# =============================================================================
# statistics helpers (ported from src/ARGS/Args_ana.py, pure numpy)
# =============================================================================
def _feature_entropy(x, n_bins=256):
    """Shannon entropy (bits) of the histogram of ``x`` — mirrors Args_ana.calculate_feature_entropy."""
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
               base_factor=1.5, base_lower_percentile=10, base_upper_percentile=90):
    """
    IQR based bounds with dynamic percentiles/factor.

    Ported **1:1** from ``src/ARGS/Args_ana.py::iqr_method``.
    """
    features = np.asarray(features, dtype=np.float64).ravel()
    n = len(features)
    if n == 0:
        return 0.0, 0.0

    lower_percentile = max(1.0, base_lower_percentile - np.log(n) * 0.5 - abs(custom_skewness))
    upper_percentile = min(99.0, base_upper_percentile + np.log(n) * 0.5
                           + abs(custom_skewness) + custom_entropy * 0.05)

    std_mean_ratio = custom_std / custom_mean if custom_mean != 0 else base_factor
    std_mean_ratio = float(np.clip(std_mean_ratio, 0, 0.1))
    factor = base_factor + std_mean_ratio

    q1 = np.percentile(features, lower_percentile)
    q3 = np.percentile(features, upper_percentile)
    iqr = q3 - q1

    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr
    return float(lower_bound), float(upper_bound)


def _dynamic_upper_bound_range(final_upper_bound, custom_std, factor=1.0):
    """
    ``[lower_limit_std, upper_limit_std]`` around ``final_upper_bound``.

    Ported 1:1 from ``Args_ana.calculate_dynamic_upper_bound_range``.
    """
    adjustment = custom_std * factor
    return float(final_upper_bound - adjustment), float(final_upper_bound + adjustment)


def _dynamically_adjust_upper_bound(final_upper_bound, custom_max_value,
                                    threshold_ratio=0.5, correction_factor=0.9):
    """
    Guards against an IQR bound that sits far below the real maximum.

    Ported 1:1 from ``Args_ana.dynamically_adjust_upper_bound``.
    """
    if final_upper_bound < custom_max_value * threshold_ratio:
        return float(custom_max_value * correction_factor)
    return float(final_upper_bound)


# =============================================================================
# hook helpers — key naming is kept identical to ranger.py so that
# replace_act_all() can look the bounds up by the same key.
# =============================================================================
def _register_relu_hooks(model, activations, handles, name=''):
    for name1, layer in model.named_children():
        if len(list(layer.children())) == 0:
            if isinstance(layer, nn.ReLU):
                name_ = name1 + name

                def _make_hook(key):
                    def _hook(module, inp, out):
                        out.retain_grad()          # needed to get d(logit)/d(activation)
                        activations[key] = out
                    return _hook

                handles.append(layer.register_forward_hook(_make_hook(name_)))
        else:
            _register_relu_hooks(layer, activations, handles, name + name1)


# =============================================================================
# adaptive learning rate (ported from GABO_imp_test5_2.adjust_learning_rate,
# vectorised so that it also works for the per-neuron case)
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

    lr = torch.where(sign_change, lr / scale_factor, lr)          # 方向翻转 -> 缩小步长
    inc = ~sign_change
    patience_counter = patience_counter + inc.to(patience_counter.dtype)
    bump = inc & (patience_counter >= increase_patience)
    lr = torch.where(bump, lr * scale_factor, lr)                 # 连续同向 -> 放大步长
    patience_counter = torch.where(bump, torch.zeros_like(patience_counter), patience_counter)

    lr = torch.minimum(lr, max_lr)
    return torch.round(lr * 1e5) / 1e5, patience_counter


# =============================================================================
# core: gradient ascent on the bound
# =============================================================================
def _gradient_ascent_bound(margin0, sign_s, grad_act, act, init_lambda,
                           lower_limit, upper_limit, per_neuron,
                           max_iter=50, patience=6,
                           initial_lr_factor=0.2, scale_factor=2,
                           device="cpu", verbose=False):
    """
    Search the activation upper bound ``lambda`` that maximises
    ``prod_i clip(margin_i', -1, 1)`` with the first-order extrapolation
    ``margin_i' = margin_i + s_i * <d z_i / d a_i , lambda - a_i>``.

    ``margin0``  : [N]      signed margin  (2y-1) * logit
    ``sign_s``   : [N]      (2y-1)
    ``grad_act`` : [N, D]   per-sample gradient d z_i / d a_{i,j}
    ``act``      : [N, D]   activations
    """
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
        deviation = torch.clamp(lam - act, max=0.0)            # [N, D] = min(λ-a, 0): 低于阈值偏差为 0
        perturb = (grad_act * deviation).sum(dim=1)            # [N] 一阶外推的 logit 变化
        margin = margin0 + sign_s * perturb                    # [N] 扰动后的 signed margin
        result = margin.clamp(-1.0, 1.0)

        # prod(result) 在数值上不可靠 -> 用 log 形式, 符号单独处理
        # (对应原版 flag_cal_logsum 分支: log_sum_val * sum_sign)
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

        # ---- early stopping (same rule as the original) ----
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
            lam_new = torch.round(lam_new * 1e4) / 1e4        # 原版 round(lambda, 4)
            lam.data.copy_(lam_new)

    if verbose:
        print(f"    -> searched lambda mean={float(best_lam.mean()):.4f} "
              f"clip range=[{float(lower_limit.mean()):.4f}, {float(upper_limit.mean()):.4f}]")

    return best_lam.detach().clamp(lower_limit, upper_limit)


# =============================================================================
# public entry point — drop-in replacement of Ranger_bounds
# =============================================================================
def GABO_bounds(model, train_loader, device="cuda", bound_type='layer', bitflip='float',
                max_iter=100, patience=6, initial_lr_factor=0.2, scale_factor=2,
                max_samples=2048, protect_ratio=0.0, shrink_ratio=0.4,
                dynamic_range_factor=1.0, verbose=False):
    """
    Search ReLU activation bounds with GABO (gradient ascent) instead of Ranger's max.

    Parameters mirror ``Ranger_bounds``; the extra ones tune the ported optimiser:

    ``max_iter``, ``patience``, ``initial_lr_factor``, ``scale_factor``
        gradient-ascent hyper-parameters (same meaning as in ``GABO_imp_test5_2.py``).
    ``max_samples``
        how many training samples to collect activation statistics from.
        The original GABO used 1000 representative images; keeping this bounded
        controls both runtime and memory.
    ``protect_ratio``
        NEW in this port (the original relied on VGG16 statistics for this).
        Lower clamp of the searched bound, as a fraction of Ranger's max
        activation: ``lower_limit = max(lower_limit, max_act * protect_ratio)``.
        It prevents GABO from clipping so aggressively that clean accuracy drops.
        Set to ``0.0`` for the unconstrained original behaviour.

        The searched bound is *also* clamped from above by ``max_act``, so the
        result is guaranteed to satisfy ``gabo_bound <= ranger_bound``: a bound
        above the largest activation ever observed would never trigger and would
        silently disable the protection.
    ``dynamic_range_factor``
        factor used by ``_dynamic_upper_bound_range`` (original default 1.0).

    Returns
    -------
    ``(results_DMA, tresh, alpha)`` — same layout as ``Ranger_bounds``.
    """
    model.eval()
    model = model.to(device)

    per_neuron = (bound_type == "neuron")

    # ---- 1. hooks -------------------------------------------------------
    activations = {}
    handles = []
    _register_relu_hooks(model, activations, handles, name='')

    # ---- 2. collect activations + per-sample gradients -------------------
    loader = train_loader['sub_train']
    stats = {}
    all_logits, all_labels = [], []
    n_seen = 0

    for data, label in loader:
        data = data.to(device)
        label = label.to(device)
        activations.clear()

        out = model(data)                                          # [B, C] 多分类 logits
        # 多分类 margin = 正确类 logit - 最高错误类 logit(原版 GABO 的 fc - fj)。
        # 此前移植只用了正确类 logit, 丢了错误类信息, 目标退化为"最大化正确类 logit"
        # 而非"最大化分类间隔", 与容错(希望正确类相对错误类 margin 大)脱节。
        labels = label.view(-1, 1).long()
        correct = out.gather(1, labels)                             # [B, 1]
        wrong_mask = torch.ones_like(out, dtype=torch.bool)
        wrong_mask.scatter_(1, labels, False)
        max_wrong = out.masked_fill(wrong_mask, -1e9).max(dim=1).values  # [B]
        margin_sum = (correct.squeeze(1) - max_wrong).sum()         # 分类 margin 之和

        for key, act in activations.items():
            # 分类 margin 对激活的梯度
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
    correct = logits.gather(1, labels.unsqueeze(1)).squeeze(1)     # [N] 正确类 logit
    wrong_mask = torch.ones_like(logits, dtype=torch.bool)
    wrong_mask.scatter_(1, labels.unsqueeze(1), False)
    max_wrong = logits.masked_fill(wrong_mask, -1e9).max(dim=1).values  # [N] 最高错误类 logit
    sign_s = torch.ones_like(correct)                              # [N]
    margin0 = correct - max_wrong                                  # [N] 分类 margin(正确-错误)

    results, tresh = {}, {}

    # ---- 3. per layer: IQR initialisation + gradient ascent --------------
    for key, st in stats.items():
        act = torch.cat(st["act"], dim=0)                          # [N, D]
        grad_act = torch.cat(st["grad"], dim=0)                    # [N, D]

        # ---- tresh: reuse Ranger's definition (GABO does not optimise it) ----
        batch_means = [a.mean(dim=0) for a in st["act"]]
        mins = batch_means[0]
        for m in batch_means[1:]:
            mins = torch.minimum(mins, m)
        tresh[key] = mins if per_neuron else torch.min(mins)

        # ---- IQR based initial bound / admissible range ----------------------
        act_np = act.detach().cpu().numpy().astype(np.float64)

        if per_neuron:
            D = act_np.shape[1]
            lowers, uppers = np.zeros(D), np.zeros(D)
            for j in range(D):
                col = act_np[:, j]
                lo, up = iqr_method(col, float(col.mean()), float(col.std()),
                                    _feature_entropy(col), _skewness(col))
                lowers[j], uppers[j] = lo, up
        else:
            flat = act_np.ravel()
            lo, up = iqr_method(flat, float(flat.mean()), float(flat.std()),
                                _feature_entropy(flat), _skewness(flat))
            lowers, uppers = np.array([lo]), np.array([up])

        max_act_np = float(act_np.max())
        std_np = float(act_np.std())

        # 原版: final_upper_bound = dynamically_adjust_upper_bound(iqr_upper, max_value)
        uppers = np.array([_dynamically_adjust_upper_bound(u, max_act_np) for u in uppers])

        # 移植新增 (关键): 候选上界不允许超过 Ranger 的 max。
        # 训练时从未出现超过 max 的激活, 边界大于 max 时 BReLU 恒等, 约束完全不生效,
        # 容错效果会退化到"无保护"。IQR 的 q3 + k*IQR 在右偏分布上常常大于 max,
        # 原版靠 VGG16 统计迁移抑制了这一点, 这里直接 clamp。
        uppers = np.minimum(uppers, max_act_np)

        # 原版: [lower_limit_std, upper_limit_std] = final_upper_bound ± std * factor
        lower_limit = np.array([_dynamic_upper_bound_range(u, std_np, dynamic_range_factor)[0]
                                for u in uppers])
        upper_limit = np.array([_dynamic_upper_bound_range(u, std_np, dynamic_range_factor)[1]
                                for u in uppers])

        # 同上: 搜索范围同样压到 max 以内, 保证 GABO 边界一定不比 Ranger 松
        upper_limit = np.minimum(upper_limit, max_act_np)
        # 精度保护下限（移植新增，原版靠 VGG16 统计约束）
        if protect_ratio > 0:
            lower_limit = np.maximum(lower_limit, max_act_np * protect_ratio)
        upper_limit = np.maximum(upper_limit, lower_limit)
        # 收缩上界（移植新增）: 强制 λ <= max_act * shrink_ratio, 并同步压 lower_limit。
        # 必须放在 max(upper, lower) 之后, 否则会被 lower_limit 顶回(前面的 lower≈max_act)。
        # 实验(收缩因子扫描)证明该任务 bound 越小容错越好, clean 不变。
        if shrink_ratio < 1.0:
            upper_limit = np.minimum(upper_limit, max_act_np * shrink_ratio)
            lower_limit = np.minimum(lower_limit, upper_limit)

        init_lambda = float(np.mean(uppers))

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

        # results_DMA 的形状约定必须与 Ranger_bounds 完全一致
        results[key] = bound if per_neuron else bound.reshape(())

    return results, tresh, None



