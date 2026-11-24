# -*- coding: utf-8 -*-
"""
GABO_bounds: 面向回归任务的梯度激活边界优化 (Gradient-based Activation Bound Optimization)
======================================================================================
专为 MaritimeTrajModel (轨迹预测) 改造。
核心改造点：
  1. 目标函数：从分类 Margin 改为 损失增量最小化 (Loss Increment Minimization)
  2. FDE 感知：对轨迹终点 (最后时间步) 赋予更高权重
  3. 精度保护：引入 protect_ratio 防止边界过度收紧导致 Clean ADE/FDE 暴跌
  4. 可微搜索：利用 PyTorch Autograd 直接对边界求导，避免手动计算高维 Jacobian

接口契约：与 Ranger_bounds 保持一致，返回 (results_DMA, tresh, alpha)
"""

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import copy


# ==========================================
# 0. 分布感知统计量 (特征熵 + 偏度 + 自适应 IQR)
# ==========================================
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
               base_factor=1.5, base_lower_percentile=10, base_upper_percentile=90):
    """IQR based bounds with dynamic percentiles/factor."""
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


def _dynamically_adjust_upper_bound(final_upper_bound, custom_max_value,
                                    threshold_ratio=0.5, correction_factor=0.9):
    """Guards against an IQR bound that sits far below the real maximum."""
    if final_upper_bound < custom_max_value * threshold_ratio:
        return float(custom_max_value * correction_factor)
    return float(final_upper_bound)


# ==========================================
# 1. 可微搜索模块 (Differentiable Search Module)
# ==========================================
class SearchableBoundedReLU(nn.Module):
    """
    在搜索阶段临时替换 nn.ReLU 的模块。
    支持 clamp 截断，且上界 lambda 是 requires_grad=True 的张量，
    使得损失函数可以直接对 lambda 求导。
    """

    def __init__(self, initial_upper_bound):
        super().__init__()
        # lambda 作为可学习参数
        self.upper_bound = nn.Parameter(torch.tensor(initial_upper_bound, dtype=torch.float32))

    def forward(self, x):
        # 先做标准 ReLU，再做上界截断
        x = torch.relu(x)
        return torch.clamp(x, max=self.upper_bound)


# ==========================================
# 2. 回归版损失函数 (与训练目标对齐)
# ==========================================
def regression_loss_increment(pred, target, baseline_loss, fde_weight=3.0):
    """
    计算带 FDE 权重的损失增量。
    pred, target: (B, PRED_LEN, FEAT_DIM) 归一化空间的预测与真实值
    baseline_loss: 标量，无截断时的原始损失
    """
    # 1. 常规 MSE (位置 + 速度)
    loss_regular = torch.mean((pred[:, :, :3] - target[:, :, :3]) ** 2)

    # 2. 周期性 MSE (航向 COG, min-max 归一化到 [0,1], 周期为 1.0)
    diff = torch.abs(pred[:, :, 3] - target[:, :, 3])
    cog_loss = torch.mean(torch.minimum(diff, 1.0 - diff) ** 2)

    # 3. 基础 Loss
    base_loss = loss_regular + cog_loss

    # 4. FDE 终点惩罚 (计算终点误差并加权)
    fde_mse = torch.mean((pred[:, -1, :3] - target[:, -1, :3]) ** 2)
    fde_penalty = fde_mse * fde_weight

    # 总损失 = 基础损失 + 终点惩罚
    total_loss = base_loss + fde_penalty

    # 损失增量 = 当前损失 - 基准损失 (我们希望增量越小越好)
    return total_loss - baseline_loss


# ==========================================
# 3. 核心优化器：梯度上升搜索边界
# ==========================================
def _gradient_ascent_bound(model, data_loader, device, max_samples, protect_ratio,
                           fde_weight, max_iter, verbose):
    """
    通过梯度上升寻找最优上界 lambda。
    目标：Maximize -Loss_Increment (即 Minimize Loss Increment)
    """
    model.eval()

    # 1. 收集基准状态 (Baseline) 和 统计量
    all_activations = {}
    hooks = []

    def hook_fn(name):
        def hook(module, input, output):
            # 记录该层所有样本的激活值
            if name not in all_activations:
                all_activations[name] = []
            all_activations[name].append(output.detach().cpu())

        return hook

    # 注册 Hook 并收集数据
    for name, module in model.named_modules():
        if isinstance(module, nn.ReLU):
            hooks.append(module.register_forward_hook(hook_fn(name)))

    # 跑一遍数据收集激活值，并计算 Baseline Loss
    baseline_loss_total = 0.0
    total_samples = 0

    # 准备目标数据 (用于后续计算 Loss Increment)
    X_all, Y_all = [], []

    with torch.no_grad():
        for batch_idx, (X, Y) in enumerate(data_loader):
            X, Y = X.to(device), Y.to(device)
            X_all.append(X.cpu())
            Y_all.append(Y.cpu())

            pred = model(X)

            # 计算 Baseline Loss (不带 FDE 权重的纯基础 Loss，作为锚点)
            loss_reg = torch.mean((pred[:, :, :3] - Y[:, :, :3]) ** 2)
            diff = torch.abs(pred[:, :, 3] - Y[:, :, 3])
            loss_cog = torch.mean(torch.minimum(diff, 1.0 - diff) ** 2)

            baseline_loss_total += (loss_reg + loss_cog).item() * X.size(0)
            total_samples += X.size(0)

            if total_samples >= max_samples:
                break

    baseline_loss_avg = baseline_loss_total / total_samples

    # 移除 Hooks
    for h in hooks:
        h.remove()

    # 拼接激活值并计算统计量
    stats = {}
    for name, acts in all_activations.items():
        acts_tensor = torch.cat(acts, dim=0)
        # 展平计算统计量
        flat_acts = acts_tensor.flatten()

        # 展平为 numpy 计算分布感知统计量
        flat_acts_np = flat_acts.cpu().numpy().astype(np.float64)
        max_act = float(flat_acts_np.max())

        # 复刻 Ranger 的 tresh 语义: 逐位置 batch 均值的全局最小 (越界替换值)
        mean_min = torch.mean(acts_tensor, dim=0).min().item()

        # 自适应 IQR 上界: 特征熵 + 偏度 + 自适应百分位 (替代固定 25%/75% 四分位)
        entropy = _feature_entropy(flat_acts_np)
        skew = _skewness(flat_acts_np)
        _lo, up_iqr = iqr_method(flat_acts_np, float(flat_acts_np.mean()),
                                 float(flat_acts_np.std()), entropy, skew)

        # 物理约束修正: 上界过低时回退到 0.9*max, 且不超过 max
        up_iqr = _dynamically_adjust_upper_bound(up_iqr, max_act)
        iqr_upper = min(up_iqr, max_act)

        # 精度保护下限
        protect_lower = max_act * protect_ratio

        # 初始搜索点：从 max_act 开始(无截断)。若从 protect_lower 起步, reg 项在
        # λ=protect_lower 处梯度为 0(relu 在 0 处左导数为 0), 且激活稀疏右偏时
        # loss_inc 对 λ 的梯度也趋近 0, 导致 λ 完全卡死不动。
        init_lambda = max_act

        stats[name] = {
            'max_act': max_act,
            'iqr_upper': iqr_upper,
            'protect_lower': protect_lower,
            'init_lambda': init_lambda,
            'mean_min': mean_min
        }

    if verbose:
        print(f"[GABO] 收集完成: {total_samples} 样本, Baseline Loss: {baseline_loss_avg:.4f}")
        for name, s in stats.items():
            print(f"  层 {name}: Max={s['max_act']:.3f}, IQR_Upper={s['iqr_upper']:.3f}, "
                  f"Protect_Lower={s['protect_lower']:.3f}, Init_Lambda={s['init_lambda']:.3f}")

    # 2. 替换模型中的 ReLU 为 SearchableBoundedReLU
    search_model = copy.deepcopy(model)
    lambda_params = {}

    for name, module in search_model.named_modules():
        if isinstance(module, nn.ReLU):
            # 找到对应的父模块和属性名进行替换
            parts = name.split('.')
            parent = search_model
            for p in parts[:-1]:
                parent = getattr(parent, p)

            init_val = stats[name]['init_lambda']
            new_module = SearchableBoundedReLU(init_val)
            setattr(parent, parts[-1], new_module)
            lambda_params[name] = new_module.upper_bound

    # 关键: 新创建的 SearchableBoundedReLU.upper_bound 默认在 CPU, 而 search_model
    # 其余参数在 device(GPU) 上, 直接 forward 会报 device mismatch。
    # .to(device) 会原地移动各参数的 .data, Parameter 对象引用不变, lambda_params 仍有效。
    search_model = search_model.to(device)

    # 3. 设置优化器 (只优化 lambda 参数)
    params_to_opt = [p for p in lambda_params.values()]
    optimizer = torch.optim.Adam(params_to_opt, lr=0.05)

    # 将数据加载到内存以加速迭代
    X_tensor = torch.cat(X_all, dim=0)[:max_samples].to(device)
    Y_tensor = torch.cat(Y_all, dim=0)[:max_samples].to(device)

    # 4. 梯度上升迭代
    if verbose:
        print(f"[GABO] 开始梯度上升 (Max Iter: {max_iter})...")

    for it in range(max_iter):
        optimizer.zero_grad()

        # 前向传播 (经过带参数的 BoundedReLU)
        pred = search_model(X_tensor)

        # 计算损失增量 (带 FDE 权重)
        loss_inc = regression_loss_increment(pred, Y_tensor, baseline_loss_avg, fde_weight)

        # 目标：Minimize Loss Increment -> 即 Maximize -Loss Increment
        # PyTorch 默认是 Minimize，所以我们直接 backward loss_inc
        # 但为了让 lambda 变大（放宽边界以减少截断），我们需要反向操作：
        # 如果 loss_inc > 0 (性能变差)，梯度会促使 lambda 增大以放宽截断。
        # 如果 loss_inc < 0 (不可能，但理论上)，梯度会促使 lambda 减小以收紧边界。
        # 实际上，我们希望找到“最小的 lambda 使得 loss_inc 接近 0”。
        # 这里我们加一个 L1 正则化，鼓励 lambda 尽可能小（收紧边界）：

        reg_loss = 0.0
        for name, p in lambda_params.items():
            # 鼓励 lambda 收紧(靠近 protect_lower): 用归一化线性惩罚而非 relu。
            # relu 在 λ=protect_lower 处梯度为 0; 除以 max_act 归一化让不同量级
            # (max_act 3.4 vs 73.6) 的层有统一且恒定的收紧驱动力梯度。
            reg_loss += torch.mean((p - stats[name]['protect_lower']) / stats[name]['max_act'])

        total_obj = loss_inc + 0.1 * reg_loss
        total_obj.backward()
        optimizer.step()

        # 5. 投影到安全范围 (Clamp)
        with torch.no_grad():
            for name, p in lambda_params.items():
                lower = stats[name]['protect_lower']
                upper = max(stats[name]['max_act'], stats[name]['iqr_upper'])
                p.clamp_(min=lower, max=upper)

        if verbose and (it + 1) % 5 == 0:
            print(f"  Iter {it + 1:2d}: Loss_Inc={loss_inc.item():.4f}, "
                  f"Reg={reg_loss.item():.4f}, "
                  f"Lambda={[f'{v.item():.3f}' for v in lambda_params.values()]}")

    # 6. 提取最终结果
    results = {}
    tresh = {}  # 兼容 rrelu 接口
    alpha = {}  # 兼容 rrelu 接口

    for name, p in lambda_params.items():
        final_val = p.item()
        results[name] = torch.tensor(final_val)
        tresh[name] = torch.tensor(stats[name]['mean_min'])  # 独立下界 (激活均值最小值)
        alpha[name] = torch.tensor(0.0)  # 兼容 rrelu 接口

    if verbose:
        print("[GABO] 优化完成。最终边界:")
        for name, val in results.items():
            print(f"  {name}: {val.item():.4f}")

    return results, tresh, alpha


# ==========================================
# 4. 对外统一接口 (兼容 Ranger_bounds)
# ==========================================
def GABO_bounds(model, data_loader_dict, device, bound_type='layer', bitflip='fixed',
                max_samples=2048, protect_ratio=0.8, fde_weight=3.0,
                max_iter=100, verbose=True):
    """
    GABO 边界搜索主入口。

    参数:
        model: 轨迹预测模型
        data_loader_dict: 必须包含 'sub_train' 键的 DataLoader 字典
        device: 'cuda' 或 'cpu'
        max_samples: 用于统计和优化的最大样本数
        protect_ratio: 精度保护下限比例 (0.85~0.95)
        fde_weight: FDE 终点惩罚权重 (建议 3.0~5.0)
        max_iter: 梯度上升最大迭代次数
        verbose: 是否打印过程

    返回:
        (results_DMA, tresh, alpha) 元组，与 Ranger_bounds 接口一致
    """
    if 'sub_train' not in data_loader_dict:
        raise ValueError("GABO_bounds requires 'sub_train' key in data_loader_dict")

    loader = data_loader_dict['sub_train']

    print(f"\n[GABO] 启动回归版边界优化 (Protect Ratio: {protect_ratio}, "
          f"FDE Weight: {fde_weight}, Max Samples: {max_samples})")

    return _gradient_ascent_bound(
        model, loader, device, max_samples, protect_ratio,
        fde_weight, max_iter, verbose
    )