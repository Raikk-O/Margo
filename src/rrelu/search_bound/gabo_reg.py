

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import copy


def _feature_entropy(x, n_bins=256):
    
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
    
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size == 0:
        return 0.0
    s = x.std()
    if s == 0:
        return 0.0
    return float(((x - x.mean()) ** 3).mean() / (s ** 3))


def iqr_method(features, custom_mean, custom_std, custom_entropy, custom_skewness,
               base_factor=1.5, base_lower_percentile=10, base_upper_percentile=90):
    
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
    
    if final_upper_bound < custom_max_value * threshold_ratio:
        return float(custom_max_value * correction_factor)
    return float(final_upper_bound)


class SearchableBoundedReLU(nn.Module):
    

    def __init__(self, initial_upper_bound):
        super().__init__()
        self.upper_bound = nn.Parameter(torch.tensor(initial_upper_bound, dtype=torch.float32))

    def forward(self, x):
        x = torch.relu(x)
        return torch.clamp(x, max=self.upper_bound)


def regression_loss_increment(pred, target, baseline_loss, fde_weight=3.0):
    
    loss_regular = torch.mean((pred[:, :, :3] - target[:, :, :3]) ** 2)

    diff = torch.abs(pred[:, :, 3] - target[:, :, 3])
    cog_loss = torch.mean(torch.minimum(diff, 1.0 - diff) ** 2)

    base_loss = loss_regular + cog_loss

    fde_mse = torch.mean((pred[:, -1, :3] - target[:, -1, :3]) ** 2)
    fde_penalty = fde_mse * fde_weight

    total_loss = base_loss + fde_penalty

    return total_loss - baseline_loss


def _gradient_ascent_bound(model, data_loader, device, max_samples, protect_ratio,
                           fde_weight, max_iter, verbose):
    
    model.eval()

    all_activations = {}
    hooks = []

    def hook_fn(name):
        def hook(module, input, output):
            if name not in all_activations:
                all_activations[name] = []
            all_activations[name].append(output.detach().cpu())

        return hook

    for name, module in model.named_modules():
        if isinstance(module, nn.ReLU):
            hooks.append(module.register_forward_hook(hook_fn(name)))

    baseline_loss_total = 0.0
    total_samples = 0

    X_all, Y_all = [], []

    with torch.no_grad():
        for batch_idx, (X, Y) in enumerate(data_loader):
            X, Y = X.to(device), Y.to(device)
            X_all.append(X.cpu())
            Y_all.append(Y.cpu())

            pred = model(X)

            loss_reg = torch.mean((pred[:, :, :3] - Y[:, :, :3]) ** 2)
            diff = torch.abs(pred[:, :, 3] - Y[:, :, 3])
            loss_cog = torch.mean(torch.minimum(diff, 1.0 - diff) ** 2)

            baseline_loss_total += (loss_reg + loss_cog).item() * X.size(0)
            total_samples += X.size(0)

            if total_samples >= max_samples:
                break

    baseline_loss_avg = baseline_loss_total / total_samples

    for h in hooks:
        h.remove()

    stats = {}
    for name, acts in all_activations.items():
        acts_tensor = torch.cat(acts, dim=0)
        flat_acts = acts_tensor.flatten()

        flat_acts_np = flat_acts.cpu().numpy().astype(np.float64)
        max_act = float(flat_acts_np.max())

        mean_min = torch.mean(acts_tensor, dim=0).min().item()

        entropy = _feature_entropy(flat_acts_np)
        skew = _skewness(flat_acts_np)
        _lo, up_iqr = iqr_method(flat_acts_np, float(flat_acts_np.mean()),
                                 float(flat_acts_np.std()), entropy, skew)

        up_iqr = _dynamically_adjust_upper_bound(up_iqr, max_act)
        iqr_upper = min(up_iqr, max_act)

        protect_lower = max_act * protect_ratio

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

    search_model = copy.deepcopy(model)
    lambda_params = {}

    for name, module in search_model.named_modules():
        if isinstance(module, nn.ReLU):
            parts = name.split('.')
            parent = search_model
            for p in parts[:-1]:
                parent = getattr(parent, p)

            init_val = stats[name]['init_lambda']
            new_module = SearchableBoundedReLU(init_val)
            setattr(parent, parts[-1], new_module)
            lambda_params[name] = new_module.upper_bound

    search_model = search_model.to(device)

    params_to_opt = [p for p in lambda_params.values()]
    optimizer = torch.optim.Adam(params_to_opt, lr=0.05)

    X_tensor = torch.cat(X_all, dim=0)[:max_samples].to(device)
    Y_tensor = torch.cat(Y_all, dim=0)[:max_samples].to(device)

    if verbose:
        print(f"[GABO] 开始梯度上升 (Max Iter: {max_iter})...")

    for it in range(max_iter):
        optimizer.zero_grad()

        pred = search_model(X_tensor)

        loss_inc = regression_loss_increment(pred, Y_tensor, baseline_loss_avg, fde_weight)


        reg_loss = 0.0
        for name, p in lambda_params.items():
            reg_loss += torch.mean((p - stats[name]['protect_lower']) / stats[name]['max_act'])

        total_obj = loss_inc + 0.1 * reg_loss
        total_obj.backward()
        optimizer.step()

        with torch.no_grad():
            for name, p in lambda_params.items():
                lower = stats[name]['protect_lower']
                upper = max(stats[name]['max_act'], stats[name]['iqr_upper'])
                p.clamp_(min=lower, max=upper)

        if verbose and (it + 1) % 5 == 0:
            print(f"  Iter {it + 1:2d}: Loss_Inc={loss_inc.item():.4f}, "
                  f"Reg={reg_loss.item():.4f}, "
                  f"Lambda={[f'{v.item():.3f}' for v in lambda_params.values()]}")

    results = {}
    tresh = {}  
    alpha = {}  

    for name, p in lambda_params.items():
        final_val = p.item()
        results[name] = torch.tensor(final_val)
        tresh[name] = torch.tensor(stats[name]['mean_min'])  
        alpha[name] = torch.tensor(0.0)  

    if verbose:
        print("[GABO] 优化完成。最终边界:")
        for name, val in results.items():
            print(f"  {name}: {val.item():.4f}")

    return results, tresh, alpha


def GABO_bounds(model, data_loader_dict, device, bound_type='layer', bitflip='fixed',
                max_samples=2048, protect_ratio=0.8, fde_weight=3.0,
                max_iter=100, verbose=True):
    
    if 'sub_train' not in data_loader_dict:
        raise ValueError("GABO_bounds requires 'sub_train' key in data_loader_dict")

    loader = data_loader_dict['sub_train']

    print(f"\n[GABO] 启动回归版边界优化 (Protect Ratio: {protect_ratio}, "
          f"FDE Weight: {fde_weight}, Max Samples: {max_samples})")

    return _gradient_ascent_bound(
        model, loader, device, max_samples, protect_ratio,
        fde_weight, max_iter, verbose
    )