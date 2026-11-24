

import torch
import torch.nn as nn
import numpy as np
import copy


ABLATION_MODES = {
    "full":       dict(use_entropy=True,  use_skewness=True,  use_adaptive=True),
    "no_entropy": dict(use_entropy=False, use_skewness=True,  use_adaptive=True),
    "no_skew":    dict(use_entropy=True,  use_skewness=False, use_adaptive=True),
    "fixed_iqr":  dict(use_entropy=False, use_skewness=False, use_adaptive=False),
}


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
               use_entropy=True, use_skewness=True, use_adaptive=True,
               bound_method="shrink_iqr_adaptive"):
    
    features = np.asarray(features, dtype=np.float64).ravel()
    n = len(features)
    if n == 0:
        return 0.0, 0.0

    if bound_method == "shrink_iqr_adaptive":
        max_act = float(np.max(features))
        pos = features[features > 0]
        if pos.size == 0:
            return float(max_act * 0.60), float(max_act * 0.95)
        q1 = np.percentile(pos, 25.0)
        q3 = np.percentile(pos, 75.0)
        iqr = q3 - q1
        iqr_ratio = float(np.clip(iqr / max_act, 0.0, 1.0))
        base = 0.95 - 0.35 * iqr_ratio
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


def _haversine_ade(pred, y_true_raw, anchor_lats, anchor_lons, feat_min, feat_max):
    
    if feat_min is not None and feat_max is not None:
        feat_min_t = torch.as_tensor(feat_min, device=pred.device, dtype=pred.dtype)
        feat_max_t = torch.as_tensor(feat_max, device=pred.device, dtype=pred.dtype)
        pred = pred * (feat_max_t - feat_min_t) + feat_min_t
    pred_dlat = pred[:, :, 0]
    pred_dlon = pred[:, :, 1]
    true_dlat = y_true_raw[:, :, 0]
    true_dlon = y_true_raw[:, :, 1]
    anchor_lats = anchor_lats.view(-1, 1)
    anchor_lons = anchor_lons.view(-1, 1)
    pred_lat = anchor_lats + pred_dlat
    pred_lon = anchor_lons + pred_dlon
    true_lat = anchor_lats + true_dlat
    true_lon = anchor_lons + true_dlon
    lat1, lon1, lat2, lon2 = map(torch.deg2rad, [pred_lat, pred_lon, true_lat, true_lon])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = torch.sin(dlat / 2.0) ** 2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2.0) ** 2
    a = torch.clamp(a, 0.0, 1.0 - 1e-8)   
    c = 2.0 * torch.asin(torch.sqrt(a))
    return torch.mean(3440.065 * c)


def _gradient_ascent_bound(model, data_loader, device, max_samples, protect_ratio,
                           fde_weight, max_iter, verbose, ablation="full",
                           correct_threshold=0.1, bound_method="shrink_iqr_adaptive",
                           feat_min=None, feat_max=None):
    
    if ablation not in ABLATION_MODES:
        raise ValueError(f"未知 ablation 模式: {ablation!r}, 可选 {sorted(ABLATION_MODES)}")
    abl = ABLATION_MODES[ablation]

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
    Y_raw_all, anchor_lat_all, anchor_lon_all = [], [], []
    use_ade = (feat_min is not None) and (feat_max is not None)

    with torch.no_grad():
        for batch in data_loader:
            if use_ade:
                X, Y, Y_raw, anchor_lat, anchor_lon = batch
            else:
                X, Y = batch
                Y_raw = anchor_lat = anchor_lon = None
            X, Y = X.to(device), Y.to(device)
            X_all.append(X.cpu())
            Y_all.append(Y.cpu())
            if use_ade:
                Y_raw_all.append(Y_raw.cpu())
                anchor_lat_all.append(anchor_lat.cpu())
                anchor_lon_all.append(anchor_lon.cpu())

            pred = model(X)

            if use_ade:
                baseline_loss_total += _haversine_ade(
                    pred, Y_raw.to(device), anchor_lat.to(device), anchor_lon.to(device),
                    feat_min, feat_max).item() * X.size(0)
            else:
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
        std_act = float(flat_acts_np.std())
        lo_iqr, up_iqr = iqr_method(flat_acts_np, float(flat_acts_np.mean()),
                                    std_act, entropy, skew,
                                    use_entropy=abl["use_entropy"],
                                    use_skewness=abl["use_skewness"],
                                    use_adaptive=abl["use_adaptive"],
                                    bound_method=bound_method)

        up_iqr = _dynamically_adjust_upper_bound(up_iqr, max_act, threshold_ratio=correct_threshold)
        iqr_upper = min(up_iqr, max_act)
        protect_lower = max_act * protect_ratio

        init_lambda = iqr_upper
        lower = max(iqr_upper - std_act, protect_lower)
        upper = min(iqr_upper + std_act, max_act)
        if upper < lower:
            upper = lower

        print(f"[diag] {name}: up_iqr={up_iqr:.3f}  max_act={max_act:.3f}  "
              f"ratio={up_iqr/max_act:.3f}  std={std_act:.3f}  protect_lower={protect_lower:.3f}  "
              f"init={init_lambda:.3f}  range=[{lower:.3f}, {upper:.3f}]")

        stats[name] = {
            'max_act': max_act,
            'iqr_upper': iqr_upper,
            'protect_lower': protect_lower,
            'init_lambda': init_lambda,
            'lower': lower,
            'upper': upper,
            'mean_min': mean_min
        }

    if verbose:
        print(f"[GABO-ablation] mode={ablation}  "
              f"(entropy={abl['use_entropy']}, skew={abl['use_skewness']}, "
              f"adaptive={abl['use_adaptive']})  样本={total_samples}  "
              f"Baseline Loss={baseline_loss_avg:.4f}")
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
    if use_ade:
        Y_raw_tensor = torch.cat(Y_raw_all, dim=0)[:max_samples].to(device)
        anchor_lat_tensor = torch.cat(anchor_lat_all, dim=0)[:max_samples].to(device)
        anchor_lon_tensor = torch.cat(anchor_lon_all, dim=0)[:max_samples].to(device)

    if verbose:
        print(f"[GABO] 开始梯度上升 (Max Iter: {max_iter})...")

    for it in range(max_iter):
        optimizer.zero_grad()

        pred = search_model(X_tensor)

        if use_ade:
            ade_now = _haversine_ade(pred, Y_raw_tensor, anchor_lat_tensor, anchor_lon_tensor,
                                     feat_min, feat_max)
            loss_inc = ade_now - baseline_loss_avg
        else:
            loss_inc = regression_loss_increment(pred, Y_tensor, baseline_loss_avg, fde_weight)

        reg_loss = 0.0
        for name, p in lambda_params.items():
            reg_loss += torch.mean((p - stats[name]['protect_lower']) / stats[name]['max_act'])

        total_obj = loss_inc + 0.1 * reg_loss
        total_obj.backward()
        optimizer.step()

        with torch.no_grad():
            for name, p in lambda_params.items():
                lower = stats[name]['lower']
                upper = stats[name]['upper']
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
        print(f"[diag] {name}: final_bound={final_val:.3f}  init={stats[name]['init_lambda']:.3f}")

    if verbose:
        print("[GABO] 优化完成。最终边界:")
        for name, val in results.items():
            print(f"  {name}: {val.item():.4f}")

    return results, tresh, alpha


def GABO_bounds(model, data_loader_dict, device, bound_type='layer', bitflip='fixed',
                max_samples=2048, protect_ratio=0.8, fde_weight=3.0,
                max_iter=100, verbose=True, ablation="full",
                correct_threshold=0.1, bound_method="shrink_iqr_adaptive",
                feat_min=None, feat_max=None):
    
    if 'sub_train' not in data_loader_dict:
        raise ValueError("GABO_bounds requires 'sub_train' key in data_loader_dict")

    use_ade = (feat_min is not None) and (feat_max is not None) and ('sub_train_full' in data_loader_dict)
    loader = data_loader_dict['sub_train_full'] if use_ade else data_loader_dict['sub_train']

    print(f"\n[GABO-ablation] 启动回归版边界优化 (mode={ablation}, bound_method={bound_method}, "
          f"Protect Ratio: {protect_ratio}, FDE Weight: {fde_weight}, "
          f"Max Samples: {max_samples}, correct_threshold={correct_threshold})")

    return _gradient_ascent_bound(
        model, loader, device, max_samples, protect_ratio,
        fde_weight, max_iter, verbose, ablation, correct_threshold,
        bound_method,
        feat_min if use_ade else None, feat_max if use_ade else None
    )
