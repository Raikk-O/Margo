
import os, sys, argparse, warnings, copy, importlib.util
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from tqdm import tqdm
from fxpmath import Fxp
from sklearn.metrics import f1_score

warnings.filterwarnings("ignore")

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, '..', '..'))
for _p in (_PROJECT_ROOT, os.path.join(_PROJECT_ROOT, 'src')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.rrelu.pytorchfi.core import FaultInjection
from src.rrelu.pytorchfi.weight_error_models import multi_weight_inj_fixed, multi_weight_inj_float
from src.rrelu.relu_bound.bound_relu import Relu_bound
from src.rrelu.relu_bound.bound_tresh import bounded_relu_tresh
from src.rrelu.relu_bound.bound_zero import bounded_relu_zero

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

from data_prepare import prepare_multiclass_data, CLASS_NAMES


def _load_rrelu_module(mod_name, rel_path):
    file_path = os.path.join(_PROJECT_ROOT, 'src', 'rrelu', rel_path)
    spec = importlib.util.spec_from_file_location(mod_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


Ranger_bounds = _load_rrelu_module('rrelu_ranger', 'search_bound/ranger.py').Ranger_bounds
GABO_bounds = _load_rrelu_module('rrelu_gabo_multiclass', 'search_bound/gabo_multiclass.py').GABO_bounds
GABO_SAC_bounds = _load_rrelu_module('rrelu_gabo_sac', 'search_bound/RL-SAC/gabo_multiClass.py').GABO_SAC_bounds
_opt_mod = _load_rrelu_module('rrelu_global_opt', 'search_bound/RL-SAC/global_optimizers.py')
PSO_bounds = _opt_mod.PSO_bounds
GA_bounds = _opt_mod.GA_bounds
BayesOpt_bounds = _opt_mod.BayesOpt_bounds

RELU_BOUND_FN = {
    "tresh": bounded_relu_tresh,
    "zero": bounded_relu_zero,
}


def _bitflip_inplace_fixed(t, fault_rate, total_bits, n_frac, n_int):
    
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


def _bitflip_inplace_float(t, fault_rate, total_bits=32):
    
    n_fault = int(t.numel() * total_bits * fault_rate)
    if n_fault <= 0 or t.dtype != torch.float32:
        return
    dev = t.device
    int_view = t.detach().contiguous().view(torch.int32).view(-1)
    idx = torch.randint(0, int_view.numel(), (n_fault,), device=dev)
    b = torch.randint(0, total_bits, (n_fault,), device=dev).to(torch.int32)
    int_view[idx] ^= torch.bitwise_left_shift(torch.ones_like(b), b)


SAC_CONFIG = dict(
    gabo_max_samples=2048,
    gabo_protect_ratio=0.7,    
    sac_episodes=50,          
    sac_steps_per_episode=8,
    sac_lr=3e-4,
    sac_batch_size=64,
    sac_updates_per_step=2,
    tolerance=0.03,          
    penalty_weight=10.0,
    protect_bonus_weight=0.0,   
    action_scale=0.5,         
    n_fault_trials=5,           
    eval_samples=2048,
    clamp_to_ranger=True,
    verbose=False,
)

OPT_CONFIG = dict(
    gabo_max_samples=2048,
    gabo_protect_ratio=0.7,
    tolerance=0.03,
    penalty_weight=10.0,
    protect_bonus_weight=0.0,
    action_scale=0.5,
    n_fault_trials=5,
    eval_samples=2048,
    clamp_to_ranger=True,
    verbose=False,
)


def make_fault_injector(bitflip="fixed", fault_rate=1e-4, total_bits=32, n_frac=16, n_int=15):
    
    fixed = (bitflip == "fixed")

    def _flip(t):
        if fixed:
            _bitflip_inplace_fixed(t, fault_rate, total_bits, n_frac, n_int)
        else:
            _bitflip_inplace_float(t, fault_rate, total_bits)

    cache = {"model": None, "state": None}

    def _injector(model, data):
        if cache["model"] is not model:
            cache["model"] = model
            cache["state"] = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(cache["state"])
        with torch.no_grad():
            for mod in model.modules():
                if isinstance(mod, nn.Linear):
                    _flip(mod.weight)
                    if mod.bias is not None:
                        _flip(mod.bias)
            pred = model(data)
        return torch.nan_to_num(pred, nan=0.0, posinf=1e4, neginf=-1e4)

    return _injector


def replace_act_all(model, relu_bound_fn, bounds, tresh, alpha=None, name=''):
    
    for name1, layer in model.named_children():
        if list(layer.children()) == []:
            if isinstance(layer, nn.ReLU) and 'last' not in name1:
                name_ = name1 + name
                model._modules[name1] = relu_bound_fn(
                    bounds[name_].detach(), tresh[name_].detach(), alpha)
            elif isinstance(layer, nn.ReLU) and 'last' in name1:
                name_ = name1 + name
                model._modules[name1] = relu_bound_fn(
                    bounds[name_].detach(), tresh[name_].detach(), None, k=-20.0)
        else:
            replace_act_all(layer, relu_bound_fn, bounds, tresh, alpha, name + name1)
    return model


def apply_bound_protection(model, data_loader_dict, device, bitflip="fixed",
                           relu_bound="tresh", search_bound="ranger",
                           total_bits=32, n_frac=16, n_int=15, sac_fault_rate=1e-4,
                           gabo_shrink_ratio=0.4):
    
    if search_bound == "gabo":
        bounds, tresh, _ = GABO_bounds(model, data_loader_dict, device=device,
                                       bound_type='layer', bitflip=bitflip,
                                       shrink_ratio=gabo_shrink_ratio)
    elif search_bound == "gabo_sac":
        bounds, tresh, _ = GABO_SAC_bounds(
            model, data_loader_dict, device=device,
            fault_injector=make_fault_injector(bitflip=bitflip, fault_rate=sac_fault_rate,
                                               total_bits=total_bits, n_frac=n_frac, n_int=n_int),
            **SAC_CONFIG)
    elif search_bound in ("pso", "ga", "bo"):
        opt_fn = {"pso": PSO_bounds, "ga": GA_bounds, "bo": BayesOpt_bounds}[search_bound]
        bounds, tresh, _ = opt_fn(
            model, data_loader_dict, device=device,
            fault_injector=make_fault_injector(bitflip=bitflip, fault_rate=sac_fault_rate,
                                               total_bits=total_bits, n_frac=n_frac, n_int=n_int),
            **OPT_CONFIG)
    else:
        bounds, tresh, _ = Ranger_bounds(model, data_loader_dict, device=device,
                                         bound_type='layer', bitflip=bitflip)
    print(f"[{search_bound}] 每层激活上界(bounds): "
          f"{ {k: round(float(v), 4) for k, v in bounds.items()} }")
    protected = replace_act_all(copy.deepcopy(model), RELU_BOUND_FN[relu_bound], bounds, tresh)
    return protected


SEED = 0
BATCH_SIZE = 128
HIDDEN_DIMS = (256, 192, 128, 64, 32)   
DROPOUT = 0.3
N_CLASS = 4
METRIC_KEYS = ["accuracy", "macro_f1"]


class Config:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MLP(nn.Module):
    

    def __init__(self, input_dim, hidden_dims=HIDDEN_DIMS, n_class=N_CLASS, dropout=DROPOUT):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, n_class))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)   


class TrajDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)   

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


@torch.no_grad()
def eval_metrics(model, loader, device):
    model.eval()
    ys, preds = [], []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        pred = logits.argmax(1).cpu().numpy()
        preds.append(pred)
        ys.append(y.numpy())
    y_true = np.concatenate(ys)
    y_pred = np.concatenate(preds)
    acc = float((y_pred == y_true).mean())
    mf1 = float(f1_score(y_true, y_pred, average="macro"))
    return {
        "accuracy": acc,
        "macro_f1": mf1,
        "val_acc": acc,
        "val_loss": 1.0 - mf1,
    }


def eval_fault(model, data_loader_dict, fault_rate, iterations=20,
               bitflip="fixed", total_bits=32, n_frac=16, n_int=15):
    device = next(model.parameters()).device
    inputs, _ = next(iter(data_loader_dict['val']))

    if bitflip == 'fixed':
        with torch.no_grad():
            for _, param in model.named_parameters():
                if param is not None:
                    param.copy_(torch.tensor(
                        Fxp(param.clone().cpu().numpy(), True,
                            n_word=total_bits, n_frac=n_frac, n_int=n_int).get_val()))

    pfi_model = FaultInjection(
        model, inputs.shape[0],
        input_shape=[inputs.shape[1]],
        layer_types=[torch.nn.Linear, Relu_bound],
        total_bits=total_bits, n_frac=n_frac, n_int=n_int,
        use_cuda=USE_CUDA,
    )

    agg = {k: [] for k in METRIC_KEYS}
    for _ in tqdm(range(iterations), desc=f"Fault {fault_rate:.0e}"):
        if bitflip == 'float':
            corrupted = multi_weight_inj_float(pfi_model, fault_rate)
        elif bitflip == 'fixed':
            corrupted = multi_weight_inj_fixed(pfi_model, fault_rate)
        else:
            corrupted = pfi_model.original_model
        m = eval_metrics(corrupted, data_loader_dict["val"], device)
        for k in METRIC_KEYS:
            agg[k].append(m[k])

    mean = {k: float(np.mean(agg[k])) for k in METRIC_KEYS}
    mean["val_loss"] = 1.0 - mean["macro_f1"]
    mean["val_acc"] = mean["accuracy"]
    mean["fault_rate"] = fault_rate
    return mean


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-base", type=str,
                        default=r"D:\实验2\DNN_SPECTRE_SDN_v1\AIS\trajectory_anomaly_detection\data")
    parser.add_argument("--regions", type=str, default="west,mexico")
    parser.add_argument("--model_path", type=str, default="traj_anomaly_classifier.pt")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0, help="随机种子(数据划分/故障注入)")
    parser.add_argument("--n_word", type=int, default=32)
    parser.add_argument("--n_frac", type=int, default=16)
    parser.add_argument("--n_int", type=int, default=15)
    parser.add_argument("--bitflip", type=str, default="fixed", choices=["fixed", "float"])
    parser.add_argument("--fault_rates", type=str, default="1e-5,3e-5,5e-5,1e-4,3e-4")
    parser.add_argument("--relu-bound", type=str, default="tresh", choices=["tresh", "zero"],
                        help="有界 ReLU 类型")
    parser.add_argument("--search-bound", type=str, default="ga",
                        choices=["ranger", "gabo", "gabo_sac", "pso", "ga", "bo"],
                        help="边界搜索策略: ranger / gabo / gabo_sac / pso / ga / bo(贝叶斯)")
    parser.add_argument("--gabo-shrink-ratio", type=float, default=0.9,
                        help="纯 GABO 收缩因子: λ <= max_act * ratio(0.4=clean 无损下容错最优)")
    parser.add_argument("--sac-fault-rate", type=float, default=1e-4,
                        help="GABO-SAC 内部搜索用的位翻转率(与最终评估故障注入对齐)")
    parser.add_argument("--gabo-max-samples", type=int, default=2048,
                        help="Phase1 GABO 收集激活统计的样本数")
    parser.add_argument("--gabo-protect-ratio", type=float, default=0.0,
                        help="Phase1 GABO 下界保护: λ >= max_act * ratio")
    parser.add_argument("--sac-episodes", type=int, default=30,
                        help="Phase2 SAC episode 数(CPU 慢可调小)")
    parser.add_argument("--sac-steps", type=int, default=8, help="每个 episode 的交互步数")
    parser.add_argument("--sac-lr", type=float, default=1e-4, help="SAC 学习率")
    parser.add_argument("--sac-batch-size", type=int, default=64, help="SAC 采样 batch")
    parser.add_argument("--sac-updates", type=int, default=2, help="每步 SAC 更新次数")
    parser.add_argument("--sac-tolerance", type=float, default=0.03, help="clean 精度允许退化比例")
    parser.add_argument("--sac-penalty", type=float, default=10.0, help="clean 退化惩罚权重")
    parser.add_argument("--sac-protect-bonus", type=float, default=0.0,
                        help="有效保护层数奖励权重(默认 0 关闭, 避免污染 fault 主信号)")
    parser.add_argument("--sac-action-scale", type=float, default=0.2,
                        help="搜索幅度: λ = max_act * exp(action * scale)")
    parser.add_argument("--sac-trials", type=int, default=5, help="每个 action 的故障注入重复次数")
    parser.add_argument("--sac-eval-samples", type=int, default=2048, help="SAC 评估样本数")
    parser.add_argument("--sac-allow-expand", action="store_true",
                        help="允许 λ 超过 Ranger 上界(默认裁剪到 Ranger 以内)")
    parser.add_argument("--sac-verbose", action="store_true", help="打印 GABO-SAC 详细过程")
    parser.add_argument("--refresh", action="store_true", help="忽略缓存,重新准备数据")
    args = parser.parse_args()

    device = Config.DEVICE
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    fault_rates = [float(x) for x in args.fault_rates.split(",") if x.strip()]

    SAC_CONFIG.update(
        gabo_max_samples=args.gabo_max_samples,
        gabo_protect_ratio=args.gabo_protect_ratio,
        sac_episodes=args.sac_episodes,
        sac_steps_per_episode=args.sac_steps,
        sac_lr=args.sac_lr,
        sac_batch_size=args.sac_batch_size,
        sac_updates_per_step=args.sac_updates,
        tolerance=args.sac_tolerance,
        penalty_weight=args.sac_penalty,
        protect_bonus_weight=args.sac_protect_bonus,
        action_scale=args.sac_action_scale,
        n_fault_trials=args.sac_trials,
        eval_samples=args.sac_eval_samples,
        clamp_to_ranger=not args.sac_allow_expand,
        verbose=args.sac_verbose,
    )

    print("== 1) 准备数据 ==")
    feature_cols, (X_train, y_train), (X_valid, y_valid), (X_test, y_test) = \
        prepare_multiclass_data(args.data_base, args.regions, seed=args.seed, refresh=args.refresh)

    print("\n== 2) 加载多分类模型 ==")
    model_path = os.path.join(_SCRIPT_DIR, args.model_path) if not os.path.isabs(args.model_path) else args.model_path
    ckpt = torch.load(model_path, map_location=device, weights_only=False)

    model = MLP(input_dim=ckpt["input_dim"], hidden_dims=ckpt["hidden_dims"],
                n_class=len(ckpt["class_names"]), dropout=DROPOUT).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"[Model] loaded {args.model_path}, params={sum(p.numel() for p in model.parameters()):,}")
    print(f"[Model] 隐藏层: {' -> '.join(map(str, ckpt['hidden_dims']))} -> {len(ckpt['class_names'])}")

    data_loader_dict = {
        "train": DataLoader(TrajDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True),
        "val": DataLoader(TrajDataset(X_valid, y_valid), batch_size=BATCH_SIZE, shuffle=False),
        "test": DataLoader(TrajDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False),
        "sub_train": DataLoader(TrajDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=False),
    }

    if args.bitflip == 'fixed':
        with torch.no_grad():
            for _, param in model.named_parameters():
                if param is not None:
                    param.copy_(torch.tensor(
                        Fxp(param.clone().cpu().numpy(), True,
                            n_word=args.n_word, n_frac=args.n_frac, n_int=args.n_int).get_val()))
    base = eval_metrics(model, data_loader_dict["val"], device)
    print("\n[1] 基线 (定点, 无故障):")
    for k in METRIC_KEYS:
        print(f"    {k:12s}: {base[k]:.4f}")

    print("\n[2] 故障注入测试 (无边界保护):")
    results_no = []

    print(f"\n[3] 边界保护 ({args.search_bound}):")
    model_protected = apply_bound_protection(model, data_loader_dict, device,
                                             bitflip=args.bitflip, relu_bound=args.relu_bound,
                                             search_bound=args.search_bound,
                                             total_bits=args.n_word, n_frac=args.n_frac,
                                             n_int=args.n_int, sac_fault_rate=args.sac_fault_rate,
                                             gabo_shrink_ratio=args.gabo_shrink_ratio)
    base_protect = eval_metrics(model_protected, data_loader_dict["val"], device)
    print(f"  有保护基线(无故障): acc={base_protect['accuracy']:.4f} "
          f"macro_f1={base_protect['macro_f1']:.4f}")

    print(f"\n[4] 故障注入测试 ({args.search_bound} 边界保护):")
    results_protect = []
    for fr in fault_rates:
        res = eval_fault(model_protected, data_loader_dict, fr, args.iterations,
                         args.bitflip, args.n_word, args.n_frac, args.n_int)
        results_protect.append(res)
        print(f"  {fr:.0e}: acc={res['accuracy']:.4f} macro_f1={res['macro_f1']:.4f}")

    print("\n" + "=" * 68)
    print(f"{'Rate':>8s} | {'无保护F1':>9s} | {'有保护F1':>9s} | {'提升':>8s}")
    print("-" * 68)
    for r0, r1 in zip(results_no, results_protect):
        gain = r1['macro_f1'] - r0['macro_f1']
        print(f"{r0['fault_rate']:>8.0e} | {r0['macro_f1']:>9.4f} | "
              f"{r1['macro_f1']:>9.4f} | {gain:>+8.4f}")


if __name__ == "__main__":
    main()
