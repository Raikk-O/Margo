
import os, math, importlib.util
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    '_gabo_multiClass_internal', os.path.join(_HERE, 'gabo_multiClass.py'))
_gabo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_gabo)
_gabo_phase = _gabo.gabo_phase
_BoundEvalEnv = _gabo.BoundEvalEnv
_ordered_relu_keys = _gabo._ordered_relu_keys


def _gabo_init(model, train_loader, device, gabo_max_samples, gabo_protect_ratio,
               tolerance, penalty_weight, verbose):
    
    init_lambdas, bounds, tresh, _stats, mse_clean, max_acts = _gabo_phase(
        model=model, train_loader=train_loader, device=device,
        max_samples=gabo_max_samples, protect_ratio=gabo_protect_ratio,
        tolerance=tolerance, penalty_weight=penalty_weight, verbose=verbose)
    layer_keys = [k for k in _ordered_relu_keys(model) if k in max_acts]
    missing = [k for k in max_acts if k not in layer_keys]
    if missing:
        layer_keys.extend(sorted(missing))
    return init_lambdas, bounds, tresh, mse_clean, max_acts, layer_keys


def _build_env(model, train_loader, device, fault_injector, max_acts, layer_keys,
               mse_clean, tolerance, penalty_weight, protect_bonus_weight,
               eval_samples, n_fault_trials, action_scale, clamp_to_ranger):
    return _BoundEvalEnv(
        model=model, data_loader=train_loader, max_acts=max_acts, layer_keys=layer_keys,
        device=device, mse_clean_baseline=mse_clean, fault_injector=fault_injector,
        tolerance=tolerance, penalty_weight=penalty_weight,
        protect_bonus_weight=protect_bonus_weight,
        eval_samples=eval_samples, n_fault_trials=n_fault_trials,
        action_scale=action_scale, clamp_to_ranger=clamp_to_ranger)


def _gabo_action(init_lambdas, max_acts, layer_keys, action_scale):
    
    a = []
    for key in layer_keys:
        m = max_acts[key]
        lam = float(init_lambdas[key])
        v = math.log(max(lam, 1e-8) / max(m, 1e-8)) / action_scale
        a.append(float(np.clip(v, -1.0, 1.0)))
    return np.array(a, dtype=np.float32)


def _finalize(env, best_action, layer_keys, device):
    final_lambdas = env._action_to_lambda(best_action)
    results = {}
    for i, key in enumerate(layer_keys):
        results[key] = torch.tensor(final_lambdas[i], device=device)
    return results


def _report(name, best_f, reward_gabo, best_info):
    print(f"  [{name}] best_r={best_f:.6f}  GABO_init={reward_gabo:.6f}  "
          f"Δ={best_f - reward_gabo:+.6f}  clean={best_info['mse_clean']:.6f}  "
          f"fault={best_info['mse_fault']:.6f}  prot={best_info['protect_ratio']:.2f}")


def PSO_bounds(model, train_loader, device="cpu", fault_injector=None,
               gabo_max_samples=2048, gabo_protect_ratio=0.7,
               pop_size=20, n_iter=13, w=0.7, c1=1.5, c2=1.5,
               tolerance=0.03, penalty_weight=10.0, protect_bonus_weight=0.0,
               action_scale=0.5, n_fault_trials=5, eval_samples=2048,
               clamp_to_ranger=True, verbose=False):
    print("=" * 70)
    print(" PSO: GABO warm-start + Particle Swarm Optimisation "
          f"({'Shrink-only' if clamp_to_ranger else 'Bidirectional'})")
    print("=" * 70)

    init_lambdas, bounds, tresh, mse_clean, max_acts, layer_keys = _gabo_init(
        model, train_loader, device, gabo_max_samples, gabo_protect_ratio,
        tolerance, penalty_weight, verbose)
    action_dim = len(layer_keys)
    if action_dim == 0:
        return {}, {}, None

    env = _build_env(model, train_loader, device, fault_injector, max_acts, layer_keys,
                     mse_clean, tolerance, penalty_weight, protect_bonus_weight,
                     eval_samples, n_fault_trials, action_scale, clamp_to_ranger)
    g0 = _gabo_action(init_lambdas, max_acts, layer_keys, action_scale)
    reward_gabo, _ = env.evaluate(g0)

    rng = np.random.default_rng(0)
    X = np.empty((pop_size, action_dim), dtype=np.float32)
    X[0] = g0
    for i in range(1, pop_size):
        if i % 2 == 0:
            X[i] = np.clip(g0 + rng.normal(0.0, 0.3, action_dim), -1.0, 1.0)
        else:
            X[i] = rng.uniform(-1.0, 0.0, size=action_dim).astype(np.float32)
    V = np.zeros_like(X)
    pbest_X = X.copy()
    pbest_f = np.full(pop_size, -np.inf)
    gbest_X = g0.copy()
    gbest_f = -np.inf
    gbest_info = None

    def _eval(a):
        r, info = env.evaluate(a)
        return r, info

    for i in range(pop_size):
        r, info = _eval(X[i])
        pbest_f[i] = r
        if r > gbest_f:
            gbest_f, gbest_X, gbest_info = r, X[i].copy(), info

    _report("PSO", gbest_f, reward_gabo, gbest_info)
    for it in range(n_iter):
        for i in range(pop_size):
            rp = rng.random(action_dim)
            rg = rng.random(action_dim)
            V[i] = w * V[i] + c1 * rp * (pbest_X[i] - X[i]) + c2 * rg * (gbest_X - X[i])
            X[i] = np.clip(X[i] + V[i], -1.0, 1.0)
            r, info = _eval(X[i])
            if r > pbest_f[i]:
                pbest_f[i], pbest_X[i] = r, X[i].copy()
            if r > gbest_f:
                gbest_f, gbest_X, gbest_info = r, X[i].copy(), info
        if verbose and (it + 1) % 5 == 0:
            _report("PSO", gbest_f, reward_gabo, gbest_info)

    _report("PSO-final", gbest_f, reward_gabo, gbest_info)
    return _finalize(env, gbest_X, layer_keys, device), tresh, None


def GA_bounds(model, train_loader, device="cpu", fault_injector=None,
              gabo_max_samples=2048, gabo_protect_ratio=0.7,
              pop_size=20, n_iter=13, mutation_rate=0.1, mutation_scale=0.3,
              elite=2, tournament_k=3,
              tolerance=0.03, penalty_weight=10.0, protect_bonus_weight=0.0,
              action_scale=0.5, n_fault_trials=5, eval_samples=2048,
              clamp_to_ranger=True, verbose=False):
    print("=" * 70)
    print(" GA: GABO warm-start + Genetic Algorithm "
          f"({'Shrink-only' if clamp_to_ranger else 'Bidirectional'})")
    print("=" * 70)

    init_lambdas, bounds, tresh, mse_clean, max_acts, layer_keys = _gabo_init(
        model, train_loader, device, gabo_max_samples, gabo_protect_ratio,
        tolerance, penalty_weight, verbose)
    action_dim = len(layer_keys)
    if action_dim == 0:
        return {}, {}, None

    env = _build_env(model, train_loader, device, fault_injector, max_acts, layer_keys,
                     mse_clean, tolerance, penalty_weight, protect_bonus_weight,
                     eval_samples, n_fault_trials, action_scale, clamp_to_ranger)
    g0 = _gabo_action(init_lambdas, max_acts, layer_keys, action_scale)
    reward_gabo, _ = env.evaluate(g0)

    rng = np.random.default_rng(0)
    pop = np.empty((pop_size, action_dim), dtype=np.float32)
    pop[0] = g0
    for i in range(1, pop_size):
        if i % 2 == 0:
            pop[i] = np.clip(g0 + rng.normal(0.0, 0.3, action_dim), -1.0, 1.0)
        else:
            pop[i] = rng.uniform(-1.0, 0.0, size=action_dim).astype(np.float32)

    fit = np.array([env.evaluate(a)[0] for a in pop])
    best_idx = int(np.argmax(fit))
    gbest_X = pop[best_idx].copy()
    gbest_f = float(fit[best_idx])
    _, gbest_info = env.evaluate(gbest_X)

    _report("GA", gbest_f, reward_gabo, gbest_info)
    for it in range(n_iter):
        order = np.argsort(-fit)
        new_pop = [pop[order[j]].copy() for j in range(elite)]
        while len(new_pop) < pop_size:
            def _tournament():
                cand = rng.choice(pop_size, tournament_k, replace=False)
                return pop[int(cand[np.argmax(fit[cand])])].copy()
            p1, p2 = _tournament(), _tournament()
            mask = rng.random(action_dim) < 0.5
            child = np.where(mask, p1, p2).astype(np.float32)
            child += rng.normal(0.0, mutation_scale, action_dim).astype(np.float32) * \
                (rng.random(action_dim) < mutation_rate)
            child = np.clip(child, -1.0, 1.0)
            new_pop.append(child)
        pop = np.stack(new_pop)

        fit = np.array([env.evaluate(a)[0] for a in pop])
        bi = int(np.argmax(fit))
        if float(fit[bi]) > gbest_f:
            gbest_f = float(fit[bi])
            gbest_X = pop[bi].copy()
            _, gbest_info = env.evaluate(gbest_X)
        if verbose and (it + 1) % 5 == 0:
            _report("GA", gbest_f, reward_gabo, gbest_info)

    _report("GA-final", gbest_f, reward_gabo, gbest_info)
    return _finalize(env, gbest_X, layer_keys, device), tresh, None


def BayesOpt_bounds(model, train_loader, device="cpu", fault_injector=None,
                    gabo_max_samples=2048, gabo_protect_ratio=0.7,
                    n_init=20, n_iter=260,
                    tolerance=0.03, penalty_weight=10.0, protect_bonus_weight=0.0,
                    action_scale=0.5, n_fault_trials=5, eval_samples=2048,
                    clamp_to_ranger=True, verbose=False):
    print("=" * 70)
    print(" BO: GABO warm-start + Bayesian Optimisation (GP + EI) "
          f"({'Shrink-only' if clamp_to_ranger else 'Bidirectional'})")
    print("=" * 70)

    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, ConstantKernel
    from scipy.stats import norm

    init_lambdas, bounds, tresh, mse_clean, max_acts, layer_keys = _gabo_init(
        model, train_loader, device, gabo_max_samples, gabo_protect_ratio,
        tolerance, penalty_weight, verbose)
    action_dim = len(layer_keys)
    if action_dim == 0:
        return {}, {}, None

    env = _build_env(model, train_loader, device, fault_injector, max_acts, layer_keys,
                     mse_clean, tolerance, penalty_weight, protect_bonus_weight,
                     eval_samples, n_fault_trials, action_scale, clamp_to_ranger)
    g0 = _gabo_action(init_lambdas, max_acts, layer_keys, action_scale)
    reward_gabo, _ = env.evaluate(g0)

    rng = np.random.default_rng(0)
    X = [g0]
    y = [reward_gabo]
    for _ in range(n_init - 1):
        a = rng.uniform(-1.0, 0.0, size=action_dim).astype(np.float32)
        r, _ = env.evaluate(a)
        X.append(a)
        y.append(r)
    X = np.stack(X)
    y = np.array(y, dtype=np.float64)

    best_i = int(np.argmax(y))
    gbest_X = X[best_i].copy()
    gbest_f = float(y[best_i])
    _, gbest_info = env.evaluate(gbest_X)

    _report("BO", gbest_f, reward_gabo, gbest_info)

    kernel = ConstantKernel(1.0) * Matern(nu=2.5)
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=0)

    for it in range(n_iter):
        gp.fit(X, y)
        mu, sigma = gp.predict(X, return_std=True)
        sigma = np.maximum(sigma, 1e-6)
        best_sofar = float(np.max(y))
        z = (mu - best_sofar) / sigma
        ei = (mu - best_sofar) * norm.cdf(z) + sigma * norm.pdf(z)

        cands = []
        top_i = int(np.argmax(ei))
        cands.append(np.clip(X[top_i] + rng.normal(0.0, 0.2, action_dim), -1.0, 1.0))
        for _ in range(63):
            cands.append(rng.uniform(-1.0, 0.0, size=action_dim))
        cands = np.stack(cands).astype(np.float64)
        mu_c, sigma_c = gp.predict(cands, return_std=True)
        sigma_c = np.maximum(sigma_c, 1e-6)
        z_c = (mu_c - best_sofar) / sigma_c
        ei_c = (mu_c - best_sofar) * norm.cdf(z_c) + sigma_c * norm.pdf(z_c)
        a_new = cands[int(np.argmax(ei_c))].astype(np.float32)
        r, info = env.evaluate(a_new)
        X = np.concatenate([X, a_new[None, :]], axis=0)
        y = np.concatenate([y, [r]])
        if r > gbest_f:
            gbest_f = r
            gbest_X = a_new.copy()
            gbest_info = info
        if verbose and (it + 1) % 50 == 0:
            _report("BO", gbest_f, reward_gabo, gbest_info)

    _report("BO-final", gbest_f, reward_gabo, gbest_info)
    return _finalize(env, gbest_X, layer_keys, device), tresh, None
