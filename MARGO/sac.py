"""Phase 2: SAC - global joint bound optimization.

state   : s_k = [m_k, 0.1*m_k, 0.5*m_k, 2.0*m_k]  (m_k = per-layer max activation / Ranger bound)
action  : a in [-1,1]^L  ->  lambda_k = m_k * exp(beta * a_k)
reward  : r = f_main - w_pen * p_clean
  classification : f_main = A_fault - A_base
  trajectory     : f_main = -ln(ADE_fault / ADE_base)
  CRI            : f_main = -ln((1 - R2_fault) / (1 - R2_base))

The optimization is a fixed-context bandit (a single-step MDP), so there is no next
state: the SAC Bellman target reduces to the immediate reward.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from config import SACConfig


# ---------------------------------------------------------------- state / action / reward

def build_state(per_layer_max: list) -> np.ndarray:
    """Build the 4L-dimensional fixed context state."""
    s = []
    for m in per_layer_max:
        s += [m, 0.1 * m, 0.5 * m, 2.0 * m]
    return np.array(s, dtype=np.float32)


def action_to_bounds(action, per_layer_max: list, cfg: SACConfig) -> list:
    """Map action to clipping bounds lambda_k = m_k * exp(beta * a_k), clamped to <= m_k."""
    bounds = []
    for a, m in zip(action, per_layer_max):
        lam = m * np.exp(cfg.beta * a)
        bounds.append(min(lam, m))
    return bounds


def action_prior_from_bounds(warm_bounds: list, per_layer_max: list,
                             cfg: SACConfig) -> np.ndarray:
    """Recover the action prior: a_gabo = (1/beta) * ln(lambda_gabo / m)."""
    a = []
    for lb, m in zip(warm_bounds, per_layer_max):
        a.append((1.0 / cfg.beta) * np.log(min(lb / m, 1.0)))
    return np.array(a, dtype=np.float32)


def compute_reward(task: str, fault_metric: float, base_metric: float,
                   clean_deg: float, cfg: SACConfig) -> float:
    """Compute the reward for a task.

    - classification: f_main = A_fault - A_base, p_clean = 0
    - trajectory/CRI : f_main = -ln(metric_fault / metric_base)
    """
    if task == "classification":
        f_main = fault_metric - base_metric
        p_clean = 0.0
    else:
        f_main = -np.log(fault_metric / base_metric)
        p_clean = max(0.0, clean_deg - (1 + cfg.tau))
    return float(f_main - cfg.w_pen * p_clean)


# ---------------------------------------------------------------- SAC networks

class TanhGaussianPolicy(nn.Module):
    """tanh-Gaussian policy: outputs mean and log-std, actions squashed to [-1,1]."""
    LOG_STD_MIN = -20.0
    LOG_STD_MAX = 2.0

    def __init__(self, state_dim: int, action_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.mean = nn.Linear(hidden, action_dim)
        self.log_std = nn.Linear(hidden, action_dim)

    def forward(self, s: torch.Tensor):
        h = self.net(s)
        mean = self.mean(h)
        log_std = self.log_std(h).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mean, log_std

    def sample(self, s: torch.Tensor):
        """Reparameterized sample with tanh squash; returns (action, log_prob, mean)."""
        mean, log_std = self.forward(s)
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        z = dist.rsample()
        action = torch.tanh(z)
        log_prob = dist.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)
        return action, log_prob, mean


class TwinCritic(nn.Module):
    """Twin soft Q-networks to mitigate overestimation."""
    def __init__(self, state_dim: int, action_dim: int, hidden: int = 256):
        super().__init__()
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1),
        )

    def forward(self, s: torch.Tensor, a: torch.Tensor):
        x = torch.cat([s, a], dim=-1)
        return self.q1(x), self.q2(x)


# ---------------------------------------------------------------- optimization loop

def sac_optimize(model, per_layer_max: list, warm_start_bounds: list,
                 fault_injector, cfg: SACConfig, eval_fn=None, n_steps: int = 1000,
                 device: str = "cpu"):
    """Run the SAC global joint bound optimization (fixed-context bandit).

    Parameters
    ----------
    eval_fn : callable or None
        eval_fn(model, bounds) -> dict with keys 'task', 'fault_metric',
        'base_metric', 'clean_deg'. It should apply `bounds`, run fault injection,
        and evaluate the task metric. If None, a placeholder reward is used so that
        the loop still runs end-to-end for structure verification.

    Returns the final per-layer bounds (list of floats).
    """
    state_dim = 4 * len(per_layer_max)
    action_dim = len(per_layer_max)
    s_np = build_state(per_layer_max)
    s = torch.tensor(s_np, dtype=torch.float32, device=device).unsqueeze(0)

    policy = TanhGaussianPolicy(state_dim, action_dim).to(device)
    critic = TwinCritic(state_dim, action_dim).to(device)
    critic_target = TwinCritic(state_dim, action_dim).to(device)
    critic_target.load_state_dict(critic.state_dict())
    critic_target.eval()

    opt_policy = optim.Adam(policy.parameters(), lr=cfg.lr)
    opt_critic = optim.Adam(critic.parameters(), lr=cfg.lr)

    # Automatic entropy tuning
    log_alpha = torch.tensor(0.0, dtype=torch.float32, device=device, requires_grad=True)
    opt_alpha = optim.Adam([log_alpha], lr=cfg.lr)
    target_entropy = -action_dim

    # Seed the replay buffer with the GABO warm-start action prior.
    a_prior = action_prior_from_bounds(warm_start_bounds, per_layer_max, cfg)
    buffer_a = torch.tensor(a_prior, dtype=torch.float32, device=device).unsqueeze(0)

    for step in range(n_steps):
        # --- sample action (with the warm-start prior injected at the first step)
        if step == 0:
            action = buffer_a
            _, log_prob, _ = policy.sample(s)
        else:
            action, log_prob, _ = policy.sample(s)

        # --- evaluate reward under fault injection
        bounds = action_to_bounds(action[0].detach().cpu().numpy(), per_layer_max, cfg)
        if eval_fn is not None:
            m = eval_fn(model, bounds)
            reward = compute_reward(m["task"], m["fault_metric"], m["base_metric"],
                                    m["clean_deg"], cfg)
        else:
            reward = compute_reward("CRI", 0.5, 0.9, 0.0, cfg)  # placeholder

        reward_t = torch.tensor([[reward]], dtype=torch.float32, device=device)

        # --- critic update (bandit: target == reward)
        q1, q2 = critic(s, action)
        critic_loss = F.mse_loss(q1, reward_t) + F.mse_loss(q2, reward_t)
        opt_critic.zero_grad()
        critic_loss.backward()
        opt_critic.step()

        # --- policy update
        new_action, new_log_prob, _ = policy.sample(s)
        q1_new, q2_new = critic(s, new_action)
        q_min = torch.min(q1_new, q2_new)
        alpha = log_alpha.exp().detach()
        policy_loss = (alpha * new_log_prob - q_min).mean()
        opt_policy.zero_grad()
        policy_loss.backward()
        opt_policy.step()

        # --- entropy temperature update
        alpha_loss = -(log_alpha * (new_log_prob.detach() + target_entropy)).mean()
        opt_alpha.zero_grad()
        alpha_loss.backward()
        opt_alpha.step()

        # --- soft-update target critic
        with torch.no_grad():
            for tp, p in zip(critic_target.parameters(), critic.parameters()):
                tp.data.mul_(0.995).add_(p.data, alpha=0.005)

    # Final deterministic bounds from the learned mean action.
    with torch.no_grad():
        mean, _ = policy.forward(s)
        final_action = torch.tanh(mean)[0].cpu().numpy()

    return action_to_bounds(final_action, per_layer_max, cfg)
