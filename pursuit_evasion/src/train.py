"""PPO training (CleanRL style) for the pursuit-evasion environment.

Two atomic training functions:

* :func:`train_attacker` — updates a :class:`NeuralAttackerNet` against a
  fixed (heuristic or frozen-neural) defender opponent.
* :func:`train_defender` — updates a :class:`NeuralDefenderNet` (shared
  parameters, k agents per env) against a fixed attacker opponent.

The high-level alternating self-play loop is :func:`alternating_train`.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .env import PursuitEvasionEnv
from .policies import (
    CentralizedDefender,
    CentralizedDefenderNet,
    HeuristicAttacker,
    HeuristicDefender,
    NeuralAttacker,
    NeuralAttackerNet,
    NeuralDefender,
    NeuralDefenderNet,
    PopulationAttacker,
)


# ---------------------------------------------------------------- types & cfg


class _ActsOnEnv(Protocol):
    """Anything that can produce an action from an env."""

    def act(self, env: PursuitEvasionEnv) -> np.ndarray: ...


@dataclass
class PPOConfig:
    """Standard PPO hyperparameters (CleanRL defaults)."""

    n_steps: int = 128
    n_epochs: int = 4
    minibatch_size: int = 512
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5


# -------------------------------------------------------------- shared helpers


def _gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    next_value: torch.Tensor,
    next_done: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generalized Advantage Estimation.

    All tensors should share leading shape ``(T, ...)`` (where ``...`` is the
    parallel-trajectory dim — ``B`` for attacker, ``B*k`` flattened for the
    defender). ``next_value`` and ``next_done`` are shape ``(...)`` —
    bootstrap values after the rollout.
    """
    T = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros_like(rewards[0])
    for t in reversed(range(T)):
        if t == T - 1:
            non_term = 1.0 - next_done
            next_v = next_value
        else:
            non_term = 1.0 - dones[t + 1]
            next_v = values[t + 1]
        delta = rewards[t] + gamma * next_v * non_term - values[t]
        last_gae = delta + gamma * gae_lambda * non_term * last_gae
        advantages[t] = last_gae
    returns = advantages + values
    return advantages, returns


def _ppo_update(
    net: nn.Module,
    optimizer: optim.Optimizer,
    obs: torch.Tensor,
    actions: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    old_values: torch.Tensor,
    cfg: PPOConfig,
) -> tuple[float, float, float]:
    """Run ``n_epochs`` of PPO updates over the buffered rollout.

    Returns
    -------
    (mean_policy_loss, mean_value_loss, mean_entropy) over the epochs.
    """
    n = obs.shape[0]
    idxs = np.arange(n)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    pl_acc, vl_acc, ent_acc, count = 0.0, 0.0, 0.0, 0
    for _ in range(cfg.n_epochs):
        np.random.shuffle(idxs)
        for start in range(0, n, cfg.minibatch_size):
            mb = idxs[start : start + cfg.minibatch_size]
            mb_obs = obs[mb]
            mb_act = actions[mb]
            mb_oldlp = old_logprobs[mb]
            mb_adv = advantages[mb]
            mb_ret = returns[mb]

            mean, logstd, value = net(mb_obs)
            std = logstd.exp()
            dist = torch.distributions.Normal(mean, std)
            new_lp = dist.log_prob(mb_act).sum(-1)
            entropy = dist.entropy().sum(-1).mean()

            ratio = torch.exp(new_lp - mb_oldlp)
            unclipped = ratio * mb_adv
            clipped = torch.clamp(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef) * mb_adv
            policy_loss = -torch.min(unclipped, clipped).mean()
            value_loss = 0.5 * ((value - mb_ret) ** 2).mean()
            loss = policy_loss + cfg.vf_coef * value_loss - cfg.ent_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), cfg.max_grad_norm)
            optimizer.step()

            pl_acc += float(policy_loss.detach())
            vl_acc += float(value_loss.detach())
            ent_acc += float(entropy.detach())
            count += 1
    return pl_acc / count, vl_acc / count, ent_acc / count


def _episode_tracker_init(B: int) -> dict:
    """Per-env scratch state for accumulating returns/lengths and successes."""
    return {
        "ret": np.zeros(B, dtype=np.float64),
        "len": np.zeros(B, dtype=np.int64),
        "ep_rets": [],
        "ep_lens": [],
        "ep_outcomes": [],
    }


def _episode_tracker_step(
    tracker: dict,
    rewards: np.ndarray,
    just_done: np.ndarray,
    outcomes: np.ndarray,
) -> None:
    tracker["ret"] += rewards
    tracker["len"] += 1
    if just_done.any():
        idx = np.where(just_done)[0]
        tracker["ep_rets"].extend(tracker["ret"][idx].tolist())
        tracker["ep_lens"].extend(tracker["len"][idx].tolist())
        tracker["ep_outcomes"].extend(outcomes[idx].tolist())
        tracker["ret"][idx] = 0.0
        tracker["len"][idx] = 0


# ============================================================================
# Attacker training (k-agnostic from the policy POV: 1 agent per env)
# ============================================================================


def train_attacker(
    env: PursuitEvasionEnv,
    net: NeuralAttackerNet,
    defender_policy: _ActsOnEnv,
    *,
    n_iters: int,
    cfg: PPOConfig,
    optimizer: optim.Optimizer | None = None,
    device: torch.device | None = None,
    log_every: int = 10,
    log_fn=print,
    snapshot_iters: list[int] | None = None,
) -> dict:
    """PPO-train the attacker network against a fixed defender opponent.

    Parameters
    ----------
    env : PursuitEvasionEnv
        Env with ``B = n_envs``. Will be reset internally.
    net : NeuralAttackerNet
    defender_policy : object with ``.act(env)`` returning ``(B, k, 2)``.
    n_iters : int
    cfg : PPOConfig
    optimizer : optional, defaults to Adam(lr=cfg.lr).
    device : optional, defaults to CPU.

    Returns
    -------
    dict with logged metrics list.
    """
    device = device or torch.device("cpu")
    if optimizer is None:
        optimizer = optim.Adam(net.parameters(), lr=cfg.lr, eps=1e-5)
    net.to(device)

    B = env.B
    obs_dim = env.attacker_obs_dim
    T = cfg.n_steps

    env.reset()
    a_obs, _ = env._get_obs()
    next_obs = torch.as_tensor(a_obs, device=device)
    next_done = torch.zeros(B, device=device)

    metrics: list[dict] = []
    snapshots: list[dict] = []  # list of state_dict copies at requested iters
    snapshot_set = set(snapshot_iters or [])
    t_start = time.perf_counter()
    total_steps = 0

    for it in range(n_iters):
        # ----- rollout -----
        obs_buf = torch.zeros(T, B, obs_dim, device=device)
        act_buf = torch.zeros(T, B, 2, device=device)
        lp_buf = torch.zeros(T, B, device=device)
        val_buf = torch.zeros(T, B, device=device)
        rew_buf = torch.zeros(T, B, device=device)
        done_buf = torch.zeros(T, B, device=device)

        tracker = _episode_tracker_init(B)

        for t in range(T):
            obs_buf[t] = next_obs
            done_buf[t] = next_done

            with torch.no_grad():
                mean, logstd, value = net(next_obs)
                std = logstd.exp()
                dist = torch.distributions.Normal(mean, std)
                action = dist.sample()
                logprob = dist.log_prob(action).sum(-1)

            d_action = defender_policy.act(env)
            a_action_np = PursuitEvasionEnv._clip_speed(
                action.cpu().numpy(), env.v_attacker
            )
            res = env.step(a_action_np, d_action)

            act_buf[t] = action
            lp_buf[t] = logprob
            val_buf[t] = value
            rew_buf[t] = torch.as_tensor(res.attacker_reward, device=device)

            _episode_tracker_step(
                tracker, res.attacker_reward, res.just_done, res.outcome
            )
            if res.just_done.any():
                env.reset_idxs(res.just_done)

            a_obs, _ = env._get_obs()
            next_obs = torch.as_tensor(a_obs, device=device)
            next_done = torch.as_tensor(res.just_done.astype(np.float32), device=device)

        # bootstrap value for the final state
        with torch.no_grad():
            _, _, next_value = net(next_obs)

        advantages, returns = _gae(
            rew_buf, val_buf, done_buf, next_value, next_done, cfg.gamma, cfg.gae_lambda
        )

        # ----- update -----
        flat_obs = obs_buf.reshape(-1, obs_dim)
        flat_act = act_buf.reshape(-1, 2)
        flat_lp = lp_buf.reshape(-1)
        flat_adv = advantages.reshape(-1)
        flat_ret = returns.reshape(-1)
        flat_val = val_buf.reshape(-1)

        pl, vl, ent = _ppo_update(
            net, optimizer, flat_obs, flat_act, flat_lp, flat_adv, flat_ret, flat_val, cfg
        )

        total_steps += T * B
        elapsed = time.perf_counter() - t_start
        succ = (
            float(np.mean([o == 1 for o in tracker["ep_outcomes"]]))
            if tracker["ep_outcomes"]
            else float("nan")
        )
        m = {
            "iter": it,
            "step": total_steps,
            "ep_return": float(np.mean(tracker["ep_rets"])) if tracker["ep_rets"] else float("nan"),
            "ep_length": float(np.mean(tracker["ep_lens"])) if tracker["ep_lens"] else float("nan"),
            "attacker_succ": succ,
            "policy_loss": pl,
            "value_loss": vl,
            "entropy": ent,
            "fps": total_steps / max(elapsed, 1e-6),
            "n_eps": len(tracker["ep_rets"]),
        }
        metrics.append(m)
        if log_every and (it % log_every == 0 or it == n_iters - 1):
            log_fn(
                f"  [att it={it:3d} step={total_steps:>7d}] "
                f"ret={m['ep_return']:+.3f} len={m['ep_length']:5.1f} "
                f"succ={m['attacker_succ']:.3f} "
                f"pl={pl:+.4f} vl={vl:.4f} H={ent:+.3f} fps={m['fps']:.0f}"
            )
        if it in snapshot_set:
            snapshots.append({k: v.detach().cpu().clone() for k, v in net.state_dict().items()})
            log_fn(f"  [att it={it:3d}] snapshot saved (#{len(snapshots)})")
    return {"metrics": metrics, "snapshots": snapshots}


# ============================================================================
# Defender training: k agents per env, shared parameters
# ============================================================================


def train_defender(
    env: PursuitEvasionEnv,
    net: NeuralDefenderNet,
    attacker_policy: _ActsOnEnv,
    *,
    n_iters: int,
    cfg: PPOConfig,
    optimizer: optim.Optimizer | None = None,
    device: torch.device | None = None,
    log_every: int = 10,
    log_fn=print,
) -> dict:
    """PPO-train the shared-param defender net against a fixed attacker.

    Each defender is treated as an independent rollout sample (with its own
    obs and action), but all defenders share parameters and receive the
    common team reward.
    """
    device = device or torch.device("cpu")
    if optimizer is None:
        optimizer = optim.Adam(net.parameters(), lr=cfg.lr, eps=1e-5)
    net.to(device)

    B = env.B
    k = env.k
    obs_dim = env.defender_obs_dim
    T = cfg.n_steps

    env.reset()
    _, d_obs = env._get_obs()
    next_obs = torch.as_tensor(d_obs.reshape(B * k, obs_dim), device=device)
    next_done = torch.zeros(B * k, device=device)

    metrics: list[dict] = []
    t_start = time.perf_counter()
    total_steps = 0

    for it in range(n_iters):
        obs_buf = torch.zeros(T, B * k, obs_dim, device=device)
        act_buf = torch.zeros(T, B * k, 2, device=device)
        lp_buf = torch.zeros(T, B * k, device=device)
        val_buf = torch.zeros(T, B * k, device=device)
        rew_buf = torch.zeros(T, B * k, device=device)
        done_buf = torch.zeros(T, B * k, device=device)

        tracker = _episode_tracker_init(B)

        for t in range(T):
            obs_buf[t] = next_obs
            done_buf[t] = next_done

            with torch.no_grad():
                mean, logstd, value = net(next_obs)
                std = logstd.exp()
                dist = torch.distributions.Normal(mean, std)
                action = dist.sample()
                logprob = dist.log_prob(action).sum(-1)

            a_action = attacker_policy.act(env)
            d_action_np = PursuitEvasionEnv._clip_speed(
                action.cpu().numpy().reshape(B, k, 2), env.v_defender
            )
            res = env.step(a_action, d_action_np)

            act_buf[t] = action
            lp_buf[t] = logprob
            val_buf[t] = value
            rew_buf[t] = torch.as_tensor(
                np.broadcast_to(res.defender_reward[:, None], (B, k))
                .reshape(B * k)
                .copy(),
                device=device,
            )

            _episode_tracker_step(
                tracker, res.defender_reward, res.just_done, res.outcome
            )
            if res.just_done.any():
                env.reset_idxs(res.just_done)

            _, d_obs = env._get_obs()
            next_obs = torch.as_tensor(d_obs.reshape(B * k, obs_dim), device=device)
            done_b = res.just_done.astype(np.float32)
            next_done = torch.as_tensor(
                np.broadcast_to(done_b[:, None], (B, k)).reshape(B * k).copy(),
                device=device,
            )

        with torch.no_grad():
            _, _, next_value = net(next_obs)

        advantages, returns = _gae(
            rew_buf, val_buf, done_buf, next_value, next_done, cfg.gamma, cfg.gae_lambda
        )

        flat_obs = obs_buf.reshape(-1, obs_dim)
        flat_act = act_buf.reshape(-1, 2)
        flat_lp = lp_buf.reshape(-1)
        flat_adv = advantages.reshape(-1)
        flat_ret = returns.reshape(-1)
        flat_val = val_buf.reshape(-1)

        pl, vl, ent = _ppo_update(
            net, optimizer, flat_obs, flat_act, flat_lp, flat_adv, flat_ret, flat_val, cfg
        )

        total_steps += T * B
        elapsed = time.perf_counter() - t_start
        # defender 'success' = attacker did NOT win
        not_att = float(
            np.mean([o != 1 for o in tracker["ep_outcomes"]])
        ) if tracker["ep_outcomes"] else float("nan")
        m = {
            "iter": it,
            "step": total_steps,
            "ep_return": float(np.mean(tracker["ep_rets"])) if tracker["ep_rets"] else float("nan"),
            "ep_length": float(np.mean(tracker["ep_lens"])) if tracker["ep_lens"] else float("nan"),
            "defender_succ": not_att,
            "policy_loss": pl,
            "value_loss": vl,
            "entropy": ent,
            "fps": total_steps / max(elapsed, 1e-6),
            "n_eps": len(tracker["ep_rets"]),
        }
        metrics.append(m)
        if log_every and (it % log_every == 0 or it == n_iters - 1):
            log_fn(
                f"  [def it={it:3d} step={total_steps:>7d}] "
                f"ret={m['ep_return']:+.3f} len={m['ep_length']:5.1f} "
                f"d_succ={m['defender_succ']:.3f} "
                f"pl={pl:+.4f} vl={vl:.4f} H={ent:+.3f} fps={m['fps']:.0f}"
            )
    return {"metrics": metrics}


# ============================================================================
# Alternating loop
# ============================================================================


def alternating_train(
    *,
    k: int,
    sigma: float,
    p: float,
    cfg: PPOConfig,
    n_envs: int,
    attacker_warmup_iters: int,
    defender_iters: int,
    attacker_iters: int,
    n_alternations: int,
    seed: int = 0,
    env_kwargs: dict | None = None,
    save_dir: Path | str | None = None,
    log_fn=print,
) -> dict:
    """Run the full alternating self-play schedule for one (k, σ, p) cell.

    Schedule:
      Phase 0: attacker (neural) trains against heuristic defender for
      ``attacker_warmup_iters`` PPO iters.
      Then ``n_alternations`` rounds of:
        Phase A: freeze attacker, train neural defender for ``defender_iters``
        Phase B: freeze defender, train neural attacker for ``attacker_iters``

    Returns
    -------
    dict
        Trained ``attacker_net``, ``defender_net``, and per-phase ``metrics``
        lists.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    env_kwargs = env_kwargs or {}

    env = PursuitEvasionEnv(
        batch_size=n_envs, k=k, sigma=sigma, p=p, seed=seed, **env_kwargs
    )

    attacker_net = NeuralAttackerNet(env.attacker_obs_dim)
    defender_net = NeuralDefenderNet(k=k)

    log_fn(
        f"=== alternating_train  k={k} sigma={sigma} p={p}  "
        f"warmup={attacker_warmup_iters} alt={n_alternations}x"
        f"(def={defender_iters}, att={attacker_iters})  B={n_envs} T={cfg.n_steps} ==="
    )

    heuristic_defender = HeuristicDefender(
        v_defender=env.v_defender, capture_radius=env.capture_radius
    )

    all_metrics: dict[str, list] = {"warmup": [], "rounds": []}

    # Phase 0
    log_fn("[phase 0] attacker warmup vs heuristic defender")
    res = train_attacker(
        env, attacker_net, heuristic_defender,
        n_iters=attacker_warmup_iters, cfg=cfg, log_fn=log_fn,
    )
    all_metrics["warmup"] = res["metrics"]

    for r in range(n_alternations):
        round_log: dict = {"defender": [], "attacker": []}

        # Phase A: train defender vs frozen attacker
        log_fn(f"[round {r+1}/{n_alternations} A] train defender")
        attacker_frozen = NeuralAttacker(attacker_net, env.v_attacker, deterministic=False)
        env.reset()
        res_d = train_defender(
            env, defender_net, attacker_frozen,
            n_iters=defender_iters, cfg=cfg, log_fn=log_fn,
        )
        round_log["defender"] = res_d["metrics"]

        # Phase B: train attacker vs frozen defender
        log_fn(f"[round {r+1}/{n_alternations} B] train attacker")
        defender_frozen = NeuralDefender(defender_net, env.v_defender, deterministic=False)
        env.reset()
        res_a = train_attacker(
            env, attacker_net, defender_frozen,
            n_iters=attacker_iters, cfg=cfg, log_fn=log_fn,
        )
        round_log["attacker"] = res_a["metrics"]
        all_metrics["rounds"].append(round_log)

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(attacker_net.state_dict(), save_dir / "attacker.pt")
        torch.save(defender_net.state_dict(), save_dir / "defender.pt")
        log_fn(f"saved checkpoints → {save_dir}")

    phases: list[dict] = [
        {
            "name": "warmup",
            "trainee": "neural_attacker",
            "opponent": "HeuristicDefender",
            "n_iters": attacker_warmup_iters,
        }
    ]
    for r in range(n_alternations):
        phases.append({
            "name": f"round_{r+1}_defender",
            "trainee": "neural_defender",
            "opponent": "frozen_neural_attacker",
            "n_iters": defender_iters,
        })
        phases.append({
            "name": f"round_{r+1}_attacker",
            "trainee": "neural_attacker",
            "opponent": "frozen_neural_defender",
            "n_iters": attacker_iters,
        })
    env_steps_per_iter = n_envs * cfg.n_steps
    total_iters = (
        attacker_warmup_iters + n_alternations * (attacker_iters + defender_iters)
    )
    schedule = {
        "method": "alternating_train",
        "k": k, "sigma": sigma, "p": p, "seed": seed,
        "n_envs": n_envs, "n_steps": cfg.n_steps,
        "attacker_warmup_iters": attacker_warmup_iters,
        "defender_iters": defender_iters,
        "attacker_iters": attacker_iters,
        "n_alternations": n_alternations,
        "phases": phases,
        "totals": {
            "attacker_iters": attacker_warmup_iters + n_alternations * attacker_iters,
            "defender_iters": n_alternations * defender_iters,
            "env_steps_per_iter": env_steps_per_iter,
            "total_env_steps": total_iters * env_steps_per_iter,
        },
    }

    return {
        "attacker_net": attacker_net,
        "defender_net": defender_net,
        "metrics": all_metrics,
        "env_meta": env.metadata(),
        "schedule": schedule,
    }


# ============================================================================
# Independent training: each side trains against a *fixed heuristic* opponent.
# No alternation, no oscillation — just two best-response policies.
# ============================================================================


def independent_train(
    *,
    k: int,
    sigma: float,
    p: float,
    cfg: PPOConfig,
    n_envs: int,
    attacker_iters: int,
    defender_iters: int,
    seed: int = 0,
    env_kwargs: dict | None = None,
    save_dir: Path | str | None = None,
    log_fn=print,
) -> dict:
    """Train attacker and defender independently, each vs a heuristic opponent.

    Phase A: train ``NeuralAttackerNet`` vs ``HeuristicDefender`` for
    ``attacker_iters`` PPO iters.

    Phase B: starting from a *fresh env*, train ``NeuralDefenderNet`` vs
    ``HeuristicAttacker`` for ``defender_iters`` PPO iters.

    Neither side ever sees the other; both have a stationary target to
    optimize against. This avoids the alternation oscillation entirely.

    Returns
    -------
    dict with ``attacker_net``, ``defender_net``, ``metrics``.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    env_kwargs = env_kwargs or {}

    log_fn(
        f"=== independent_train  k={k} sigma={sigma} p={p}  "
        f"attacker_iters={attacker_iters}  defender_iters={defender_iters}  "
        f"B={n_envs} T={cfg.n_steps} ==="
    )

    # ---- Phase A: attacker vs heuristic defender ----
    log_fn("[phase A] train attacker vs heuristic defender")
    env_a = PursuitEvasionEnv(
        batch_size=n_envs, k=k, sigma=sigma, p=p, seed=seed, **env_kwargs
    )
    attacker_net = NeuralAttackerNet(env_a.attacker_obs_dim)
    heuristic_defender = HeuristicDefender(
        v_defender=env_a.v_defender, capture_radius=env_a.capture_radius
    )
    a_log = train_attacker(
        env_a,
        attacker_net,
        heuristic_defender,
        n_iters=attacker_iters,
        cfg=cfg,
        log_fn=log_fn,
    )

    # ---- Phase B: defender vs heuristic attacker ----
    log_fn("[phase B] train defender vs heuristic attacker")
    env_b = PursuitEvasionEnv(
        batch_size=n_envs, k=k, sigma=sigma, p=p, seed=seed + 1, **env_kwargs
    )
    defender_net = NeuralDefenderNet(k=k)
    heuristic_attacker = HeuristicAttacker(
        target=env_b.target, v_attacker=env_b.v_attacker
    )
    d_log = train_defender(
        env_b,
        defender_net,
        heuristic_attacker,
        n_iters=defender_iters,
        cfg=cfg,
        log_fn=log_fn,
    )

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(attacker_net.state_dict(), save_dir / "attacker.pt")
        torch.save(defender_net.state_dict(), save_dir / "defender.pt")
        log_fn(f"saved checkpoints → {save_dir}")

    return {
        "attacker_net": attacker_net,
        "defender_net": defender_net,
        "metrics": {"attacker": a_log["metrics"], "defender": d_log["metrics"]},
        "env_meta": env_a.metadata(),
    }


# ============================================================================
# Centralized defender training
# ============================================================================


def train_defender_centralized(
    env: PursuitEvasionEnv,
    net: CentralizedDefenderNet,
    attacker_policy: _ActsOnEnv,
    *,
    n_iters: int,
    cfg: PPOConfig,
    optimizer: optim.Optimizer | None = None,
    device: torch.device | None = None,
    log_every: int = 10,
    log_fn=print,
    population: PopulationAttacker | None = None,
) -> dict:
    """PPO-train the centralized defender (one brain, k×2-D action) against
    a fixed attacker policy.

    If ``population`` is supplied, it is used as the attacker for actions and
    is reshuffled each time envs reset — this gives the defender opponent
    diversity (mixture of attacker snapshots / strengths).
    """
    device = device or torch.device("cpu")
    if optimizer is None:
        optimizer = optim.Adam(net.parameters(), lr=cfg.lr, eps=1e-5)
    net.to(device)

    B = env.B
    k = env.k
    obs_dim = 4 + 4 * k
    action_dim = 2 * k
    T = cfg.n_steps

    env.reset()
    next_obs = torch.as_tensor(CentralizedDefenderNet.build_obs(env), device=device)
    next_done = torch.zeros(B, device=device)
    if population is not None:
        population.reshuffle()

    metrics: list[dict] = []
    t_start = time.perf_counter()
    total_steps = 0

    for it in range(n_iters):
        obs_buf = torch.zeros(T, B, obs_dim, device=device)
        act_buf = torch.zeros(T, B, action_dim, device=device)
        lp_buf = torch.zeros(T, B, device=device)
        val_buf = torch.zeros(T, B, device=device)
        rew_buf = torch.zeros(T, B, device=device)
        done_buf = torch.zeros(T, B, device=device)

        tracker = _episode_tracker_init(B)

        for t in range(T):
            obs_buf[t] = next_obs
            done_buf[t] = next_done

            with torch.no_grad():
                mean, logstd, value = net(next_obs)
                std = logstd.exp()
                dist = torch.distributions.Normal(mean, std)
                action = dist.sample()  # (B, 2k)
                logprob = dist.log_prob(action).sum(-1)

            opp = population if population is not None else attacker_policy
            a_action = opp.act(env)
            d_action_np = PursuitEvasionEnv._clip_speed(
                action.cpu().numpy().reshape(B, k, 2), env.v_defender
            )
            res = env.step(a_action, d_action_np)

            act_buf[t] = action
            lp_buf[t] = logprob
            val_buf[t] = value
            rew_buf[t] = torch.as_tensor(res.defender_reward, device=device)

            _episode_tracker_step(
                tracker, res.defender_reward, res.just_done, res.outcome
            )
            if res.just_done.any():
                env.reset_idxs(res.just_done)
                if population is not None:
                    population.reshuffle(res.just_done)

            next_obs = torch.as_tensor(
                CentralizedDefenderNet.build_obs(env), device=device
            )
            next_done = torch.as_tensor(
                res.just_done.astype(np.float32), device=device
            )

        with torch.no_grad():
            _, _, next_value = net(next_obs)

        advantages, returns = _gae(
            rew_buf, val_buf, done_buf, next_value, next_done, cfg.gamma, cfg.gae_lambda
        )

        flat_obs = obs_buf.reshape(-1, obs_dim)
        flat_act = act_buf.reshape(-1, action_dim)
        flat_lp = lp_buf.reshape(-1)
        flat_adv = advantages.reshape(-1)
        flat_ret = returns.reshape(-1)
        flat_val = val_buf.reshape(-1)

        pl, vl, ent = _ppo_update(
            net, optimizer, flat_obs, flat_act, flat_lp, flat_adv, flat_ret, flat_val, cfg
        )

        total_steps += T * B
        elapsed = time.perf_counter() - t_start
        not_att = (
            float(np.mean([o != 1 for o in tracker["ep_outcomes"]]))
            if tracker["ep_outcomes"]
            else float("nan")
        )
        m = {
            "iter": it,
            "step": total_steps,
            "ep_return": float(np.mean(tracker["ep_rets"])) if tracker["ep_rets"] else float("nan"),
            "ep_length": float(np.mean(tracker["ep_lens"])) if tracker["ep_lens"] else float("nan"),
            "defender_succ": not_att,
            "policy_loss": pl,
            "value_loss": vl,
            "entropy": ent,
            "fps": total_steps / max(elapsed, 1e-6),
            "n_eps": len(tracker["ep_rets"]),
        }
        metrics.append(m)
        if log_every and (it % log_every == 0 or it == n_iters - 1):
            log_fn(
                f"  [cdef it={it:3d} step={total_steps:>7d}] "
                f"ret={m['ep_return']:+.3f} len={m['ep_length']:5.1f} "
                f"d_succ={m['defender_succ']:.3f} "
                f"pl={pl:+.4f} vl={vl:.4f} H={ent:+.3f} fps={m['fps']:.0f}"
            )
    return {"metrics": metrics}


# ============================================================================
# Sequential training: warmup attacker (with snapshots) → centralized defender
# vs population of attackers. No alternation, no oscillation.
# ============================================================================


def sequential_train(
    *,
    k: int,
    sigma: float,
    p: float,
    cfg: PPOConfig,
    n_envs: int,
    attacker_warmup_iters: int,
    central_defender_iters: int,
    snapshot_fractions: tuple[float, ...] = (0.4, 0.7, 1.0),
    attacker_hidden: int = 256,
    attacker_n_layers: int = 2,
    central_hidden: int = 256,
    central_n_layers: int = 3,
    seed: int = 0,
    env_kwargs: dict | None = None,
    save_dir: Path | str | None = None,
    log_fn=print,
) -> dict:
    """Two-phase training with population-based opponent during phase B.

    Phase A: train ``NeuralAttackerNet`` (bigger MLP) vs ``HeuristicDefender``
    for ``attacker_warmup_iters`` PPO iters. Save attacker state_dict at
    iterations ``attacker_warmup_iters * fraction`` for each fraction in
    ``snapshot_fractions``. The final iteration is always saved.

    Phase B: train ``CentralizedDefenderNet`` vs a :class:`PopulationAttacker`
    composed of (the heuristic attacker) + (every saved RL snapshot, both
    deterministic and stochastic variants). The defender therefore sees a
    diverse opponent population at every batch step and can't over-fit to a
    single attacker.

    Returns ``{attacker_net, central_defender_net, metrics, env_meta}``.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    env_kwargs = env_kwargs or {}

    log_fn(
        f"=== sequential_train  k={k} sigma={sigma} p={p}  "
        f"warmup={attacker_warmup_iters} cdef={central_defender_iters}  "
        f"att_arch={attacker_hidden}x{attacker_n_layers} "
        f"cdef_arch={central_hidden}x{central_n_layers}  B={n_envs} ==="
    )

    # ---- Phase A: attacker vs heuristic defender ----
    log_fn("[phase A] attacker (bigger MLP) vs heuristic defender")
    env_a = PursuitEvasionEnv(
        batch_size=n_envs, k=k, sigma=sigma, p=p, seed=seed, **env_kwargs
    )
    attacker_net = NeuralAttackerNet(
        env_a.attacker_obs_dim, hidden=attacker_hidden, n_layers=attacker_n_layers
    )
    heuristic_defender = HeuristicDefender(
        v_defender=env_a.v_defender, capture_radius=env_a.capture_radius
    )
    snap_iters = sorted(
        {
            max(0, int(round(attacker_warmup_iters * f)) - 1)
            for f in snapshot_fractions
        }
    )
    a_log = train_attacker(
        env_a,
        attacker_net,
        heuristic_defender,
        n_iters=attacker_warmup_iters,
        cfg=cfg,
        log_fn=log_fn,
        snapshot_iters=snap_iters,
    )
    snapshots = a_log["snapshots"]
    log_fn(f"[phase A] saved {len(snapshots)} attacker snapshots at iters {snap_iters}")

    # ---- Phase B: centralized defender vs attacker population ----
    log_fn("[phase B] centralized defender vs attacker population")
    env_b = PursuitEvasionEnv(
        batch_size=n_envs, k=k, sigma=sigma, p=p, seed=seed + 1, **env_kwargs
    )
    central_defender_net = CentralizedDefenderNet(
        k=k, hidden=central_hidden, n_layers=central_n_layers
    )

    # Build the population: heuristic + each snapshot (deterministic) + final
    # snapshot stochastic variant (extra noise diversity).
    pop_policies: list = [
        HeuristicAttacker(target=env_b.target, v_attacker=env_b.v_attacker)
    ]
    for sd in snapshots:
        snap_net = NeuralAttackerNet(
            env_b.attacker_obs_dim,
            hidden=attacker_hidden,
            n_layers=attacker_n_layers,
        )
        snap_net.load_state_dict(sd)
        snap_net.eval()
        pop_policies.append(
            NeuralAttacker(snap_net, env_b.v_attacker, deterministic=True)
        )
    # Stochastic version of the strongest (final) snapshot.
    if snapshots:
        final_net = NeuralAttackerNet(
            env_b.attacker_obs_dim,
            hidden=attacker_hidden,
            n_layers=attacker_n_layers,
        )
        final_net.load_state_dict(snapshots[-1])
        final_net.eval()
        pop_policies.append(
            NeuralAttacker(final_net, env_b.v_attacker, deterministic=False)
        )
    population = PopulationAttacker(pop_policies, env_b.B, rng_seed=seed + 2)
    log_fn(f"[phase B] population size = {len(pop_policies)}")

    # Pass a sentinel attacker_policy too (unused when population is set)
    cd_log = train_defender_centralized(
        env_b,
        central_defender_net,
        attacker_policy=pop_policies[0],
        n_iters=central_defender_iters,
        cfg=cfg,
        log_fn=log_fn,
        population=population,
    )

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(attacker_net.state_dict(), save_dir / "attacker.pt")
        torch.save(central_defender_net.state_dict(), save_dir / "central_defender.pt")
        for i, sd in enumerate(snapshots):
            torch.save(sd, save_dir / f"attacker_snapshot_{i}.pt")
        log_fn(f"saved checkpoints → {save_dir}")

    pop_desc = ["HeuristicAttacker"]
    for i in range(len(snapshots)):
        pop_desc.append(f"NeuralAttacker(snapshot={i}, deterministic=True)")
    if snapshots:
        pop_desc.append(
            f"NeuralAttacker(snapshot={len(snapshots)-1}, deterministic=False)"
        )
    env_steps_per_iter = n_envs * cfg.n_steps
    total_iters = attacker_warmup_iters + central_defender_iters
    schedule = {
        "method": "sequential_train",
        "k": k, "sigma": sigma, "p": p, "seed": seed,
        "n_envs": n_envs, "n_steps": cfg.n_steps,
        "attacker_warmup_iters": attacker_warmup_iters,
        "central_defender_iters": central_defender_iters,
        "snapshot_fractions": list(snapshot_fractions),
        "snapshot_iters": snap_iters,
        "snapshot_count": len(snapshots),
        "phases": [
            {
                "name": "phase_A_attacker_warmup",
                "trainee": "neural_attacker",
                "opponent": "HeuristicDefender",
                "n_iters": attacker_warmup_iters,
                "snapshot_iters": snap_iters,
            },
            {
                "name": "phase_B_central_defender",
                "trainee": "central_defender",
                "opponent": f"PopulationAttacker(size={len(pop_policies)})",
                "n_iters": central_defender_iters,
                "population_composition": pop_desc,
            },
        ],
        "attacker_arch": {"hidden": attacker_hidden, "n_layers": attacker_n_layers},
        "central_defender_arch": {"hidden": central_hidden, "n_layers": central_n_layers},
        "totals": {
            "attacker_iters": attacker_warmup_iters,
            "central_defender_iters": central_defender_iters,
            "env_steps_per_iter": env_steps_per_iter,
            "total_env_steps": total_iters * env_steps_per_iter,
        },
    }

    return {
        "attacker_net": attacker_net,
        "central_defender_net": central_defender_net,
        "snapshots": snapshots,
        "metrics": {"attacker": a_log["metrics"], "central_defender": cd_log["metrics"]},
        "env_meta": env_a.metadata(),
        "schedule": schedule,
    }
