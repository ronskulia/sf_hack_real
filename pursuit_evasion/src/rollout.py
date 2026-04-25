"""Vectorized policy evaluation with bootstrap confidence intervals."""
from __future__ import annotations

from typing import Protocol

import numpy as np

from .env import (
    OUTCOME_ATTACKER_WIN,
    OUTCOME_DEFENDER_CAPTURE,
    OUTCOME_DEFENDER_TIMEOUT,
    PursuitEvasionEnv,
)


class _ActsOnEnv(Protocol):
    def act(self, env: PursuitEvasionEnv) -> np.ndarray: ...


def _bootstrap_ci(samples: np.ndarray, n_boot: int = 2000, alpha: float = 0.05, rng: np.random.Generator | None = None) -> tuple[float, float]:
    """Percentile bootstrap CI on the mean of a 1-D 0/1 array."""
    if rng is None:
        rng = np.random.default_rng(0)
    n = len(samples)
    if n == 0:
        return float("nan"), float("nan")
    idx = rng.integers(0, n, size=(n_boot, n))
    means = samples[idx].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def evaluate(
    *,
    k: int,
    sigma: float,
    p: float,
    attacker_policy: _ActsOnEnv,
    defender_policy: _ActsOnEnv,
    n_episodes: int = 2000,
    seed: int = 0,
    env_kwargs: dict | None = None,
) -> dict:
    """Run ``n_episodes`` episodes in parallel and report the attacker
    success rate, mean episode length, and bootstrap 95% CI.
    """
    env_kwargs = env_kwargs or {}
    env = PursuitEvasionEnv(
        batch_size=n_episodes, k=k, sigma=sigma, p=p, seed=seed, **env_kwargs
    )
    for _ in range(env.max_steps + 5):
        a = attacker_policy.act(env)
        d = defender_policy.act(env)
        env.step(a, d)
        if bool(env.done.all()):
            break

    wins = (env.outcome == OUTCOME_ATTACKER_WIN).astype(np.float32)
    rate = float(wins.mean())
    rng = np.random.default_rng(seed)
    lo, hi = _bootstrap_ci(wins, rng=rng)
    return {
        "k": k,
        "sigma": sigma,
        "p": p,
        "n_episodes": n_episodes,
        "attacker_success_rate": rate,
        "ci_95": (lo, hi),
        "mean_episode_length": float(env.step_count.mean()),
        "n_attacker_wins": int(wins.sum()),
        "n_capture": int((env.outcome == OUTCOME_DEFENDER_CAPTURE).sum()),
        "n_timeout": int((env.outcome == OUTCOME_DEFENDER_TIMEOUT).sum()),
    }
