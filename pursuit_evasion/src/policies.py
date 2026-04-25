"""Heuristic and neural policies for the pursuit-evasion environment.

Heuristic policies operate directly on the env's state arrays (no observation
flattening) — this keeps them fast and easy to vectorize. Neural policies are
defined later (step 3 of the implementation plan) and consume the flat
observations exposed by the env.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .env import PursuitEvasionEnv


class HeuristicAttacker:
    """Greedy attacker that heads to the target and steers around defenders.

    The base direction is straight toward the target. If the closest defender
    is within ``danger_radius``, a tangential repulsion term is mixed in: the
    attacker veers perpendicular to the line connecting it to that defender,
    in whichever of the two perpendicular directions is more aligned with the
    target. The mixing weight ramps linearly from 0 (defender at danger
    radius) to ``urgency_gain`` (defender at the attacker's location).

    Parameters
    ----------
    target : sequence of float
        Goal location ``(x, y)``.
    v_attacker : float
        Max attacker speed (output velocity is rescaled to this magnitude).
    danger_radius : float
        Distance below which defenders trigger the avoidance term.
    urgency_gain : float
        Maximum weight on the tangential avoidance term.
    """

    def __init__(
        self,
        target: np.ndarray | tuple[float, float],
        v_attacker: float = 1.0,
        danger_radius: float = 0.15,
        urgency_gain: float = 1.5,
    ) -> None:
        self.target = np.asarray(target, dtype=np.float32)
        self.v = float(v_attacker)
        self.danger_radius = float(danger_radius)
        self.urgency_gain = float(urgency_gain)

    def act(self, env: PursuitEvasionEnv) -> np.ndarray:
        """Return desired attacker velocity for every env in the batch.

        Returns
        -------
        np.ndarray of shape ``(B, 2)`` with magnitude ``v_attacker``.
        """
        attacker_pos = env.attacker_pos  # (B, 2)
        defender_pos = env.defender_pos  # (B, k, 2)
        B = env.B

        to_target = self.target[None, :] - attacker_pos  # (B, 2)
        d_target = np.linalg.norm(to_target, axis=-1, keepdims=True)
        toward_target = to_target / np.maximum(d_target, 1e-8)  # (B, 2)
        toward_target_b = np.broadcast_to(toward_target[:, None, :], (B, env.k, 2))

        # Vector from attacker to every defender.
        to_def = defender_pos - attacker_pos[:, None, :]  # (B, k, 2)
        dist_def = np.linalg.norm(to_def, axis=-1, keepdims=True)  # (B, k, 1)
        to_def_unit = to_def / np.maximum(dist_def, 1e-8)

        # Two perpendiculars to the away-from-defender direction. Pick the one
        # better aligned with the target — that's the "side-step" direction.
        away_dir = -to_def_unit  # (B, k, 2)
        perp1 = np.stack([-away_dir[..., 1], away_dir[..., 0]], axis=-1)
        perp2 = np.stack([away_dir[..., 1], -away_dir[..., 0]], axis=-1)
        dot1 = (perp1 * toward_target_b).sum(axis=-1)
        dot2 = (perp2 * toward_target_b).sum(axis=-1)
        perp = np.where((dot1 >= dot2)[..., None], perp1, perp2)  # (B, k, 2)

        # Per-defender urgency: 0 outside danger radius, ramping to 1 at zero.
        dist_def_flat = dist_def.squeeze(-1)  # (B, k)
        urgency = np.clip(1.0 - dist_def_flat / self.danger_radius, 0.0, 1.0)

        # Sum tangential contributions across all defenders within danger radius.
        tangential = (urgency[..., None] * perp).sum(axis=1)  # (B, 2)
        steer = toward_target + self.urgency_gain * tangential
        steer_norm = np.linalg.norm(steer, axis=-1, keepdims=True)
        steer = steer / np.maximum(steer_norm, 1e-8) * self.v
        return steer.astype(np.float32)


class HeuristicDefender:
    """Apollonius-style intercept pursuit with angular spread.

    Each defender independently solves the constant-velocity intercept
    problem: given the attacker's current position and velocity, what's the
    earliest time at which the defender (running at full speed) can reach
    the same point? The intercept point is then offset by a defender-
    specific tangential nudge so the team fans out perpendicular to the
    attacker's heading instead of bunching on one point.

    Mathematically, intercept time ``t`` satisfies
    ``v_d^2 t^2 = || (p_a - p_d) + v_a t ||^2``,
    a quadratic in ``t`` with the smallest positive root selected. When no
    real positive root exists (e.g. attacker outrunning a slower defender
    on a divergent path) the defender falls back to pure pursuit toward the
    attacker's current position.

    Parameters
    ----------
    v_defender : float
        Max defender speed.
    capture_radius : float
        Used to size the tangential spread term.
    spread_gain : float
        Multiplier on the per-defender tangential offset.
    """

    def __init__(
        self,
        v_defender: float,
        capture_radius: float = 0.03,
        spread_gain: float = 1.0,
        spread_fade_far: float = 0.30,
        spread_fade_near: float = 0.05,
    ) -> None:
        self.v = float(v_defender)
        self.r_cap = float(capture_radius)
        self.spread_gain = float(spread_gain)
        self.spread_fade_far = float(spread_fade_far)
        self.spread_fade_near = float(spread_fade_near)

    def act(self, env: PursuitEvasionEnv) -> np.ndarray:
        """Return desired defender velocities for every env and defender.

        Returns
        -------
        np.ndarray of shape ``(B, k, 2)``.
        """
        attacker_pos = env.attacker_pos  # (B, 2)
        attacker_vel = env.attacker_vel  # (B, 2)
        defender_pos = env.defender_pos  # (B, k, 2)
        B, k = env.B, env.k

        # Solve quadratic per (env, defender) for intercept time t:
        # (v_d^2 - |v_a|^2) t^2 - 2 <p_a - p_d, v_a> t - |p_a - p_d|^2 = 0
        d_vec = attacker_pos[:, None, :] - defender_pos  # (B, k, 2)
        d_sq = (d_vec ** 2).sum(axis=-1)  # (B, k)
        v_a_sq = (attacker_vel ** 2).sum(axis=-1, keepdims=True)  # (B, 1)
        d_dot_va = (d_vec * attacker_vel[:, None, :]).sum(axis=-1)  # (B, k)

        a_coef = self.v ** 2 - v_a_sq  # (B, 1)
        a_coef = np.broadcast_to(a_coef, (B, k))
        b_coef = -2.0 * d_dot_va
        c_coef = -d_sq
        disc = b_coef ** 2 - 4.0 * a_coef * c_coef  # (B, k)

        # Use a numerically safe denominator — separately handle near-zero a.
        a_safe = np.where(np.abs(a_coef) < 1e-8, 1e-8, a_coef)
        sqrt_disc = np.sqrt(np.maximum(disc, 0.0))
        t1 = (-b_coef + sqrt_disc) / (2.0 * a_safe)
        t2 = (-b_coef - sqrt_disc) / (2.0 * a_safe)

        # Pick the smallest positive root. inf for 'no positive root'.
        t_pos1 = np.where(t1 > 0, t1, np.inf)
        t_pos2 = np.where(t2 > 0, t2, np.inf)
        t_chosen = np.minimum(t_pos1, t_pos2)
        valid = (disc >= 0) & np.isfinite(t_chosen)
        t_chosen = np.where(valid, t_chosen, 0.0)  # 0 ⇒ aim at current attacker pos

        intercept = attacker_pos[:, None, :] + attacker_vel[:, None, :] * t_chosen[..., None]
        intercept = np.where(valid[..., None], intercept, attacker_pos[:, None, :])

        # Per-defender tangential spread perpendicular to attacker's heading.
        v_a_norm = np.linalg.norm(attacker_vel, axis=-1, keepdims=True)  # (B, 1)
        # Fallback heading (when attacker is stationary): direction to target.
        fallback = env.target[None, :] - attacker_pos
        fallback_norm = np.linalg.norm(fallback, axis=-1, keepdims=True)
        fallback_unit = fallback / np.maximum(fallback_norm, 1e-8)
        moving = (v_a_norm > 1e-4)
        v_a_unit = np.where(
            moving,
            attacker_vel / np.maximum(v_a_norm, 1e-8),
            fallback_unit,
        )  # (B, 2)
        tangent = np.stack([-v_a_unit[:, 1], v_a_unit[:, 0]], axis=-1)  # (B, 2)
        # Defender i offset = (i - (k-1)/2) * spread_gain * capture_radius
        offsets = (np.arange(k, dtype=np.float32) - (k - 1) / 2.0) * (
            self.spread_gain * self.r_cap
        )  # (k,)
        spread_offsets = offsets[None, :, None] * tangent[:, None, :]  # (B, k, 2)

        # Fade the spread out as a defender closes in: full offset when far,
        # zero when within the inner fade radius. This prevents the team from
        # "missing" the attacker because every defender aims a bit off-target.
        dist_to_att = np.sqrt(d_sq)  # (B, k)
        denom = max(self.spread_fade_far - self.spread_fade_near, 1e-6)
        fade = np.clip((dist_to_att - self.spread_fade_near) / denom, 0.0, 1.0)
        spread_offsets = spread_offsets * fade[..., None]
        target_pt = intercept + spread_offsets

        direction = target_pt - defender_pos  # (B, k, 2)
        norm = np.linalg.norm(direction, axis=-1, keepdims=True)
        return (direction / np.maximum(norm, 1e-8) * self.v).astype(np.float32)


class HeuristicOrbitalDefender:
    """One closest-to-attacker chaser + spread-orbit goalies.

    Each defender ``i`` has a fixed home angle ``θ_i = 2π·i/k`` on a guard
    ring of radius ``orbit_radius`` centered on the target. Per step, the
    defender currently closest to the attacker is tagged as the chaser and
    runs an Apollonius intercept toward the attacker; every other defender
    moves toward its home angle on the orbit.

    Stateless — role assignment is per-step, so as the attacker moves the
    chaser role can swap between defenders. With ``k`` defenders evenly
    spread, there is always a defender within ~π/k radians of any threat
    direction, so the closest-defender role is geometrically reasonable.

    Parameters
    ----------
    v_defender : float
        Max defender speed.
    target_pos : sequence of float
    orbit_radius : float
        Radius of the guard ring around the target. Defaults to 0.20
        (between the env's inner=0.15 and outer=0.35 constraint radii).
    capture_radius : float
        Used only to tighten aim when very close to the attacker (no spread).
    """

    def __init__(
        self,
        v_defender: float,
        target_pos: tuple[float, float] = (0.5, 0.5),
        orbit_radius: float = 0.20,
        capture_radius: float = 0.03,
        n_chasers: int = 1,
    ) -> None:
        self.v = float(v_defender)
        self.target = np.asarray(target_pos, dtype=np.float32)
        self.orbit_radius = float(orbit_radius)
        self.r_cap = float(capture_radius)
        self.n_chasers = int(n_chasers)

    def act(self, env: PursuitEvasionEnv) -> np.ndarray:
        B, k = env.B, env.k
        attacker_pos = env.attacker_pos       # (B, 2)
        attacker_vel = env.attacker_vel       # (B, 2)
        defender_pos = env.defender_pos       # (B, k, 2)
        n_chasers = max(1, min(self.n_chasers, k))
        n_goalies = k - n_chasers

        # 1. Pick the ``n_chasers`` closest defenders per env as chasers.
        rel_to_att = attacker_pos[:, None, :] - defender_pos  # (B, k, 2)
        d_sq = (rel_to_att ** 2).sum(axis=-1)                  # (B, k)
        if n_chasers >= k:
            chaser_mask = np.ones((B, k), dtype=bool)
        else:
            chaser_idx_partial = np.argpartition(d_sq, n_chasers - 1, axis=-1)[:, :n_chasers]
            chaser_mask = np.zeros((B, k), dtype=bool)
            np.put_along_axis(chaser_mask, chaser_idx_partial, True, axis=-1)

        # 2. Apollonius intercept aim (used by the chaser slot).
        v_a_sq = (attacker_vel ** 2).sum(axis=-1, keepdims=True)  # (B, 1)
        d_dot_va = (rel_to_att * attacker_vel[:, None, :]).sum(axis=-1)  # (B, k)
        a_coef = np.broadcast_to(self.v ** 2 - v_a_sq, (B, k))
        b_coef = -2.0 * d_dot_va
        c_coef = -d_sq
        disc = b_coef ** 2 - 4.0 * a_coef * c_coef
        a_safe = np.where(np.abs(a_coef) < 1e-8, 1e-8, a_coef)
        sqrt_disc = np.sqrt(np.maximum(disc, 0.0))
        t1 = (-b_coef + sqrt_disc) / (2.0 * a_safe)
        t2 = (-b_coef - sqrt_disc) / (2.0 * a_safe)
        t_pos1 = np.where(t1 > 0, t1, np.inf)
        t_pos2 = np.where(t2 > 0, t2, np.inf)
        t_chosen = np.minimum(t_pos1, t_pos2)
        valid = (disc >= 0) & np.isfinite(t_chosen)
        t_chosen = np.where(valid, t_chosen, 0.0)
        intercept = attacker_pos[:, None, :] + attacker_vel[:, None, :] * t_chosen[..., None]
        intercept = np.where(valid[..., None], intercept, attacker_pos[:, None, :])  # (B, k, 2)

        # 3. Goalie home positions and assignment (skipped when no goalies).
        if n_goalies > 0:
            thetas = 2.0 * np.pi * np.arange(k, dtype=np.float32) / max(k, 1)  # (k,)
            ring_offsets = (
                np.stack([np.cos(thetas), np.sin(thetas)], axis=-1) * self.orbit_radius
            )  # (k, 2)
            home_pos_global = self.target[None, :] + ring_offsets  # (k, 2)

            # Greedy nearest-home assignment for non-chasers — each goalie goes
            # to the closest *unassigned* home so defenders don't cross paths.
            diff = (
                defender_pos[:, :, None, :]              # (B, k, 1, 2)
                - home_pos_global[None, None, :, :]      # (1, 1, k, 2)
            )
            cost = np.linalg.norm(diff, axis=-1).copy()  # (B, k, k)
            cost = np.where(chaser_mask[:, :, None], np.inf, cost)  # mask all chasers
            defender_home_idx = -np.ones((B, k), dtype=np.int64)
            batch_idx = np.arange(B)
            for _ in range(n_goalies):
                flat = cost.reshape(B, k * k)
                pick = np.argmin(flat, axis=-1)
                d_pick = pick // k
                h_pick = pick % k
                defender_home_idx[batch_idx, d_pick] = h_pick
                cost[batch_idx, d_pick, :] = np.inf
                cost[batch_idx, :, h_pick] = np.inf

            safe_home_idx = np.where(defender_home_idx < 0, 0, defender_home_idx)
            home_for_def = home_pos_global[safe_home_idx]    # (B, k, 2)
            aim = np.where(chaser_mask[..., None], intercept, home_for_def)
        else:
            # All defenders chase: aim = intercept everywhere.
            aim = intercept

        # 6. Unit-velocity at full speed toward aim, but park when within ~1mm
        #    of the aim point (avoids the divide-by-tiny wobble at home).
        direction = aim - defender_pos
        norm = np.linalg.norm(direction, axis=-1, keepdims=True)
        unit = direction / np.maximum(norm, 1e-8)
        moving = (norm > 1e-3).astype(np.float32)
        return (unit * self.v * moving).astype(np.float32)


class HeuristicDefenderTeam:
    """Team-aware heuristic defender: a few chasers + goalies on a guard ring.

    Each step, the closest ``n_chasers`` defenders to the attacker are tagged
    as chasers (they run Apollonius intercept). The remaining defenders are
    goalies — they hold position on a ring of radius ``r_guard`` around the
    target, each at its own current angle relative to the target.

    Role re-assignment is per-step, so if a chaser slips past the attacker the
    nearest goalie automatically becomes the new chaser. With ``n_chasers``
    small relative to ``k``, the team always keeps coverage near the goal.

    Parameters
    ----------
    v_defender : float
        Max defender speed.
    capture_radius : float
        Capture radius (used in some sizing decisions).
    target_pos : sequence of float
    n_chasers : int or None
        Number of defenders pursuing the attacker. ``None`` ⇒ ``max(1, k // 3)``.
    r_guard : float
        Goalie ring radius around target.
    """

    def __init__(
        self,
        v_defender: float,
        capture_radius: float = 0.03,
        target_pos: tuple[float, float] = (0.5, 0.5),
        n_chasers: int | None = None,
        r_guard: float = 0.10,
    ) -> None:
        self.v = float(v_defender)
        self.r_cap = float(capture_radius)
        self.target = np.asarray(target_pos, dtype=np.float32)
        self.n_chasers = n_chasers
        self.r_guard = float(r_guard)

    def act(self, env: PursuitEvasionEnv) -> np.ndarray:
        """Return defender velocities (B, k, 2)."""
        B, k = env.B, env.k
        attacker_pos = env.attacker_pos
        attacker_vel = env.attacker_vel
        defender_pos = env.defender_pos

        n_chasers = self.n_chasers if self.n_chasers is not None else max(1, k // 3)
        n_chasers = min(int(n_chasers), k)

        # ---- 1. Pick chaser_mask: closest n_chasers per env ----
        rel_to_att = attacker_pos[:, None, :] - defender_pos  # (B, k, 2)
        dist_to_att = np.linalg.norm(rel_to_att, axis=-1)  # (B, k)
        if n_chasers >= k:
            chaser_mask = np.ones((B, k), dtype=bool)
        else:
            chaser_idx = np.argpartition(dist_to_att, n_chasers - 1, axis=-1)[:, :n_chasers]
            chaser_mask = np.zeros((B, k), dtype=bool)
            np.put_along_axis(chaser_mask, chaser_idx, True, axis=-1)

        # ---- 2. Apollonius intercept (used by chasers) ----
        d_vec = attacker_pos[:, None, :] - defender_pos
        d_sq = (d_vec ** 2).sum(axis=-1)
        v_a_sq = (attacker_vel ** 2).sum(axis=-1, keepdims=True)
        d_dot_va = (d_vec * attacker_vel[:, None, :]).sum(axis=-1)
        a_coef = np.broadcast_to(self.v ** 2 - v_a_sq, (B, k))
        b_coef = -2.0 * d_dot_va
        c_coef = -d_sq
        disc = b_coef ** 2 - 4.0 * a_coef * c_coef
        a_safe = np.where(np.abs(a_coef) < 1e-8, 1e-8, a_coef)
        sqrt_disc = np.sqrt(np.maximum(disc, 0.0))
        t1 = (-b_coef + sqrt_disc) / (2.0 * a_safe)
        t2 = (-b_coef - sqrt_disc) / (2.0 * a_safe)
        t_pos1 = np.where(t1 > 0, t1, np.inf)
        t_pos2 = np.where(t2 > 0, t2, np.inf)
        t_chosen = np.minimum(t_pos1, t_pos2)
        valid = (disc >= 0) & np.isfinite(t_chosen)
        t_chosen = np.where(valid, t_chosen, 0.0)
        intercept = attacker_pos[:, None, :] + attacker_vel[:, None, :] * t_chosen[..., None]
        intercept = np.where(valid[..., None], intercept, attacker_pos[:, None, :])

        # ---- 3. Goalie home positions: r_guard ring at each defender's
        # current angle relative to target ----
        rel_to_target = defender_pos - self.target[None, None, :]
        angles = np.arctan2(rel_to_target[..., 1], rel_to_target[..., 0])
        home_pos = np.stack(
            [
                self.target[0] + self.r_guard * np.cos(angles),
                self.target[1] + self.r_guard * np.sin(angles),
            ],
            axis=-1,
        )  # (B, k, 2)

        # ---- 4. Combine: chaser → intercept, goalie → home ----
        target_pt = np.where(chaser_mask[..., None], intercept, home_pos)

        direction = target_pt - defender_pos
        norm = np.linalg.norm(direction, axis=-1, keepdims=True)
        return (direction / np.maximum(norm, 1e-8) * self.v).astype(np.float32)


# =============================================================================
# Neural policies (PyTorch)
# =============================================================================


def _layer_init(layer: nn.Linear, std: float = np.sqrt(2.0), bias_const: float = 0.0) -> nn.Linear:
    """CleanRL-style orthogonal init."""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class _MLPTrunk(nn.Module):
    """Configurable MLP trunk (tanh activations), shared between actor and critic."""

    def __init__(self, in_dim: int, hidden: int = 128, n_layers: int = 2) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            _layer_init(nn.Linear(in_dim, hidden)),
            nn.Tanh(),
        ]
        for _ in range(n_layers - 1):
            layers.append(_layer_init(nn.Linear(hidden, hidden)))
            layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class NeuralAttackerNet(nn.Module):
    """Actor-critic MLP for the attacker.

    Takes the flat attacker observation (4 + 4k) and outputs a 2-D Gaussian
    over velocity (mean, learned per-dim log-std) plus a scalar value.
    """

    def __init__(
        self, obs_dim: int, hidden: int = 256, n_layers: int = 2, action_dim: int = 2
    ) -> None:
        super().__init__()
        self.trunk = _MLPTrunk(obs_dim, hidden, n_layers)
        self.actor_mean = _layer_init(nn.Linear(hidden, action_dim), std=0.01)
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))
        self.critic = _layer_init(nn.Linear(hidden, 1), std=1.0)

    def forward(
        self, obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (action_mean, action_logstd, value). All on the same device."""
        h = self.trunk(obs)
        mean = self.actor_mean(h)
        logstd = self.actor_logstd.expand_as(mean)
        value = self.critic(h).squeeze(-1)
        return mean, logstd, value


class NeuralDefenderNet(nn.Module):
    """Shared-parameter actor-critic MLP for k defenders.

    Each defender's observation is unflattened into (own, attacker, teammates)
    parts. Teammate features are mean-pooled, giving permutation invariance
    and a consistent input size for any k. The pooled teammate vector is
    concatenated with the own/attacker features and fed to a standard
    actor-critic head.
    """

    def __init__(self, k: int, hidden: int = 128, action_dim: int = 2) -> None:
        super().__init__()
        self.k = int(k)
        # Each "agent" piece (own / attacker / teammate-pool) is 4 floats:
        # (pos_x, pos_y, vel_x, vel_y).
        in_dim = 4 + 4 + 4  # own + attacker + pooled teammates
        self.trunk = _MLPTrunk(in_dim, hidden)
        self.actor_mean = _layer_init(nn.Linear(hidden, action_dim), std=0.01)
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))
        self.critic = _layer_init(nn.Linear(hidden, 1), std=1.0)

    @staticmethod
    def split_obs(obs: torch.Tensor, k: int) -> torch.Tensor:
        """Flatten ``(..., 4 + 4k)`` defender obs into the trunk-ready
        ``(..., 12)`` tensor: own (4), attacker (4), mean-pooled teammates (4)."""
        own = obs[..., 0:4]
        att = obs[..., 4:8]
        # Remaining is teammate (pos+vel) flattened in groups of 4.
        # For k = 1: zero-length teammate slot ⇒ pool to zeros.
        if k > 1:
            tm = obs[..., 8:].reshape(*obs.shape[:-1], k - 1, 4)
            tm_pool = tm.mean(dim=-2)
        else:
            tm_pool = torch.zeros_like(own)
        return torch.cat([own, att, tm_pool], dim=-1)

    def forward(
        self, obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute (mean, logstd, value) from a flat per-defender observation
        of shape ``(..., 4 + 4k)``."""
        x = self.split_obs(obs, self.k)
        h = self.trunk(x)
        mean = self.actor_mean(h)
        logstd = self.actor_logstd.expand_as(mean)
        value = self.critic(h).squeeze(-1)
        return mean, logstd, value


class NeuralAttacker:
    """Inference-only wrapper: runs ``NeuralAttackerNet`` on env state.

    Uses the deterministic mean action by default (suitable for evaluation
    and for use as a frozen opponent). For PPO collection use the network
    directly with sampling.
    """

    def __init__(
        self, net: NeuralAttackerNet, v_attacker: float, *, deterministic: bool = True
    ) -> None:
        self.net = net
        self.v = float(v_attacker)
        self.deterministic = bool(deterministic)

    @torch.no_grad()
    def act(self, env: PursuitEvasionEnv) -> np.ndarray:
        a_obs, _ = env._get_obs()
        obs_t = torch.from_numpy(a_obs)
        mean, logstd, _ = self.net(obs_t)
        if self.deterministic:
            action = mean
        else:
            std = logstd.exp()
            action = torch.normal(mean, std)
        return PursuitEvasionEnv._clip_speed(action.numpy(), self.v)


class NeuralDefender:
    """Inference-only wrapper: runs ``NeuralDefenderNet`` on env state."""

    def __init__(
        self, net: NeuralDefenderNet, v_defender: float, *, deterministic: bool = True
    ) -> None:
        self.net = net
        self.v = float(v_defender)
        self.deterministic = bool(deterministic)

    @torch.no_grad()
    def act(self, env: PursuitEvasionEnv) -> np.ndarray:
        _, d_obs = env._get_obs()  # (B, k, 4+4k)
        B, k, D = d_obs.shape
        obs_t = torch.from_numpy(d_obs.reshape(B * k, D))
        mean, logstd, _ = self.net(obs_t)
        if self.deterministic:
            action = mean
        else:
            std = logstd.exp()
            action = torch.normal(mean, std)
        action = action.numpy().reshape(B, k, 2)
        return PursuitEvasionEnv._clip_speed(action, self.v)


# =============================================================================
# Centralized defender (one brain, sees full state, outputs all k velocities)
# =============================================================================


class CentralizedDefenderNet(nn.Module):
    """Centralized actor-critic for the defender team.

    A single network sees the entire game state (attacker pos+vel and every
    defender's pos+vel) and outputs a velocity *for every defender* —
    ``2 * k`` continuous outputs total. This breaks the parameter-sharing
    symmetry of :class:`NeuralDefenderNet` and lets the team learn explicit
    role differentiation (e.g. "you chase, I guard") since the network can
    decide what each defender does individually.

    Trade-off: the action space and observation are k-specific, so a network
    trained at k=4 cannot be re-used at k=6. We train one centralized
    network per ``k``.
    """

    def __init__(
        self,
        k: int,
        hidden: int = 256,
        n_layers: int = 3,
        action_dim_per_agent: int = 2,
    ) -> None:
        super().__init__()
        self.k = int(k)
        in_dim = 4 + 4 * self.k  # att_pos, att_vel, def_pos×k, def_vel×k
        out_dim = action_dim_per_agent * self.k
        self.trunk = _MLPTrunk(in_dim, hidden, n_layers)
        self.actor_mean = _layer_init(nn.Linear(hidden, out_dim), std=0.01)
        self.actor_logstd = nn.Parameter(torch.zeros(1, out_dim))
        self.critic = _layer_init(nn.Linear(hidden, 1), std=1.0)

    @staticmethod
    def build_obs(env: PursuitEvasionEnv) -> np.ndarray:
        """Build the centralized observation tensor of shape ``(B, 4 + 4k)``.

        The attacker position slot uses :meth:`PursuitEvasionEnv.observed_attacker_pos`
        — true position when noise is disabled (default), smoothly noised and
        distance-gated otherwise. Defenders therefore see a "fog of war"
        belief that sharpens as the team closes in.
        """
        B, k = env.B, env.k
        return np.concatenate(
            [
                env.observed_attacker_pos(),
                env.attacker_vel,
                env.defender_pos.reshape(B, 2 * k),
                env.defender_vel.reshape(B, 2 * k),
            ],
            axis=-1,
        ).astype(np.float32)

    @staticmethod
    def build_sorted_obs(env: PursuitEvasionEnv) -> tuple[np.ndarray, np.ndarray]:
        """Build centralized obs with defenders sorted by distance to attacker.

        Sorting uses the *true* attacker position so the slot ordering is
        unambiguous; only the attacker-position observation is noised.

        Returns
        -------
        sorted_obs : np.ndarray, shape (B, 4 + 4k), float32
        perm : np.ndarray, shape (B, k), int64
            ``perm[b, j]`` is the *original* index of the j-th closest defender
            in env b. Use :meth:`unsort_action` with this perm to convert
            network outputs (which are in sorted order) back to env order.
        """
        B, k = env.B, env.k
        rel = env.defender_pos - env.attacker_pos[:, None, :]  # (B, k, 2)
        d_sq = (rel ** 2).sum(axis=-1)  # (B, k)
        perm = np.argsort(d_sq, axis=-1, kind="stable").astype(np.int64)  # (B, k)
        sorted_def_pos = np.take_along_axis(env.defender_pos, perm[..., None], axis=1)
        sorted_def_vel = np.take_along_axis(env.defender_vel, perm[..., None], axis=1)
        sorted_obs = np.concatenate(
            [
                env.observed_attacker_pos(),
                env.attacker_vel,
                sorted_def_pos.reshape(B, 2 * k),
                sorted_def_vel.reshape(B, 2 * k),
            ],
            axis=-1,
        ).astype(np.float32)
        return sorted_obs, perm

    @staticmethod
    def unsort_action(action_sorted: np.ndarray, perm: np.ndarray) -> np.ndarray:
        """Map (B, k, 2) actions in sorted order back to env (original) order.

        ``original[b, i] = sorted[b, inv_perm[b, i]]`` where
        ``inv_perm = argsort(perm)``.
        """
        inv_perm = np.argsort(perm, axis=-1)  # (B, k)
        return np.take_along_axis(action_sorted, inv_perm[..., None], axis=1)

    def forward(
        self, obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (action_mean (B, 2k), action_logstd (B, 2k), value (B,))."""
        h = self.trunk(obs)
        mean = self.actor_mean(h)
        logstd = self.actor_logstd.expand_as(mean)
        value = self.critic(h).squeeze(-1)
        return mean, logstd, value


class CentralizedDefender:
    """Inference-only wrapper for :class:`CentralizedDefenderNet`.

    If ``sort_by_distance`` is True, defenders are reordered by distance to
    the attacker before each forward pass (closest = slot 0). The network
    output is then unsorted back to env-defender order before being issued.
    This makes the policy effectively permutation-invariant — slot 0 always
    means "the closest defender" — and avoids the failure mode where the
    centralized net binds a fixed slot to "the chaser".
    """

    def __init__(
        self,
        net: CentralizedDefenderNet,
        v_defender: float,
        *,
        deterministic: bool = True,
        sort_by_distance: bool = False,
    ) -> None:
        self.net = net
        self.v = float(v_defender)
        self.deterministic = bool(deterministic)
        self.sort_by_distance = bool(sort_by_distance)
        self.k = net.k

    @torch.no_grad()
    def act(self, env: PursuitEvasionEnv) -> np.ndarray:
        if self.sort_by_distance:
            obs, perm = CentralizedDefenderNet.build_sorted_obs(env)
        else:
            obs = CentralizedDefenderNet.build_obs(env)
            perm = None
        obs_t = torch.from_numpy(obs)
        mean, logstd, _ = self.net(obs_t)
        if self.deterministic:
            action = mean
        else:
            std = logstd.exp()
            action = torch.normal(mean, std)
        action_np = action.numpy().reshape(env.B, env.k, 2)
        if perm is not None:
            action_np = CentralizedDefenderNet.unsort_action(action_np, perm)
        return PursuitEvasionEnv._clip_speed(action_np, self.v)


# =============================================================================
# Opponent population (per-env mixture of policies)
# =============================================================================


class PopulationAttacker:
    """Mixture-of-attackers wrapper.

    On construction, each of the ``B`` environments in the batch is randomly
    assigned to one of the supplied attacker policies. ``act`` runs every
    policy in the pool and dispatches each env's action from its assigned
    policy. Call :meth:`reshuffle` (typically when envs reset) to
    re-randomize assignments so the defender keeps seeing a diverse mix.

    Used during centralized-defender training to prevent the defender from
    over-fitting to a single attacker's specific gait — instead, it sees
    weak / medium / strong RL attacker snapshots plus a heuristic baseline.
    """

    def __init__(
        self,
        policies: list,
        batch_size: int,
        rng_seed: int = 0,
    ) -> None:
        if not policies:
            raise ValueError("PopulationAttacker needs at least one policy.")
        self.policies = list(policies)
        self.B = int(batch_size)
        self.rng = np.random.default_rng(rng_seed)
        self.assignment: np.ndarray = self.rng.integers(
            0, len(self.policies), size=self.B
        )

    def reshuffle(self, idxs: np.ndarray | None = None) -> None:
        """Re-randomize policy assignment for the given envs (or all if None)."""
        if idxs is None:
            self.assignment = self.rng.integers(
                0, len(self.policies), size=self.B
            )
            return
        idxs = np.asarray(idxs)
        if idxs.dtype == bool:
            idxs = np.where(idxs)[0]
        if len(idxs) == 0:
            return
        self.assignment[idxs] = self.rng.integers(
            0, len(self.policies), size=len(idxs)
        )

    def act(self, env: PursuitEvasionEnv) -> np.ndarray:
        # Compute every policy's action for the whole batch, then gather.
        per_policy = np.stack([p.act(env) for p in self.policies], axis=0)  # (n_pol, B, 2)
        b_idx = np.arange(self.B)
        return per_policy[self.assignment, b_idx, :].astype(np.float32)
