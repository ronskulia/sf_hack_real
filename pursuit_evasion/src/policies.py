"""Heuristic and neural policies for the pursuit-evasion environment.

Heuristic policies operate directly on the env's state arrays (no observation
flattening) — this keeps them fast and easy to vectorize. Neural policies are
defined later (step 3 of the implementation plan) and consume the flat
observations exposed by the env.
"""
from __future__ import annotations

import numpy as np

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
