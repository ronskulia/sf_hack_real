"""Vectorized 2D pursuit-evasion environment.

A batch of ``B`` independent environments runs in parallel. Each holds one
attacker and ``k`` defenders inside the unit square ``[0, 1]^2``. The attacker
tries to reach a fixed target; defenders try to intercept it.

All operations are vectorized over the batch dimension — there are no Python
``for``-loops over the batch axis. ``k`` is fixed per env instance.

Coordinate convention: position (x, y) with x = col, y = row, both in [0, 1].
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np

# Outcome codes (0 = episode still running)
OUTCOME_NONE: int = 0
OUTCOME_ATTACKER_WIN: int = 1
OUTCOME_DEFENDER_CAPTURE: int = 2
OUTCOME_DEFENDER_TIMEOUT: int = 3


class StepResult(NamedTuple):
    """Output of a single ``env.step`` call.

    Attributes
    ----------
    attacker_obs : np.ndarray
        Shape ``(B, attacker_obs_dim)``.
    defender_obs : np.ndarray
        Shape ``(B, k, defender_obs_dim)``.
    attacker_reward : np.ndarray
        Shape ``(B,)``. Reward delivered to the attacker on this step.
    defender_reward : np.ndarray
        Shape ``(B,)``. Shared team reward for the defenders.
    just_done : np.ndarray
        Shape ``(B,)`` bool. ``True`` for envs that terminated *on this step*.
    outcome : np.ndarray
        Shape ``(B,)`` int. One of the ``OUTCOME_*`` constants. ``0`` for
        still-running envs.
    """

    attacker_obs: np.ndarray
    defender_obs: np.ndarray
    attacker_reward: np.ndarray
    defender_reward: np.ndarray
    just_done: np.ndarray
    outcome: np.ndarray


class PursuitEvasionEnv:
    """Vectorized 2D pursuit-evasion environment.

    Parameters
    ----------
    batch_size : int
        Number of independent environments simulated in parallel.
    k : int
        Number of defender drones (>= 1).
    sigma : float
        Defender max speed expressed as a fraction of attacker max speed.
    p : float
        Per-shot kill probability for each defender within capture radius.
        After a missed shot, that defender enters a stun cooldown of
        ``stun_steps`` steps during which it keeps drifting at its current
        velocity and cannot fire again.
    dt : float
        Simulation time step.
    max_steps : int
        Step limit; if reached, defenders win by timeout.
    stun_steps : int
        Number of steps a defender is stunned after a failed kill attempt.
        During stun the policy's action is ignored and the defender keeps
        moving with its current velocity. Set to 0 to disable.
    capture_radius : float
        Distance below which a defender threatens the attacker.
    target_radius : float
        Distance below which the attacker wins by reaching the target.
    target_pos : tuple[float, float]
        World coordinates of the goal.
    v_attacker : float
        Attacker max speed.
    inner_radius, outer_radius : float
        Annulus around the target where defenders spawn.
    shaping : float
        Coefficient applied to per-step distance shaping reward.
    seed : int
        RNG seed.
    """

    def __init__(
        self,
        batch_size: int,
        k: int,
        sigma: float = 0.7,
        p: float = 0.8,
        *,
        dt: float = 0.02,
        max_steps: int = 500,
        stun_steps: int = 15,
        capture_radius: float = 0.03,
        target_radius: float = 0.10,
        target_pos: tuple[float, float] = (0.5, 0.5),
        v_attacker: float = 1.0,
        inner_radius: float = 0.15,
        outer_radius: float = 0.35,
        shaping: float = 0.01,
        c_progress_target: float = 0.0,
        c_chase: float = 0.0,
        c_block: float = 0.0,
        c_block_threshold: float = 0.05,
        c_pressure: float = 0.0,
        c_pressure_radius: float = 0.15,
        c_cluster: float = 0.0,
        c_cluster_scale: float = 0.10,
        c_timeout: float = 0.0,
        c_line_of_sight: float = 0.0,
        c_los_threshold: float = 0.05,
        noise_sigma: float = 0.0,
        noise_tau_steps: float = 30.0,
        noise_r_clean: float = 0.05,
        noise_r_full: float = 0.30,
        seed: int = 0,
    ) -> None:
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")

        self.B: int = int(batch_size)
        self.k: int = int(k)
        self.sigma: float = float(sigma)
        self.p: float = float(p)
        self.dt: float = float(dt)
        self.max_steps: int = int(max_steps)
        self.stun_steps: int = int(stun_steps)
        self.capture_radius: float = float(capture_radius)
        self.target_radius: float = float(target_radius)
        self.target: np.ndarray = np.asarray(target_pos, dtype=np.float32)
        self.v_attacker: float = float(v_attacker)
        self.v_defender: float = self.sigma * self.v_attacker
        self.inner_radius: float = float(inner_radius)
        self.outer_radius: float = float(outer_radius)
        self.shaping: float = float(shaping)
        # Defender shaping coefs (all default 0 — opt-in per call site).
        self.c_progress_target: float = float(c_progress_target)
        self.c_chase: float = float(c_chase)
        self.c_block: float = float(c_block)
        self.c_block_threshold: float = float(c_block_threshold)
        self.c_pressure: float = float(c_pressure)
        self.c_pressure_radius: float = float(c_pressure_radius)
        self.c_cluster: float = float(c_cluster)
        self.c_cluster_scale: float = float(c_cluster_scale)
        self.c_timeout: float = float(c_timeout)
        self.c_line_of_sight: float = float(c_line_of_sight)
        self.c_los_threshold: float = float(c_los_threshold)
        # Defender-observation uncertainty: smooth OU offset on the attacker's
        # position, gated by min defender→attacker distance (close ⇒ no noise,
        # far ⇒ full magnitude). Stationary std is ``noise_sigma``; correlation
        # time is ``noise_tau_steps`` (in env steps). All defaults zero ⇒
        # bit-exact identical to old behaviour.
        self.noise_sigma: float = float(noise_sigma)
        self.noise_tau_steps: float = float(noise_tau_steps)
        self.noise_r_clean: float = float(noise_r_clean)
        self.noise_r_full: float = float(noise_r_full)
        # OU update so stationary std == noise_sigma exactly:
        #   x_{t+1} = decay * x_t + sqrt(1 - decay^2) * sigma * randn()
        if self.noise_sigma > 0.0:
            self._noise_decay: float = float(np.exp(-1.0 / max(self.noise_tau_steps, 1e-6)))
            self._noise_innov: float = float(
                np.sqrt(max(1.0 - self._noise_decay**2, 0.0)) * self.noise_sigma
            )
        else:
            self._noise_decay = 1.0
            self._noise_innov = 0.0

        self.rng: np.random.Generator = np.random.default_rng(seed)

        # State buffers
        self.attacker_pos: np.ndarray = np.zeros((self.B, 2), dtype=np.float32)
        self.attacker_vel: np.ndarray = np.zeros((self.B, 2), dtype=np.float32)
        self.defender_pos: np.ndarray = np.zeros((self.B, self.k, 2), dtype=np.float32)
        self.defender_vel: np.ndarray = np.zeros((self.B, self.k, 2), dtype=np.float32)
        self.step_count: np.ndarray = np.zeros(self.B, dtype=np.int32)
        self.done: np.ndarray = np.zeros(self.B, dtype=bool)
        self.outcome: np.ndarray = np.zeros(self.B, dtype=np.int32)
        # Per-defender stun cooldown. > 0 ⇒ defender is drifting and can't fire.
        self.cooldown: np.ndarray = np.zeros((self.B, self.k), dtype=np.int32)
        # Cached "previous step" distances for delta-shaping rewards.
        self.prev_target_dist: np.ndarray = np.zeros(self.B, dtype=np.float32)
        self.prev_min_dist: np.ndarray = np.zeros(self.B, dtype=np.float32)
        # OU offset on what the *defenders* observe of the attacker's position.
        # Re-sampled from the stationary distribution on reset; evolves each
        # step. Multiplied by a smoothstep gate that depends on min_dist so
        # uncertainty fades to 0 as defenders close in.
        self.attacker_obs_offset: np.ndarray = np.zeros((self.B, 2), dtype=np.float32)

        # Pre-compute the teammate index table once: shape (k, k-1)
        if self.k > 1:
            self._teammate_idx: np.ndarray = np.stack(
                [np.delete(np.arange(self.k), i) for i in range(self.k)],
                axis=0,
            )
        else:
            self._teammate_idx = np.zeros((1, 0), dtype=np.int64)

        self.reset()

    # ------------------------------------------------------------------ shapes
    @property
    def attacker_obs_dim(self) -> int:
        """Length of the flat attacker observation vector.

        Layout: own pos (2), target (2), all defender pos (2k), all defender vel (2k).
        """
        return 4 + 4 * self.k

    @property
    def defender_obs_dim(self) -> int:
        """Per-defender observation length.

        Layout: own pos/vel (4), attacker pos/vel (4), teammate pos+vel (4(k-1)).
        For ``k = 1`` the teammate slots collapse to zero width.
        """
        return 4 + 4 * self.k

    # ----------------------------------------------------------------- spawning
    def _spawn_attacker(self, n: int) -> np.ndarray:
        """Sample ``n`` attacker positions uniformly on the unit-square boundary."""
        side = self.rng.integers(0, 4, size=n)
        t = self.rng.uniform(0.0, 1.0, size=n).astype(np.float32)
        pos = np.zeros((n, 2), dtype=np.float32)
        m = side == 0  # bottom (y = 0)
        pos[m, 0] = t[m]
        pos[m, 1] = 0.0
        m = side == 1  # right (x = 1)
        pos[m, 0] = 1.0
        pos[m, 1] = t[m]
        m = side == 2  # top (y = 1)
        pos[m, 0] = t[m]
        pos[m, 1] = 1.0
        m = side == 3  # left (x = 0)
        pos[m, 0] = 0.0
        pos[m, 1] = t[m]
        return pos

    def _spawn_defenders(self, n: int) -> np.ndarray:
        """Sample ``n × k`` defender positions area-uniformly in the annulus."""
        angle = self.rng.uniform(0.0, 2.0 * np.pi, size=(n, self.k)).astype(np.float32)
        r2 = self.rng.uniform(
            self.inner_radius**2,
            self.outer_radius**2,
            size=(n, self.k),
        ).astype(np.float32)
        r = np.sqrt(r2)
        out = np.empty((n, self.k, 2), dtype=np.float32)
        out[..., 0] = self.target[0] + r * np.cos(angle)
        out[..., 1] = self.target[1] + r * np.sin(angle)
        return out

    # -------------------------------------------------------------------- reset
    def reset(self) -> tuple[np.ndarray, np.ndarray]:
        """Reset every environment in the batch.

        Returns
        -------
        attacker_obs, defender_obs : np.ndarray
            Shapes ``(B, attacker_obs_dim)`` and ``(B, k, defender_obs_dim)``.
        """
        self.attacker_pos = self._spawn_attacker(self.B)
        self.defender_pos = self._spawn_defenders(self.B)
        self.attacker_vel.fill(0.0)
        self.defender_vel.fill(0.0)
        self.step_count.fill(0)
        self.done.fill(False)
        self.outcome.fill(OUTCOME_NONE)
        self.cooldown.fill(0)
        self.prev_target_dist = np.linalg.norm(
            self.attacker_pos - self.target[None, :], axis=-1
        ).astype(np.float32)
        rel0 = self.defender_pos - self.attacker_pos[:, None, :]
        self.prev_min_dist = np.linalg.norm(rel0, axis=-1).min(axis=-1).astype(np.float32)
        if self.noise_sigma > 0.0:
            self.attacker_obs_offset = (
                self.rng.standard_normal((self.B, 2)).astype(np.float32) * self.noise_sigma
            )
        else:
            self.attacker_obs_offset.fill(0.0)
        return self._get_obs()

    def reset_idxs(self, mask: np.ndarray) -> None:
        """Reset the environments where ``mask`` is ``True``.

        Parameters
        ----------
        mask : np.ndarray
            Boolean array of shape ``(B,)``.
        """
        n = int(mask.sum())
        if n == 0:
            return
        self.attacker_pos[mask] = self._spawn_attacker(n)
        self.defender_pos[mask] = self._spawn_defenders(n)
        self.attacker_vel[mask] = 0.0
        self.defender_vel[mask] = 0.0
        self.step_count[mask] = 0
        self.done[mask] = False
        self.outcome[mask] = OUTCOME_NONE
        self.cooldown[mask] = 0
        # Re-init delta-shaping caches for the reset envs.
        new_target_dist = np.linalg.norm(
            self.attacker_pos[mask] - self.target[None, :], axis=-1
        ).astype(np.float32)
        self.prev_target_dist[mask] = new_target_dist
        rel_new = self.defender_pos[mask] - self.attacker_pos[mask][:, None, :]
        self.prev_min_dist[mask] = np.linalg.norm(rel_new, axis=-1).min(axis=-1).astype(np.float32)
        if self.noise_sigma > 0.0:
            self.attacker_obs_offset[mask] = (
                self.rng.standard_normal((n, 2)).astype(np.float32) * self.noise_sigma
            )
        else:
            self.attacker_obs_offset[mask] = 0.0

    # ---------------------------------------------------------------- dynamics
    @staticmethod
    def _clip_speed(v: np.ndarray, vmax: float) -> np.ndarray:
        """Clip 2D velocity vectors so ``||v|| <= vmax`` along the last axis."""
        norm = np.linalg.norm(v, axis=-1, keepdims=True)
        scale = np.minimum(1.0, vmax / np.maximum(norm, 1e-8))
        return v * scale

    def step(
        self,
        attacker_action: np.ndarray,
        defender_action: np.ndarray,
    ) -> StepResult:
        """Advance every environment by one ``dt``.

        Parameters
        ----------
        attacker_action : np.ndarray
            Desired attacker velocity, shape ``(B, 2)``. Magnitude is clipped to
            ``v_attacker`` along the last axis.
        defender_action : np.ndarray
            Desired defender velocities, shape ``(B, k, 2)``. Magnitudes clipped
            to ``v_defender``.

        Notes
        -----
        Already-finished environments (``self.done == True``) are frozen: their
        positions, velocities, step counts and outcomes do not change. They
        receive zero reward.
        """
        active = ~self.done  # (B,)
        active_a = active.astype(np.float32)[:, None]  # (B, 1) for broadcasting
        active_d = active.astype(np.float32)[:, None, None]  # (B, 1, 1)

        a_vel = self._clip_speed(
            np.asarray(attacker_action, dtype=np.float32), self.v_attacker
        )  # (B, 2)
        d_action = np.asarray(defender_action, dtype=np.float32)
        # Stunned defenders ignore the policy and keep drifting at current vel.
        stunned = self.cooldown > 0  # (B, k)
        if stunned.any():
            d_action = np.where(stunned[..., None], self.defender_vel, d_action)
        d_vel = self._clip_speed(d_action, self.v_defender)  # (B, k, 2)

        # Integrate, then clamp to world bounds. Frozen envs keep their state.
        new_a_pos = np.clip(self.attacker_pos + a_vel * self.dt, 0.0, 1.0)
        new_d_pos = np.clip(self.defender_pos + d_vel * self.dt, 0.0, 1.0)

        self.attacker_pos = active_a * new_a_pos + (1.0 - active_a) * self.attacker_pos
        self.attacker_vel = active_a * a_vel + (1.0 - active_a) * self.attacker_vel
        self.defender_pos = active_d * new_d_pos + (1.0 - active_d) * self.defender_pos
        self.defender_vel = active_d * d_vel + (1.0 - active_d) * self.defender_vel

        self.step_count = self.step_count + active.astype(np.int32)

        # ---- termination tests (only fire on active envs) ----
        target_dist = np.linalg.norm(
            self.attacker_pos - self.target[None, :], axis=-1
        )  # (B,)
        reached = active & (target_dist < self.target_radius)

        rel = self.defender_pos - self.attacker_pos[:, None, :]  # (B, k, 2)
        dists = np.linalg.norm(rel, axis=-1)  # (B, k)
        min_dist = dists.min(axis=-1)  # (B,)

        # Evolve OU offset for active envs (frozen envs keep their offset).
        if self.noise_sigma > 0.0:
            innov = self.rng.standard_normal((self.B, 2)).astype(np.float32) * self._noise_innov
            new_offset = self._noise_decay * self.attacker_obs_offset + innov
            active_off = active.astype(np.float32)[:, None]
            self.attacker_obs_offset = (
                active_off * new_offset + (1.0 - active_off) * self.attacker_obs_offset
            )
        # Per-defender capture & shot. Only defenders that are (a) in the
        # capture radius and (b) not currently stunned actually fire.
        defender_in_capture = active[:, None] & (dists < self.capture_radius)  # (B, k)
        can_fire = defender_in_capture & (self.cooldown == 0)  # (B, k)
        defender_rolls = self.rng.uniform(0.0, 1.0, size=(self.B, self.k))
        defender_hit = can_fire & (defender_rolls < self.p)  # (B, k)
        killed = active & defender_hit.any(axis=-1) & (~reached)
        # Decrement existing cooldowns for active envs, then stun any defender
        # that fired this step and missed.
        self.cooldown = np.maximum(
            self.cooldown - active[:, None].astype(np.int32), 0
        )
        defender_missed = can_fire & ~defender_hit
        if defender_missed.any():
            self.cooldown = np.where(defender_missed, self.stun_steps, self.cooldown)

        timeout = (
            active
            & (self.step_count >= self.max_steps)
            & (~reached)
            & (~killed)
        )

        new_done = reached | killed | timeout

        # ---- rewards ----
        active_f = active.astype(np.float32)
        a_reward = np.zeros(self.B, dtype=np.float32)
        a_reward = a_reward + reached.astype(np.float32) * 1.0
        a_reward = a_reward + killed.astype(np.float32) * (-1.0)
        a_reward = a_reward - self.shaping * target_dist * active_f

        d_reward = np.zeros(self.B, dtype=np.float32)
        d_reward = d_reward + reached.astype(np.float32) * (-1.0)
        d_reward = d_reward + killed.astype(np.float32) * 1.0
        d_reward = d_reward - self.shaping * min_dist * active_f

        # ---- additional defender shaping rewards (opt-in) ----
        # All terms are masked by ``active_f`` so frozen envs contribute zero.
        # 1) Progress-to-target: reward when attacker moves AWAY from target.
        if self.c_progress_target != 0.0:
            d_reward = d_reward + (
                self.c_progress_target
                * (target_dist - self.prev_target_dist)
                * active_f
            )
        # 2) Chase-progress: reward closing distance to attacker (closest defender).
        if self.c_chase != 0.0:
            d_reward = d_reward + (
                self.c_chase * (self.prev_min_dist - min_dist) * active_f
            )
        # 3+7) Blocking & line-of-sight: project each defender onto the
        # attacker→target line; reward "good blockers" (close to line, between
        # attacker and target) and penalise non-productive defenders sitting
        # behind the attacker on the same line.
        if self.c_block != 0.0 or self.c_line_of_sight != 0.0:
            at_vec = self.target[None, :] - self.attacker_pos                 # (B, 2)
            at_norm = np.linalg.norm(at_vec, axis=-1, keepdims=True)
            at_unit = at_vec / np.maximum(at_norm, 1e-8)                       # (B, 2)
            d_to_a = self.defender_pos - self.attacker_pos[:, None, :]         # (B, k, 2)
            t_proj = (d_to_a * at_unit[:, None, :]).sum(axis=-1)               # (B, k)
            perp = d_to_a - t_proj[..., None] * at_unit[:, None, :]
            perp_dist = np.linalg.norm(perp, axis=-1)                          # (B, k)
            on_segment = (t_proj > 0.0) & (t_proj < at_norm)                  # (B, k)
            if self.c_block != 0.0:
                alignment = (
                    np.maximum(0.0, 1.0 - perp_dist / max(self.c_block_threshold, 1e-6))
                    * on_segment.astype(np.float32)
                )
                d_reward = d_reward + (
                    self.c_block * alignment.sum(axis=-1) * active_f
                )
            if self.c_line_of_sight != 0.0:
                # Penalty for defenders on the line BEHIND the attacker (t_proj < 0)
                # or far past it — i.e. they're not contributing to blocking.
                behind = (t_proj < 0.0) | (t_proj > at_norm)
                on_line_useless = (perp_dist < self.c_los_threshold) & behind
                d_reward = d_reward - (
                    self.c_line_of_sight * on_line_useless.sum(axis=-1).astype(np.float32) * active_f
                )
        # 4) Capture-zone pressure: reward sustained presence inside danger ring.
        if self.c_pressure != 0.0:
            pressure = np.maximum(0.0, self.c_pressure_radius - dists).mean(axis=-1)  # (B,)
            d_reward = d_reward + self.c_pressure * pressure * active_f
        # 5) Anti-cluster: penalise pairs of defenders that bunch together.
        if self.c_cluster != 0.0 and self.k > 1:
            diff = self.defender_pos[:, :, None, :] - self.defender_pos[:, None, :, :]
            pair_dist = np.linalg.norm(diff, axis=-1)                          # (B, k, k)
            decay = np.exp(-pair_dist / max(self.c_cluster_scale, 1e-6))
            mask = ~np.eye(self.k, dtype=bool)
            cluster_score = (decay * mask[None, :, :]).sum(axis=(-1, -2)) / max(
                self.k * (self.k - 1), 1
            )
            d_reward = d_reward - self.c_cluster * cluster_score * active_f
        # 6) Timeout bonus.
        if self.c_timeout != 0.0:
            d_reward = d_reward + self.c_timeout * timeout.astype(np.float32)

        # Update delta-shaping caches for the next step (no-op on frozen envs).
        self.prev_target_dist = np.where(active, target_dist, self.prev_target_dist).astype(np.float32)
        self.prev_min_dist = np.where(active, min_dist, self.prev_min_dist).astype(np.float32)

        # ---- record outcomes & done flags ----
        self.outcome = np.where(reached, OUTCOME_ATTACKER_WIN, self.outcome)
        self.outcome = np.where(killed, OUTCOME_DEFENDER_CAPTURE, self.outcome)
        self.outcome = np.where(timeout, OUTCOME_DEFENDER_TIMEOUT, self.outcome)
        self.done = self.done | new_done

        a_obs, d_obs = self._get_obs()
        return StepResult(
            attacker_obs=a_obs,
            defender_obs=d_obs,
            attacker_reward=a_reward,
            defender_reward=d_reward,
            just_done=new_done,
            outcome=self.outcome.copy(),
        )

    # ------------------------------------------------------ observation noise
    def _noise_gate(self, min_dist: np.ndarray) -> np.ndarray:
        """Smoothstep ramp 0→1 over [r_clean, r_full] of min defender distance.

        Returns
        -------
        np.ndarray
            Shape ``(B,)``, float32. ``0`` ⇒ a defender is within ``r_clean``
            (no noise); ``1`` ⇒ closest defender is at or beyond ``r_full``
            (full ``noise_sigma`` magnitude). Smoothstep in between for a
            continuously differentiable transition.
        """
        if self.noise_r_full <= self.noise_r_clean:
            return (min_dist >= self.noise_r_full).astype(np.float32)
        denom = self.noise_r_full - self.noise_r_clean
        t = np.clip((min_dist - self.noise_r_clean) / denom, 0.0, 1.0)
        return (t * t * (3.0 - 2.0 * t)).astype(np.float32)

    def observed_attacker_pos(self) -> np.ndarray:
        """Defender-visible attacker position: ``attacker_pos + offset · gate``.

        ``noise_sigma == 0`` ⇒ returns the true position bit-exact.
        """
        if self.noise_sigma <= 0.0:
            return self.attacker_pos
        rel = self.defender_pos - self.attacker_pos[:, None, :]
        min_dist = np.linalg.norm(rel, axis=-1).min(axis=-1)  # (B,)
        gate = self._noise_gate(min_dist)  # (B,)
        return (self.attacker_pos + self.attacker_obs_offset * gate[:, None]).astype(np.float32)

    # ----------------------------------------------------------- observations
    def _get_obs(self) -> tuple[np.ndarray, np.ndarray]:
        """Build flat attacker and per-defender observation tensors.

        The attacker's own observation always uses the *true* attacker
        position. The per-defender observation uses :meth:`observed_attacker_pos`
        so defenders see the (smoothly noised, distance-gated) belief of where
        the attacker is.
        """
        B, k = self.B, self.k
        target_b = np.broadcast_to(self.target[None, :], (B, 2))

        attacker_obs = np.concatenate(
            [
                self.attacker_pos,
                target_b,
                self.defender_pos.reshape(B, 2 * k),
                self.defender_vel.reshape(B, 2 * k),
            ],
            axis=-1,
        ).astype(np.float32)  # (B, 4 + 4k)

        observed_a_pos = self.observed_attacker_pos()  # (B, 2) — noisy for defenders
        own_pos = self.defender_pos  # (B, k, 2)
        own_vel = self.defender_vel  # (B, k, 2)
        att_pos = np.broadcast_to(observed_a_pos[:, None, :], (B, k, 2))
        att_vel = np.broadcast_to(self.attacker_vel[:, None, :], (B, k, 2))

        if k > 1:
            tm_pos = self.defender_pos[:, self._teammate_idx, :]  # (B, k, k-1, 2)
            tm_vel = self.defender_vel[:, self._teammate_idx, :]
            tm_pos_flat = tm_pos.reshape(B, k, 2 * (k - 1))
            tm_vel_flat = tm_vel.reshape(B, k, 2 * (k - 1))
        else:
            tm_pos_flat = np.zeros((B, k, 0), dtype=np.float32)
            tm_vel_flat = np.zeros((B, k, 0), dtype=np.float32)

        defender_obs = np.concatenate(
            [own_pos, own_vel, att_pos, att_vel, tm_pos_flat, tm_vel_flat],
            axis=-1,
        ).astype(np.float32)  # (B, k, 4 + 4k)
        return attacker_obs, defender_obs

    # ------------------------------------------------------------------ utility
    def metadata(self) -> dict:
        """Return a small dict describing the env (for logging / animation)."""
        return {
            "k": self.k,
            "sigma": self.sigma,
            "p": self.p,
            "dt": self.dt,
            "max_steps": self.max_steps,
            "stun_steps": self.stun_steps,
            "capture_radius": self.capture_radius,
            "target_radius": self.target_radius,
            "target_pos": self.target.tolist(),
            "v_attacker": self.v_attacker,
            "v_defender": self.v_defender,
            "noise_sigma": self.noise_sigma,
            "noise_tau_steps": self.noise_tau_steps,
            "noise_r_clean": self.noise_r_clean,
            "noise_r_full": self.noise_r_full,
        }
