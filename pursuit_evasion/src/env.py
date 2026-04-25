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
        Per-step probability that the attacker dies when at least one
        defender is within capture radius.
    dt : float
        Simulation time step.
    max_steps : int
        Step limit; if reached, defenders win by timeout.
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
    defender_sensor_epsilon : float
        Maximum per-coordinate attacker-position error visible to defenders.
        A value of ``0`` means defenders observe the true attacker position.
        Otherwise, defenders receive a smooth value-noise offset whose amplitude
        increases with the distance from the attacker to the closest defender.
    defender_sensor_distance_scale : float
        Closest-defender distance at which defender perception reaches the full
        ``defender_sensor_epsilon``. The error ramps down to zero at the capture
        radius.
    defender_sensor_smooth_time : float
        Approximate seconds between new smooth-noise targets for defender
        perception. Larger values make uncertainty drift more slowly.
    shaping : float, optional
        Backward-compatible coefficient applied to both teams when
        ``attacker_shaping`` / ``defender_shaping`` are omitted.
    attacker_shaping, defender_shaping : float, optional
        Coefficients for dense heuristic rewards. The attacker receives
        target-progress reward and danger/blocking penalties. The defender
        receives chase-progress, pressure, blocking, spread, and timeout
        rewards.
    danger_radius : float, optional
        Radius used for attacker danger penalty and defender pressure reward.
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
        capture_radius: float = 0.03,
        target_radius: float = 0.10,
        target_pos: tuple[float, float] = (0.5, 0.5),
        v_attacker: float = 1.0,
        inner_radius: float = 0.15,
        outer_radius: float = 0.35,
        defender_sensor_epsilon: float = 0.0,
        defender_sensor_distance_scale: float = 0.5,
        defender_sensor_smooth_time: float = 0.5,
        shaping: float | None = None,
        attacker_shaping: float | None = None,
        defender_shaping: float | None = None,
        danger_radius: float | None = None,
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
        self.capture_radius: float = float(capture_radius)
        self.target_radius: float = float(target_radius)
        self.target: np.ndarray = np.asarray(target_pos, dtype=np.float32)
        self.v_attacker: float = float(v_attacker)
        self.v_defender: float = self.sigma * self.v_attacker
        self.inner_radius: float = float(inner_radius)
        self.outer_radius: float = float(outer_radius)
        self.defender_sensor_epsilon: float = float(defender_sensor_epsilon)
        if self.defender_sensor_epsilon < 0.0:
            raise ValueError(
                "defender_sensor_epsilon must be >= 0, "
                f"got {self.defender_sensor_epsilon}"
            )
        self.defender_sensor_distance_scale: float = float(
            defender_sensor_distance_scale
        )
        if self.defender_sensor_distance_scale <= 0.0:
            raise ValueError(
                "defender_sensor_distance_scale must be > 0, "
                f"got {self.defender_sensor_distance_scale}"
            )
        self.defender_sensor_distance_scale = max(
            self.defender_sensor_distance_scale,
            self.capture_radius + 1e-6,
        )
        self.defender_sensor_smooth_time: float = float(defender_sensor_smooth_time)
        if self.defender_sensor_smooth_time <= 0.0:
            raise ValueError(
                "defender_sensor_smooth_time must be > 0, "
                f"got {self.defender_sensor_smooth_time}"
            )
        base_shaping = 0.01 if shaping is None else float(shaping)
        self.attacker_shaping: float = (
            base_shaping if attacker_shaping is None else float(attacker_shaping)
        )
        self.defender_shaping: float = (
            base_shaping if defender_shaping is None else float(defender_shaping)
        )
        # Legacy name kept for older helper code and saved metadata readers.
        self.shaping: float = base_shaping
        default_danger_radius = max(0.15, self.capture_radius * 4.0)
        self.danger_radius: float = max(
            float(danger_radius) if danger_radius is not None else default_danger_radius,
            self.capture_radius + 1e-6,
        )
        self.block_width: float = max(0.08, self.capture_radius * 3.0)
        self.cluster_scale: float = max(0.10, self.capture_radius * 4.0)

        self.rng: np.random.Generator = np.random.default_rng(seed)

        # State buffers
        self.attacker_pos: np.ndarray = np.zeros((self.B, 2), dtype=np.float32)
        self.attacker_vel: np.ndarray = np.zeros((self.B, 2), dtype=np.float32)
        self.defender_pos: np.ndarray = np.zeros((self.B, self.k, 2), dtype=np.float32)
        self.defender_vel: np.ndarray = np.zeros((self.B, self.k, 2), dtype=np.float32)
        self.defender_perceived_attacker_pos: np.ndarray = np.zeros(
            (self.B, 2), dtype=np.float32
        )
        self.defender_attacker_pos_error: np.ndarray = np.zeros(
            (self.B, 2), dtype=np.float32
        )
        self._defender_sensor_noise_start: np.ndarray = np.zeros(
            (self.B, 2), dtype=np.float32
        )
        self._defender_sensor_noise_target: np.ndarray = np.zeros(
            (self.B, 2), dtype=np.float32
        )
        self._defender_sensor_noise_phase: np.ndarray = np.zeros(
            self.B, dtype=np.float32
        )
        self._defender_sensor_noise_value: np.ndarray = np.zeros(
            (self.B, 2), dtype=np.float32
        )
        self.step_count: np.ndarray = np.zeros(self.B, dtype=np.int32)
        self.done: np.ndarray = np.zeros(self.B, dtype=bool)
        self.outcome: np.ndarray = np.zeros(self.B, dtype=np.int32)

        # Pre-compute the teammate index table once: shape (k, k-1)
        if self.k > 1:
            self._teammate_idx: np.ndarray = np.stack(
                [np.delete(np.arange(self.k), i) for i in range(self.k)],
                axis=0,
            )
            self._pair_i, self._pair_j = np.triu_indices(self.k, k=1)
        else:
            self._teammate_idx = np.zeros((1, 0), dtype=np.int64)
            self._pair_i = np.zeros(0, dtype=np.int64)
            self._pair_j = np.zeros(0, dtype=np.int64)

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

    def _sample_defender_sensor_noise(self, n: int) -> np.ndarray:
        """Sample bounded knots for the defenders' smooth perception noise."""
        return self.rng.uniform(-1.0, 1.0, size=(n, 2)).astype(np.float32)

    def _update_defender_sensor_noise_value(
        self, idx: np.ndarray | None = None
    ) -> None:
        """Evaluate smooth value noise at the current per-env phase."""
        if idx is None:
            phase = self._defender_sensor_noise_phase
            start = self._defender_sensor_noise_start
            target = self._defender_sensor_noise_target
            smooth = phase * phase * (3.0 - 2.0 * phase)
            self._defender_sensor_noise_value = (
                start + (target - start) * smooth[:, None]
            ).astype(np.float32)
            return

        phase = self._defender_sensor_noise_phase[idx]
        start = self._defender_sensor_noise_start[idx]
        target = self._defender_sensor_noise_target[idx]
        smooth = phase * phase * (3.0 - 2.0 * phase)
        self._defender_sensor_noise_value[idx] = (
            start + (target - start) * smooth[:, None]
        ).astype(np.float32)

    def _reset_defender_sensor_noise(self, mask: np.ndarray | None = None) -> None:
        """Initialize smooth perception-noise state for new episodes."""
        if self.defender_sensor_epsilon == 0.0:
            if mask is None:
                self._defender_sensor_noise_start.fill(0.0)
                self._defender_sensor_noise_target.fill(0.0)
                self._defender_sensor_noise_phase.fill(0.0)
                self._defender_sensor_noise_value.fill(0.0)
            elif mask.any():
                self._defender_sensor_noise_start[mask] = 0.0
                self._defender_sensor_noise_target[mask] = 0.0
                self._defender_sensor_noise_phase[mask] = 0.0
                self._defender_sensor_noise_value[mask] = 0.0
            return

        if mask is None:
            self._defender_sensor_noise_start = self._sample_defender_sensor_noise(
                self.B
            )
            self._defender_sensor_noise_target = self._sample_defender_sensor_noise(
                self.B
            )
            self._defender_sensor_noise_phase = self.rng.uniform(
                0.0, 1.0, size=self.B
            ).astype(np.float32)
            self._update_defender_sensor_noise_value()
            return

        if not mask.any():
            return
        idx = np.where(mask)[0]
        n = len(idx)
        self._defender_sensor_noise_start[idx] = self._sample_defender_sensor_noise(n)
        self._defender_sensor_noise_target[idx] = self._sample_defender_sensor_noise(n)
        self._defender_sensor_noise_phase[idx] = self.rng.uniform(
            0.0, 1.0, size=n
        ).astype(np.float32)
        self._update_defender_sensor_noise_value(idx)

    def _advance_defender_sensor_noise(self, mask: np.ndarray | None = None) -> None:
        """Advance the defenders' smooth perception-noise state."""
        if self.defender_sensor_epsilon == 0.0:
            return

        if mask is None:
            idx = np.arange(self.B)
        else:
            if not mask.any():
                return
            idx = np.where(mask)[0]

        phase = (
            self._defender_sensor_noise_phase[idx]
            + self.dt / self.defender_sensor_smooth_time
        )
        rollover = phase >= 1.0
        if rollover.any():
            roll_idx = idx[rollover]
            self._defender_sensor_noise_start[roll_idx] = (
                self._defender_sensor_noise_target[roll_idx]
            )
            self._defender_sensor_noise_target[roll_idx] = (
                self._sample_defender_sensor_noise(len(roll_idx))
            )
            phase[rollover] = np.mod(phase[rollover], 1.0)

        self._defender_sensor_noise_phase[idx] = phase.astype(np.float32)
        self._update_defender_sensor_noise_value(idx)

    def _defender_sensor_amplitude(self, idx: np.ndarray | None = None) -> np.ndarray:
        """Return per-env perception error amplitude from closest-defender range."""
        if idx is None:
            attacker_pos = self.attacker_pos
            defender_pos = self.defender_pos
        else:
            attacker_pos = self.attacker_pos[idx]
            defender_pos = self.defender_pos[idx]

        rel = defender_pos - attacker_pos[:, None, :]
        closest_dist = np.linalg.norm(rel, axis=-1).min(axis=-1)
        denom = max(self.defender_sensor_distance_scale - self.capture_radius, 1e-6)
        distance_gain = np.clip(
            (closest_dist - self.capture_radius) / denom,
            0.0,
            1.0,
        )
        return (self.defender_sensor_epsilon * distance_gain).astype(np.float32)

    def _refresh_defender_perception(
        self,
        mask: np.ndarray | None = None,
        *,
        advance_noise: bool = False,
    ) -> None:
        """Update defenders' smooth noisy view of the attacker position."""
        if mask is None:
            if advance_noise:
                self._advance_defender_sensor_noise()
            pos = self.attacker_pos
            if self.defender_sensor_epsilon == 0.0:
                perceived = pos.copy()
            else:
                err = (
                    self._defender_sensor_noise_value
                    * self._defender_sensor_amplitude()[:, None]
                )
                perceived = np.clip(pos + err, 0.0, 1.0).astype(np.float32)
            self.defender_perceived_attacker_pos = perceived
            self.defender_attacker_pos_error = (
                self.defender_perceived_attacker_pos - self.attacker_pos
            ).astype(np.float32)
            return

        if not mask.any():
            return
        if advance_noise:
            self._advance_defender_sensor_noise(mask)
        idx = np.where(mask)[0]
        pos = self.attacker_pos[idx]
        if self.defender_sensor_epsilon == 0.0:
            perceived = pos.copy()
        else:
            err = (
                self._defender_sensor_noise_value[idx]
                * self._defender_sensor_amplitude(idx)[:, None]
            )
            perceived = np.clip(pos + err, 0.0, 1.0).astype(np.float32)
        self.defender_perceived_attacker_pos[idx] = perceived
        self.defender_attacker_pos_error[idx] = (
            self.defender_perceived_attacker_pos[idx] - self.attacker_pos[idx]
        ).astype(np.float32)

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
        self._reset_defender_sensor_noise()
        self._refresh_defender_perception()
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
        self._reset_defender_sensor_noise(mask)
        self._refresh_defender_perception(mask)

    # ---------------------------------------------------------------- dynamics
    @staticmethod
    def _clip_speed(v: np.ndarray, vmax: float) -> np.ndarray:
        """Clip 2D velocity vectors so ``||v|| <= vmax`` along the last axis."""
        norm = np.linalg.norm(v, axis=-1, keepdims=True)
        scale = np.minimum(1.0, vmax / np.maximum(norm, 1e-8))
        return v * scale

    def _block_score(self) -> np.ndarray:
        """Return how well any defender blocks the attacker-target segment."""
        to_target = self.target[None, :] - self.attacker_pos  # (B, 2)
        seg_len2 = np.sum(to_target * to_target, axis=-1, keepdims=True) + 1e-8
        rel_def = self.defender_pos - self.attacker_pos[:, None, :]  # (B, k, 2)
        proj = np.sum(rel_def * to_target[:, None, :], axis=-1) / seg_len2
        between = (proj > 0.0) & (proj < 1.0)
        closest = self.attacker_pos[:, None, :] + proj[..., None] * to_target[:, None, :]
        line_dist = np.linalg.norm(self.defender_pos - closest, axis=-1)
        per_defender = np.exp(-((line_dist / self.block_width) ** 2)) * between
        return per_defender.max(axis=-1).astype(np.float32)

    def _cluster_penalty(self) -> np.ndarray:
        """Return a penalty when defenders collapse onto the same location."""
        if self.k <= 1:
            return np.zeros(self.B, dtype=np.float32)
        pair_delta = (
            self.defender_pos[:, self._pair_i, :]
            - self.defender_pos[:, self._pair_j, :]
        )
        pair_dist = np.linalg.norm(pair_delta, axis=-1)
        return np.exp(-(pair_dist / self.cluster_scale)).mean(axis=-1).astype(np.float32)

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
        active_f = active.astype(np.float32)

        prev_target_dist = np.linalg.norm(
            self.attacker_pos - self.target[None, :], axis=-1
        )
        prev_rel = self.defender_pos - self.attacker_pos[:, None, :]
        prev_min_dist = np.linalg.norm(prev_rel, axis=-1).min(axis=-1)

        a_vel = self._clip_speed(
            np.asarray(attacker_action, dtype=np.float32), self.v_attacker
        )  # (B, 2)
        d_vel = self._clip_speed(
            np.asarray(defender_action, dtype=np.float32), self.v_defender
        )  # (B, k, 2)

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
        in_capture = active & (min_dist < self.capture_radius)
        rolls = self.rng.uniform(0.0, 1.0, size=self.B)
        killed = in_capture & (rolls < self.p) & (~reached)

        timeout = (
            active
            & (self.step_count >= self.max_steps)
            & (~reached)
            & (~killed)
        )

        new_done = reached | killed | timeout

        # ---- rewards ----
        target_progress = np.clip(
            (prev_target_dist - target_dist) / max(self.v_attacker * self.dt, 1e-6),
            -1.0,
            1.0,
        )
        close_progress = np.clip(
            (prev_min_dist - min_dist)
            / max((self.v_attacker + self.v_defender) * self.dt, 1e-6),
            -1.0,
            1.0,
        )
        danger_span = max(self.danger_radius - self.capture_radius, 1e-6)
        nearest_danger = np.clip(
            (self.danger_radius - min_dist) / danger_span,
            0.0,
            1.0,
        )
        defender_pressure = np.clip(
            (self.danger_radius - dists) / danger_span,
            0.0,
            1.0,
        )
        capture_pressure = (
            defender_pressure.max(axis=-1) + 0.25 * defender_pressure.mean(axis=-1)
        )
        block_score = self._block_score()
        cluster_penalty = self._cluster_penalty()

        a_reward = np.zeros(self.B, dtype=np.float32)
        a_reward = a_reward + reached.astype(np.float32) * 1.0
        a_reward = a_reward + killed.astype(np.float32) * (-1.0)
        a_reward = a_reward + timeout.astype(np.float32) * (-1.0)
        a_reward = a_reward + self.attacker_shaping * active_f * (
            target_progress - 0.35 * nearest_danger - 0.25 * block_score
        )

        d_reward = np.zeros(self.B, dtype=np.float32)
        d_reward = d_reward + reached.astype(np.float32) * (-1.0)
        d_reward = d_reward + killed.astype(np.float32) * 1.0
        d_reward = d_reward + timeout.astype(np.float32) * 1.0
        d_reward = d_reward + self.defender_shaping * active_f * (
            0.75 * close_progress
            + 0.15 * capture_pressure
            + 0.15 * block_score
            - 0.10 * cluster_penalty
        )
        a_reward = a_reward.astype(np.float32)
        d_reward = d_reward.astype(np.float32)

        # ---- record outcomes & done flags ----
        self.outcome = np.where(reached, OUTCOME_ATTACKER_WIN, self.outcome)
        self.outcome = np.where(killed, OUTCOME_DEFENDER_CAPTURE, self.outcome)
        self.outcome = np.where(timeout, OUTCOME_DEFENDER_TIMEOUT, self.outcome)
        self.done = self.done | new_done
        self._refresh_defender_perception(active, advance_noise=True)

        a_obs, d_obs = self._get_obs()
        return StepResult(
            attacker_obs=a_obs,
            defender_obs=d_obs,
            attacker_reward=a_reward,
            defender_reward=d_reward,
            just_done=new_done,
            outcome=self.outcome.copy(),
        )

    # ----------------------------------------------------------- observations
    def _get_obs(self) -> tuple[np.ndarray, np.ndarray]:
        """Build flat attacker and per-defender observation tensors."""
        B, k = self.B, self.k
        target_b = np.broadcast_to(self.target[None, :], (B, 2))

        # Attacker observations stay noise-free; only defenders receive the
        # perceived attacker position below.
        attacker_obs = np.concatenate(
            [
                self.attacker_pos,
                target_b,
                self.defender_pos.reshape(B, 2 * k),
                self.defender_vel.reshape(B, 2 * k),
            ],
            axis=-1,
        ).astype(np.float32)  # (B, 4 + 4k)

        own_pos = self.defender_pos  # (B, k, 2)
        own_vel = self.defender_vel  # (B, k, 2)
        att_pos = np.broadcast_to(
            self.defender_perceived_attacker_pos[:, None, :], (B, k, 2)
        )
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
            "capture_radius": self.capture_radius,
            "target_radius": self.target_radius,
            "target_pos": self.target.tolist(),
            "v_attacker": self.v_attacker,
            "v_defender": self.v_defender,
            "defender_sensor_epsilon": self.defender_sensor_epsilon,
            "defender_sensor_distance_scale": self.defender_sensor_distance_scale,
            "defender_sensor_smooth_time": self.defender_sensor_smooth_time,
            "attacker_shaping": self.attacker_shaping,
            "defender_shaping": self.defender_shaping,
            "danger_radius": self.danger_radius,
        }
