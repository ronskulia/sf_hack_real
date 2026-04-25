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
        capture_radius: float = 0.03,
        target_radius: float = 0.10,
        target_pos: tuple[float, float] = (0.5, 0.5),
        v_attacker: float = 1.0,
        inner_radius: float = 0.15,
        outer_radius: float = 0.35,
        shaping: float = 0.01,
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
        self.shaping: float = float(shaping)

        self.rng: np.random.Generator = np.random.default_rng(seed)

        # State buffers
        self.attacker_pos: np.ndarray = np.zeros((self.B, 2), dtype=np.float32)
        self.attacker_vel: np.ndarray = np.zeros((self.B, 2), dtype=np.float32)
        self.defender_pos: np.ndarray = np.zeros((self.B, self.k, 2), dtype=np.float32)
        self.defender_vel: np.ndarray = np.zeros((self.B, self.k, 2), dtype=np.float32)
        self.step_count: np.ndarray = np.zeros(self.B, dtype=np.int32)
        self.done: np.ndarray = np.zeros(self.B, dtype=bool)
        self.outcome: np.ndarray = np.zeros(self.B, dtype=np.int32)

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
        active_f = active.astype(np.float32)
        a_reward = np.zeros(self.B, dtype=np.float32)
        a_reward = a_reward + reached.astype(np.float32) * 1.0
        a_reward = a_reward + killed.astype(np.float32) * (-1.0)
        a_reward = a_reward - self.shaping * target_dist * active_f

        d_reward = np.zeros(self.B, dtype=np.float32)
        d_reward = d_reward + reached.astype(np.float32) * (-1.0)
        d_reward = d_reward + killed.astype(np.float32) * 1.0
        d_reward = d_reward - self.shaping * min_dist * active_f

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

    # ----------------------------------------------------------- observations
    def _get_obs(self) -> tuple[np.ndarray, np.ndarray]:
        """Build flat attacker and per-defender observation tensors."""
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

        own_pos = self.defender_pos  # (B, k, 2)
        own_vel = self.defender_vel  # (B, k, 2)
        att_pos = np.broadcast_to(self.attacker_pos[:, None, :], (B, k, 2))
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
        }
