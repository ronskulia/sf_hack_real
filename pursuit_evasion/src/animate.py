"""Trajectory animation for pursuit-evasion episodes.

Runs a single episode (``batch_size = 1``) under the supplied policies, records
the per-step state, and renders a Matplotlib animation written to MP4 via
``imageio-ffmpeg``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

import matplotlib

matplotlib.use("Agg")  # headless: never open a window during rendering

import matplotlib.animation as manim  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .env import (
    OUTCOME_ATTACKER_WIN,
    OUTCOME_DEFENDER_CAPTURE,
    OUTCOME_DEFENDER_TIMEOUT,
    PursuitEvasionEnv,
)


class _ActsOnEnv(Protocol):
    """Anything that can produce an action from an env (heuristic or neural)."""

    def act(self, env: PursuitEvasionEnv) -> np.ndarray: ...


_OUTCOME_LABEL: dict[int, str] = {
    OUTCOME_ATTACKER_WIN: "attacker won",
    OUTCOME_DEFENDER_CAPTURE: "defender capture",
    OUTCOME_DEFENDER_TIMEOUT: "defender timeout",
}


def run_episode(
    env: PursuitEvasionEnv,
    attacker_policy: _ActsOnEnv,
    defender_policy: _ActsOnEnv,
    *,
    reset: bool = True,
) -> dict[str, Any]:
    """Roll out one episode in a ``B=1`` env and record the trajectory.

    Parameters
    ----------
    env : PursuitEvasionEnv
        Must have ``B == 1``.
    attacker_policy, defender_policy : has ``.act(env) -> ndarray``
        Heuristic policies callable on the env state.
    reset : bool
        If True, reset the env before running.

    Returns
    -------
    dict
        Keys: ``attacker_pos`` (T+1, 2), ``defender_pos`` (T+1, k, 2),
        ``outcome`` (int), ``steps`` (int).
    """
    if env.B != 1:
        raise ValueError(f"run_episode expects env.B == 1, got {env.B}")
    if reset:
        env.reset()

    a_traj: list[np.ndarray] = [env.attacker_pos[0].copy()]
    d_traj: list[np.ndarray] = [env.defender_pos[0].copy()]

    while not bool(env.done[0]):
        a_action = attacker_policy.act(env)
        d_action = defender_policy.act(env)
        env.step(a_action, d_action)
        a_traj.append(env.attacker_pos[0].copy())
        d_traj.append(env.defender_pos[0].copy())
        # Safety: never loop past max_steps.
        if int(env.step_count[0]) > env.max_steps + 1:
            break

    return {
        "attacker_pos": np.stack(a_traj, axis=0),
        "defender_pos": np.stack(d_traj, axis=0),
        "outcome": int(env.outcome[0]),
        "steps": int(env.step_count[0]),
    }


def save_animation(
    history: dict[str, Any],
    env_meta: dict[str, Any],
    out_path: str | Path,
    *,
    trail_len: int = 50,
    fps: int = 30,
    dpi: int = 100,
) -> Path:
    """Render a trajectory dict to an MP4 file.

    Parameters
    ----------
    history : dict
        Output of :func:`run_episode`.
    env_meta : dict
        Output of ``PursuitEvasionEnv.metadata()``.
    out_path : str or Path
        Destination ``.mp4`` path. Parent directory is created if missing.
    trail_len : int
        Number of past frames to draw as a fading trail.
    fps : int
        Animation frame rate.
    dpi : int
        Figure resolution.

    Returns
    -------
    Path
        The path that was written.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    a_traj: np.ndarray = history["attacker_pos"]  # (T+1, 2)
    d_traj: np.ndarray = history["defender_pos"]  # (T+1, k, 2)
    T = a_traj.shape[0]
    k = d_traj.shape[1]
    outcome_str = _OUTCOME_LABEL.get(int(history["outcome"]), "running")

    fig, ax = plt.subplots(figsize=(6, 6), dpi=dpi)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal")
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.grid(True, linestyle=":", alpha=0.3)
    title = (
        f"k={env_meta['k']}  σ={env_meta['sigma']:.2f}  p={env_meta['p']:.2f}  "
        f"| {outcome_str}  ({history['steps']} steps)"
    )
    ax.set_title(title, fontsize=11)

    target = env_meta["target_pos"]
    target_circle = plt.Circle(
        (target[0], target[1]),
        env_meta["target_radius"],
        facecolor="red",
        edgecolor="darkred",
        alpha=0.22,
        linewidth=1.5,
        zorder=1,
    )
    ax.add_patch(target_circle)
    ax.plot([target[0]], [target[1]], "*", color="darkred", markersize=14, zorder=5)

    capture_circles: list[plt.Circle] = []
    for _ in range(k):
        c = plt.Circle((0.0, 0.0), env_meta["capture_radius"], color="blue", alpha=0.12)
        ax.add_patch(c)
        capture_circles.append(c)

    (attacker_marker,) = ax.plot(
        [], [], "^", color="red", markersize=12, zorder=4, label="attacker"
    )
    (defender_markers,) = ax.plot(
        [], [], "o", color="blue", markersize=10, zorder=4, label=f"defenders ({k})"
    )
    (attacker_trail,) = ax.plot([], [], "-", color="red", alpha=0.45, linewidth=1.4)
    defender_trails = [
        ax.plot([], [], "-", color="blue", alpha=0.3, linewidth=1.0)[0]
        for _ in range(k)
    ]
    ax.legend(loc="upper right", fontsize=9)

    def _init() -> tuple:
        return (
            attacker_marker,
            defender_markers,
            attacker_trail,
            *defender_trails,
            *capture_circles,
        )

    def _update(frame: int) -> tuple:
        attacker_marker.set_data([a_traj[frame, 0]], [a_traj[frame, 1]])
        defender_markers.set_data(d_traj[frame, :, 0], d_traj[frame, :, 1])
        start = max(0, frame - trail_len)
        attacker_trail.set_data(a_traj[start : frame + 1, 0], a_traj[start : frame + 1, 1])
        for i, trail in enumerate(defender_trails):
            trail.set_data(
                d_traj[start : frame + 1, i, 0], d_traj[start : frame + 1, i, 1]
            )
        for i, circle in enumerate(capture_circles):
            circle.center = (float(d_traj[frame, i, 0]), float(d_traj[frame, i, 1]))
        return (
            attacker_marker,
            defender_markers,
            attacker_trail,
            *defender_trails,
            *capture_circles,
        )

    anim = manim.FuncAnimation(
        fig,
        _update,
        init_func=_init,
        frames=T,
        interval=1000.0 / fps,
        blit=False,
    )
    writer = manim.FFMpegWriter(fps=fps, bitrate=1800)
    anim.save(str(out_path), writer=writer, dpi=dpi)
    plt.close(fig)
    return out_path
