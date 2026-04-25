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

    def _gate_now() -> float:
        if env.noise_sigma <= 0.0:
            return 0.0
        rel = env.defender_pos[0] - env.attacker_pos[0][None, :]
        md = float(np.linalg.norm(rel, axis=-1).min())
        return float(env._noise_gate(np.asarray([md]))[0])

    a_traj: list[np.ndarray] = [env.attacker_pos[0].copy()]
    d_traj: list[np.ndarray] = [env.defender_pos[0].copy()]
    cd_traj: list[np.ndarray] = [env.cooldown[0].copy()]
    obs_a_traj: list[np.ndarray] = [env.observed_attacker_pos()[0].copy()]
    gate_traj: list[float] = [_gate_now()]

    while not bool(env.done[0]):
        a_action = attacker_policy.act(env)
        d_action = defender_policy.act(env)
        env.step(a_action, d_action)
        a_traj.append(env.attacker_pos[0].copy())
        d_traj.append(env.defender_pos[0].copy())
        cd_traj.append(env.cooldown[0].copy())
        obs_a_traj.append(env.observed_attacker_pos()[0].copy())
        gate_traj.append(_gate_now())
        # Safety: never loop past max_steps.
        if int(env.step_count[0]) > env.max_steps + 1:
            break

    return {
        "attacker_pos": np.stack(a_traj, axis=0),
        "defender_pos": np.stack(d_traj, axis=0),
        "cooldown": np.stack(cd_traj, axis=0),
        "obs_attacker_pos": np.stack(obs_a_traj, axis=0),
        "noise_gate": np.asarray(gate_traj, dtype=np.float32),
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
    cd_traj: np.ndarray = history.get(
        "cooldown", np.zeros((a_traj.shape[0], d_traj.shape[1]), dtype=np.int32)
    )  # (T+1, k); zeros for legacy histories
    obs_a_traj: np.ndarray = history.get("obs_attacker_pos", a_traj)  # (T+1, 2)
    gate_traj: np.ndarray = history.get(
        "noise_gate", np.zeros(a_traj.shape[0], dtype=np.float32)
    )
    noise_sigma: float = float(env_meta.get("noise_sigma", 0.0))
    has_uncertainty: bool = noise_sigma > 0.0
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
    if has_uncertainty:
        title += (
            f"  · obs_noise σ={noise_sigma:.2f}"
            f"  r=[{env_meta.get('noise_r_clean', 0):.2f},"
            f"{env_meta.get('noise_r_full', 0):.2f}]"
        )
    ax.set_title(title, fontsize=10)

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
    # Per-defender scatter so we can recolor individual defenders when stunned.
    defender_scatter = ax.scatter(
        np.zeros(k),
        np.zeros(k),
        s=100,
        c=["blue"] * k,
        edgecolors="black",
        linewidths=0.6,
        zorder=4,
        label=f"defenders ({k})",
    )
    (attacker_trail,) = ax.plot([], [], "-", color="red", alpha=0.45, linewidth=1.4)
    defender_trails = [
        ax.plot([], [], "-", color="blue", alpha=0.3, linewidth=1.0)[0]
        for _ in range(k)
    ]

    # ----- Defender's-eye-view of the attacker (only drawn if noise enabled) -----
    if has_uncertainty:
        # Translucent uncertainty halo centered on the perceived position.
        # Radius scales with the gate (close ⇒ small / no halo).
        uncertainty_halo = plt.Circle(
            (0.0, 0.0), 1e-6,
            facecolor="orange", edgecolor="darkorange",
            alpha=0.18, linewidth=1.0, linestyle="--", zorder=2,
        )
        ax.add_patch(uncertainty_halo)
        # Ghost marker: hollow diamond at the perceived attacker pos.
        (ghost_marker,) = ax.plot(
            [], [], marker="D", color="orange",
            markerfacecolor="none", markeredgecolor="orange",
            markersize=11, markeredgewidth=1.6,
            zorder=4, label="defender belief",
        )
        # Dashed line linking truth to belief.
        (belief_link,) = ax.plot(
            [], [], "--", color="orange", alpha=0.55, linewidth=0.9, zorder=3,
        )
        # Trail of past perceived positions.
        (ghost_trail,) = ax.plot(
            [], [], "-", color="orange", alpha=0.30, linewidth=1.0, zorder=2,
        )
        # Reference rings around the TRUE attacker showing where uncertainty
        # transitions from "clean" to "full". Drawn faintly so they don't
        # dominate the frame.
        r_clean = float(env_meta.get("noise_r_clean", 0.0))
        r_full = float(env_meta.get("noise_r_full", 0.0))
        info_ring_clean = plt.Circle(
            (0.0, 0.0), r_clean,
            facecolor="none", edgecolor="green",
            alpha=0.25, linewidth=1.0, linestyle=":", zorder=1,
        )
        info_ring_full = plt.Circle(
            (0.0, 0.0), r_full,
            facecolor="none", edgecolor="orange",
            alpha=0.18, linewidth=1.0, linestyle=":", zorder=1,
        )
        ax.add_patch(info_ring_clean)
        ax.add_patch(info_ring_full)
        # Live readout of current uncertainty magnitude.
        info_text = ax.text(
            0.02, 0.97, "", transform=ax.transAxes,
            fontsize=9, verticalalignment="top",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=2.0),
            zorder=10,
        )
    else:
        uncertainty_halo = None
        ghost_marker = None
        belief_link = None
        ghost_trail = None
        info_ring_clean = None
        info_ring_full = None
        info_text = None

    ax.legend(loc="upper right", fontsize=9)

    extra_artists = []
    if has_uncertainty:
        extra_artists = [
            uncertainty_halo, ghost_marker, belief_link, ghost_trail,
            info_ring_clean, info_ring_full, info_text,
        ]

    def _init() -> tuple:
        return (
            attacker_marker,
            defender_scatter,
            attacker_trail,
            *defender_trails,
            *capture_circles,
            *extra_artists,
        )

    def _update(frame: int) -> tuple:
        attacker_marker.set_data([a_traj[frame, 0]], [a_traj[frame, 1]])
        defender_scatter.set_offsets(d_traj[frame])
        stunned = cd_traj[frame] > 0  # (k,)
        marker_colors = np.where(stunned, "lightgray", "blue")
        defender_scatter.set_facecolors(marker_colors)
        start = max(0, frame - trail_len)
        attacker_trail.set_data(a_traj[start : frame + 1, 0], a_traj[start : frame + 1, 1])
        for i, trail in enumerate(defender_trails):
            trail.set_data(
                d_traj[start : frame + 1, i, 0], d_traj[start : frame + 1, i, 1]
            )
        for i, circle in enumerate(capture_circles):
            circle.center = (float(d_traj[frame, i, 0]), float(d_traj[frame, i, 1]))
            circle.set_color("lightgray" if stunned[i] else "blue")
            circle.set_alpha(0.18 if stunned[i] else 0.12)
        if has_uncertainty:
            ox, oy = float(obs_a_traj[frame, 0]), float(obs_a_traj[frame, 1])
            tx, ty = float(a_traj[frame, 0]), float(a_traj[frame, 1])
            g = float(gate_traj[frame])
            err = float(np.hypot(ox - tx, oy - ty))
            # Halo radius: actual current error (what the defender is off by).
            # Floor at a small value so it remains visible even when tiny.
            halo_r = max(err, 1e-3)
            uncertainty_halo.center = (ox, oy)
            uncertainty_halo.set_radius(halo_r)
            uncertainty_halo.set_alpha(0.06 + 0.30 * g)
            ghost_marker.set_data([ox], [oy])
            ghost_marker.set_alpha(0.20 + 0.75 * g)
            belief_link.set_data([tx, ox], [ty, oy])
            belief_link.set_alpha(0.10 + 0.55 * g)
            ghost_trail.set_data(
                obs_a_traj[start : frame + 1, 0],
                obs_a_traj[start : frame + 1, 1],
            )
            info_ring_clean.center = (tx, ty)
            info_ring_full.center = (tx, ty)
            info_text.set_text(
                f"gate={g:.2f}   err={err:.3f}\n"
                f"σ_max={noise_sigma:.2f}"
            )
        return (
            attacker_marker,
            defender_scatter,
            attacker_trail,
            *defender_trails,
            *capture_circles,
            *extra_artists,
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
