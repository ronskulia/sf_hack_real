"""Phase-transition plots: per-(σ,p) curve and (σ,p)×k heatmap."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def phase_curve(
    results: list[dict],
    *,
    out_path: str | Path,
    title: str = "Attacker success rate vs k",
) -> Path:
    """Plot attacker success rate vs k for one or more (σ, p) configurations.

    Parameters
    ----------
    results : list[dict]
        Each dict: ``{"label": str, "k": [...], "rate": [...], "ci_lo": [...], "ci_hi": [...]}``.
    out_path : path
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    markers = ["o", "s", "^", "D", "v", "<", ">"]
    for i, r in enumerate(results):
        ks = np.asarray(r["k"])
        rate = np.asarray(r["rate"])
        lo = np.asarray(r["ci_lo"])
        hi = np.asarray(r["ci_hi"])
        yerr = np.stack([rate - lo, hi - rate], axis=0)
        ax.errorbar(
            ks,
            rate,
            yerr=yerr,
            fmt=markers[i % len(markers)] + "-",
            capsize=4,
            linewidth=2,
            markersize=8,
            label=r.get("label", f"series {i}"),
        )
    ax.set_xlabel("k (number of defenders)", fontsize=12)
    ax.set_ylabel("attacker success rate", fontsize=12)
    ax.set_title(title, fontsize=12)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    if any("label" in r for r in results):
        ax.legend(fontsize=10, loc="upper right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def phase_heatmap(
    grid: np.ndarray,
    *,
    k_values: Iterable[int],
    row_labels: Iterable[str],
    out_path: str | Path,
    title: str = "Attacker success rate",
) -> Path:
    """2-D heatmap with k on x and (σ, p) configs on y.

    Parameters
    ----------
    grid : np.ndarray
        Shape ``(n_rows, n_k)``, values in ``[0, 1]``.
    k_values : iterable of int
    row_labels : iterable of str
    out_path : path
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    grid = np.asarray(grid)
    fig, ax = plt.subplots(figsize=(8, 1.2 + 0.5 * grid.shape[0]))
    im = ax.imshow(grid, cmap="viridis", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(grid.shape[1]))
    ax.set_xticklabels(list(k_values))
    ax.set_yticks(range(grid.shape[0]))
    ax.set_yticklabels(list(row_labels))
    ax.set_xlabel("k", fontsize=12)
    ax.set_title(title, fontsize=12)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            ax.text(
                j, i, f"{grid[i, j]:.2f}",
                ha="center", va="center",
                color="white" if grid[i, j] < 0.5 else "black",
                fontsize=9,
            )
    cbar = fig.colorbar(im, ax=ax, fraction=0.04)
    cbar.set_label("attacker success rate", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def training_progress_curve(
    metrics: dict[str, list[dict]],
    *,
    out_path: str | Path,
    title: str = "Training progress",
) -> Path:
    """Plot per-iteration success metrics for centralized training.

    Parameters
    ----------
    metrics : dict
        Expected keys are ``"attacker"`` and ``"central_defender"``. Values are
        metric dictionaries emitted by the PPO training loops.
    out_path : path
        Destination PNG path.
    title : str
        Figure title.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    attacker = metrics.get("attacker") or []
    central_defender = metrics.get("central_defender") or []

    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=False)
    ax_succ, ax_ret = axes

    attacker_offset = 0.0
    if attacker:
        attacker_offset = float(attacker[-1].get("step", len(attacker)))

    def _series(
        rows: list[dict],
        key: str,
        *,
        offset: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        x = np.asarray(
            [float(r.get("step", i)) + offset for i, r in enumerate(rows)],
            dtype=float,
        )
        y = np.asarray([float(r.get(key, np.nan)) for r in rows], dtype=float)
        ok = np.isfinite(x) & np.isfinite(y)
        return x[ok], y[ok]

    for rows, phase, color in (
        (attacker, "attacker warmup", "tab:red"),
        (central_defender, "central defender", "tab:blue"),
    ):
        offset = attacker_offset if phase == "central defender" else 0.0
        x, y = _series(rows, "attacker_succ", offset=offset)
        if x.size:
            ax_succ.plot(x, y, marker="o", markersize=3, linewidth=1.5,
                         label=f"{phase}: attacker success", color=color)

    x, y = _series(central_defender, "defender_succ", offset=attacker_offset)
    if x.size:
        ax_succ.plot(x, y, marker="s", markersize=3, linewidth=1.5,
                     label="central defender: defender success", color="tab:green")

    for rows, phase, color in (
        (attacker, "attacker warmup", "tab:red"),
        (central_defender, "central defender", "tab:blue"),
    ):
        offset = attacker_offset if phase == "central defender" else 0.0
        x, y = _series(rows, "ep_return", offset=offset)
        if x.size:
            ax_ret.plot(x, y, marker="o", markersize=3, linewidth=1.5,
                        label=f"{phase}: episode return", color=color)

    ax_succ.set_title(title, fontsize=12)
    ax_succ.set_ylabel("success rate")
    ax_succ.set_ylim(-0.05, 1.05)
    ax_succ.grid(True, alpha=0.3)
    ax_succ.legend(fontsize=9, loc="best")

    ax_ret.set_xlabel("environment steps")
    ax_ret.set_ylabel("mean episode return")
    ax_ret.grid(True, alpha=0.3)
    ax_ret.legend(fontsize=9, loc="best")

    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path
