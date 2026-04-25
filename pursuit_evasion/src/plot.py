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
