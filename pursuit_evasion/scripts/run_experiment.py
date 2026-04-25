"""Main entrypoint: heuristic baseline + (optional) RL training sweep + plots.

Usage::
    python scripts/run_experiment.py --config config.yaml

Steps:
  1. Always run the heuristic-vs-heuristic baseline across the full sweep
     and save ``phase_curve_heuristic.png`` (cheap and a fallback if RL
     training fails).
  2. If ``mode: rl`` in the config, train one neural cell per (k, σ, p)
     combination in parallel via ``multiprocessing``. Each cell trains
     under :func:`alternating_train`, then is evaluated and animated.
  3. Save ``phase_curve.png`` (RL) and, if multiple (σ, p) rows are
     present, ``phase_heatmap.png``.
"""
from __future__ import annotations

import argparse
import itertools
import json
import logging
import multiprocessing as mp
import os
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _setup_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("run_experiment")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S")
    if not logger.handlers:
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        logger.addHandler(sh)
        fh = logging.FileHandler(log_path, mode="w")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def _as_list(x) -> list:
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


# ----------------------------------------------------------- worker functions


def _heuristic_cell(args: tuple) -> dict:
    """Run one heuristic-vs-heuristic eval cell."""
    k, sigma, p, n_episodes, env_kwargs, seed = args
    # Local imports keep multiprocessing pickling cheap.
    from src.env import PursuitEvasionEnv
    from src.policies import HeuristicAttacker, HeuristicDefender
    from src.rollout import evaluate

    env = PursuitEvasionEnv(batch_size=1, k=k, sigma=sigma, p=p, **env_kwargs)
    a = HeuristicAttacker(target=env.target, v_attacker=env.v_attacker)
    d = HeuristicDefender(v_defender=env.v_defender, capture_radius=env.capture_radius)
    return evaluate(
        k=k,
        sigma=sigma,
        p=p,
        attacker_policy=a,
        defender_policy=d,
        n_episodes=n_episodes,
        seed=seed,
        env_kwargs=env_kwargs,
    )


def _rl_cell(args: tuple) -> dict:
    """Train one (k, σ, p) cell and evaluate the resulting policies."""
    cell_id, k, sigma, p, training_cfg, env_kwargs, n_eval, n_anim, seed, out_root = args
    threads = int(os.environ.get("WORKER_THREADS", "1"))
    os.environ.setdefault("OMP_NUM_THREADS", str(threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(threads))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(threads))
    import torch

    torch.set_num_threads(threads)

    from src.env import PursuitEvasionEnv
    from src.policies import (
        CentralizedDefender,
        HeuristicAttacker,
        HeuristicDefender,
        NeuralAttacker,
        NeuralDefender,
    )
    from src.rollout import evaluate
    from src.train import PPOConfig, alternating_train, alternating_train_centralized
    from src.animate import run_episode, save_animation

    cell_dir = Path(out_root) / "checkpoints" / f"k{k}_sigma{sigma}_p{p}"
    cell_dir.mkdir(parents=True, exist_ok=True)
    log_lines: list[str] = []

    def log(msg: str) -> None:
        log_lines.append(msg)

    cfg = PPOConfig(
        n_steps=training_cfg["n_steps"],
        n_epochs=training_cfg["n_epochs"],
        minibatch_size=training_cfg["minibatch_size"],
        lr=training_cfg["lr"],
        gamma=training_cfg["gamma"],
        gae_lambda=training_cfg["gae_lambda"],
        clip_coef=training_cfg["clip_coef"],
        ent_coef=training_cfg["ent_coef"],
        vf_coef=training_cfg["vf_coef"],
        max_grad_norm=training_cfg["max_grad_norm"],
    )
    defender_type = str(training_cfg.get("defender_type", "shared")).lower()
    t0 = time.perf_counter()
    if defender_type == "centralized":
        res = alternating_train_centralized(
            k=k,
            sigma=sigma,
            p=p,
            cfg=cfg,
            n_envs=training_cfg["n_envs"],
            attacker_warmup_iters=training_cfg["attacker_warmup_iters"],
            defender_iters=training_cfg["defender_iters"],
            attacker_iters=training_cfg["attacker_iters"],
            n_alternations=training_cfg["n_alternations"],
            seed=seed,
            central_hidden=int(training_cfg.get("central_hidden", 256)),
            central_n_layers=int(training_cfg.get("central_n_layers", 3)),
            env_kwargs=env_kwargs,
            save_dir=cell_dir,
            log_fn=log,
        )
    elif defender_type == "shared":
        res = alternating_train(
            k=k,
            sigma=sigma,
            p=p,
            cfg=cfg,
            n_envs=training_cfg["n_envs"],
            attacker_warmup_iters=training_cfg["attacker_warmup_iters"],
            defender_iters=training_cfg["defender_iters"],
            attacker_iters=training_cfg["attacker_iters"],
            n_alternations=training_cfg["n_alternations"],
            seed=seed,
            env_kwargs=env_kwargs,
            save_dir=cell_dir,
            log_fn=log,
        )
    else:
        raise ValueError(
            f"unknown defender_type={defender_type!r} (expected 'shared' or 'centralized')"
        )
    train_elapsed = time.perf_counter() - t0
    log(f"train_elapsed={train_elapsed:.1f}s")

    # Evaluate (final RL policies) — also use sampling=False for stable rollouts
    env_meta = res["env_meta"]
    a_pol = NeuralAttacker(res["attacker_net"], env_meta["v_attacker"], deterministic=True)
    if defender_type == "centralized":
        d_pol = CentralizedDefender(
            res["central_defender_net"], env_meta["v_defender"], deterministic=True
        )
    else:
        d_pol = NeuralDefender(
            res["defender_net"], env_meta["v_defender"], deterministic=True
        )
    eval_res = evaluate(
        k=k,
        sigma=sigma,
        p=p,
        attacker_policy=a_pol,
        defender_policy=d_pol,
        n_episodes=n_eval,
        seed=seed + 7919,
        env_kwargs=env_kwargs,
    )
    log(f"eval={eval_res}")

    # Render a few animations for visual inspection.
    anim_dir = Path(out_root) / "animations"
    anim_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n_anim):
        env_anim = PursuitEvasionEnv(
            batch_size=1, k=k, sigma=sigma, p=p, seed=seed + 1000 + i, **env_kwargs
        )
        a_pol1 = NeuralAttacker(res["attacker_net"], env_anim.v_attacker, deterministic=False)
        if defender_type == "centralized":
            d_pol1 = CentralizedDefender(
                res["central_defender_net"], env_anim.v_defender, deterministic=False
            )
        else:
            d_pol1 = NeuralDefender(
                res["defender_net"], env_anim.v_defender, deterministic=False
            )
        h = run_episode(env_anim, a_pol1, d_pol1)
        prefix = "rl_central" if defender_type == "centralized" else "rl"
        out_anim = anim_dir / f"{prefix}_k{k}_sigma{sigma}_p{p}_ep{i}.mp4"
        save_animation(h, env_anim.metadata(), out_anim)
        log(f"saved anim → {out_anim.name} ({h['steps']} steps, outcome={h['outcome']})")

    log_path = cell_dir / "train.log"
    log_path.write_text("\n".join(log_lines))

    metrics_path = cell_dir / "metrics.json"
    metrics_path.write_text(json.dumps(_jsonable(res["metrics"]), indent=2))

    phase_summaries: list[dict] = []
    warmup_ms = res["metrics"].get("warmup") or []
    if warmup_ms:
        phase_summaries.append({
            "name": "warmup", "n": len(warmup_ms),
            "first": warmup_ms[0], "last": warmup_ms[-1],
        })
    for ri, rd in enumerate(res["metrics"].get("rounds", [])):
        for side in ("defender", "attacker"):
            ms = rd.get(side) or []
            if ms:
                phase_summaries.append({
                    "name": f"round_{ri+1}_{side}", "n": len(ms),
                    "first": ms[0], "last": ms[-1],
                })

    training_info = {
        "method": (
            "alternating_train_centralized"
            if defender_type == "centralized"
            else "alternating_train"
        ),
        "defender_type": defender_type,
        "seed": seed,
        "n_envs": training_cfg["n_envs"],
        "ppo_config": asdict(cfg),
        "env_kwargs": env_kwargs,
        "training_cfg": dict(training_cfg),
        "schedule": res.get("schedule"),
        "phase_summaries": phase_summaries,
        "metrics_path": str(metrics_path),
    }

    return {
        "cell_id": cell_id,
        "k": k,
        "sigma": sigma,
        "p": p,
        "eval": eval_res,
        "train_seconds": train_elapsed,
        "log_path": str(log_path),
        "training_info": training_info,
    }


# --------------------------------------------------------------------- driver


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(_REPO_ROOT / "config.yaml"))
    parser.add_argument("--workers", type=int, default=0,
                        help="Parallel processes for the RL sweep (0 = auto = min(n_cells, cpu_count)).")
    parser.add_argument("--threads-per-worker", type=int, default=1,
                        help="BLAS / torch threads per worker process. "
                        "workers * threads_per_worker should not exceed cpu_count.")
    parser.add_argument("--out", type=str, default=str(_REPO_ROOT / "outputs"),
                        help="Output root.")
    parser.add_argument("--skip-rl", action="store_true",
                        help="Force-skip RL even if mode=rl in config.")
    parser.add_argument("--skip-heuristic", action="store_true",
                        help="Skip the heuristic baseline (for quick RL-only runs).")
    args = parser.parse_args()
    os.environ["WORKER_THREADS"] = str(args.threads_per_worker)

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    logger = _setup_logging(out_root / "run.log")

    config = yaml.safe_load(Path(args.config).read_text())
    logger.info(f"config: {json.dumps(config)}")

    k_values = list(config["sweep"]["k_values"])
    sigmas = _as_list(config["sweep"]["sigma"])
    ps = _as_list(config["sweep"]["p"])
    cells = list(itertools.product(k_values, sigmas, ps))
    logger.info(f"sweep cells: {len(cells)}  ({k_values=}, {sigmas=}, {ps=})")

    env_kwargs = {
        "dt": config["env"]["dt"],
        "max_steps": config["env"]["max_steps"],
        "stun_steps": int(config["env"].get("stun_steps", 0)),
        "capture_radius": config["env"]["capture_radius"],
        "target_radius": config["env"]["target_radius"],
        "target_pos": tuple(config["env"]["target_pos"]),
        "v_attacker": config["env"]["v_attacker"],
        "inner_radius": config["env"]["defender_inner_radius"],
        "outer_radius": config["env"]["defender_outer_radius"],
        "shaping": config["env"]["attacker_shaping"],  # symmetric: same coef both sides
    }
    seed = int(config["seed"])
    n_eval = int(config["evaluation"]["n_episodes"])
    n_anim = int(config["evaluation"]["n_animations"])

    # ---- step 1: heuristic baseline ----
    if not args.skip_heuristic:
        logger.info("=== heuristic baseline ===")
        heuristic_args = [
            (k, s, p, n_eval, env_kwargs, seed) for (k, s, p) in cells
        ]
        if args.workers != 1:
            n_workers = args.workers if args.workers > 0 else min(len(heuristic_args), os.cpu_count() or 1)
            with mp.Pool(n_workers) as pool:
                heuristic_results = pool.map(_heuristic_cell, heuristic_args)
        else:
            heuristic_results = [_heuristic_cell(a) for a in heuristic_args]

        for r in heuristic_results:
            ci_lo, ci_hi = r["ci_95"]
            logger.info(
                f"  [heur] k={r['k']:2d} σ={r['sigma']:.2f} p={r['p']:.2f}  "
                f"succ={r['attacker_success_rate']:.3f} "
                f"CI=[{ci_lo:.3f},{ci_hi:.3f}]  len={r['mean_episode_length']:.1f}"
            )
        _save_curve_from_results(heuristic_results, sigmas, ps,
                                 out=out_root / "plots" / "phase_curve_heuristic.png",
                                 title="Heuristic baseline — attacker success vs k")
    else:
        heuristic_results = []

    # ---- step 2: RL training ----
    mode = config.get("mode", "rl")
    if args.skip_rl or mode != "rl":
        logger.info("RL skipped.")
        return

    logger.info("=== RL training sweep ===")
    rl_args = []
    for i, (k, s, p) in enumerate(cells):
        rl_args.append((
            i, k, s, p, config["training"], env_kwargs, n_eval, n_anim, seed, str(out_root),
        ))
    n_workers = args.workers if args.workers > 0 else min(len(rl_args), os.cpu_count() or 1)
    logger.info(f"n_workers = {n_workers}, n_cells = {len(rl_args)}")

    t0 = time.perf_counter()
    if n_workers == 1:
        rl_results = [_rl_cell(a) for a in rl_args]
    else:
        with mp.Pool(n_workers) as pool:
            rl_results = []
            for r in pool.imap_unordered(_rl_cell, rl_args):
                rl_results.append(r)
                ci = r["eval"]["ci_95"]
                logger.info(
                    f"  [rl  ] k={r['k']:2d} σ={r['sigma']:.2f} p={r['p']:.2f}  "
                    f"succ={r['eval']['attacker_success_rate']:.3f} "
                    f"CI=[{ci[0]:.3f},{ci[1]:.3f}]  "
                    f"train={r['train_seconds']:.0f}s"
                )
    elapsed = time.perf_counter() - t0
    logger.info(f"RL sweep total wall: {elapsed:.0f}s")

    # Sort by k for plotting
    rl_results.sort(key=lambda r: (r["sigma"], r["p"], r["k"]))
    _save_curve_from_results(rl_results, sigmas, ps,
                             out=out_root / "plots" / "phase_curve.png",
                             title="RL — attacker success vs k",
                             rate_key=("eval", "attacker_success_rate"),
                             ci_key=("eval", "ci_95"))

    # Heatmap if more than one (σ, p) row.
    if len(sigmas) > 1 or len(ps) > 1:
        _save_heatmap(rl_results, k_values, sigmas, ps,
                      out=out_root / "plots" / "phase_heatmap.png")

    # Persist machine-readable results.
    out_json = out_root / "results.json"
    out_json.write_text(json.dumps({
        "config": config,
        "git": _git_info(_REPO_ROOT),
        "heuristic": [_jsonable(r) for r in heuristic_results],
        "rl": [_jsonable(r) for r in rl_results],
    }, indent=2))
    logger.info(f"results → {out_json}")


def _jsonable(r):
    """Make a result dict JSON-friendly (tuples → lists)."""
    if isinstance(r, dict):
        return {k: _jsonable(v) for k, v in r.items()}
    if isinstance(r, (list, tuple)):
        return [_jsonable(v) for v in r]
    if isinstance(r, np.ndarray):
        return r.tolist()
    if isinstance(r, (np.floating, np.integer)):
        return r.item()
    return r


def _git_info(repo_root: Path) -> dict:
    """Capture commit/branch/dirty status so saved results pin the code version."""
    def _run(*args: str) -> str:
        try:
            return subprocess.check_output(
                ["git", "-C", str(repo_root), *args],
                text=True, stderr=subprocess.DEVNULL,
            ).strip()
        except Exception:
            return ""
    status = _run("status", "--porcelain")
    return {
        "commit": _run("rev-parse", "HEAD"),
        "branch": _run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status),
        "dirty_files": [line[3:] for line in status.splitlines() if line.strip()],
    }


def _save_curve_from_results(results, sigmas, ps, *, out, title,
                             rate_key=("attacker_success_rate",),
                             ci_key=("ci_95",)):
    from src.plot import phase_curve

    def _get(d, path):
        for p in path:
            d = d[p]
        return d

    series = []
    for s in sigmas:
        for p in ps:
            sub = sorted([r for r in results if r["sigma"] == s and r["p"] == p],
                         key=lambda r: r["k"])
            if not sub:
                continue
            ks = [r["k"] for r in sub]
            rate = [_get(r, rate_key) for r in sub]
            ci = [_get(r, ci_key) for r in sub]
            ci_lo = [c[0] for c in ci]
            ci_hi = [c[1] for c in ci]
            series.append({
                "label": f"σ={s}, p={p}",
                "k": ks, "rate": rate, "ci_lo": ci_lo, "ci_hi": ci_hi,
            })
    phase_curve(series, out_path=out, title=title)


def _save_heatmap(results, k_values, sigmas, ps, *, out):
    from src.plot import phase_heatmap

    rows, labels = [], []
    for s in sigmas:
        for p in ps:
            sub = sorted([r for r in results if r["sigma"] == s and r["p"] == p],
                         key=lambda r: r["k"])
            if not sub:
                continue
            row = [r["eval"]["attacker_success_rate"] for r in sub]
            rows.append(row)
            labels.append(f"σ={s}, p={p}")
    grid = np.asarray(rows)
    phase_heatmap(grid, k_values=k_values, row_labels=labels, out_path=out,
                  title="Attacker success rate (RL)")


if __name__ == "__main__":
    main()
