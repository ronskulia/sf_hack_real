"""Centralized-defender training pipeline.

Two-phase, no-alternation: train attacker (bigger MLP) vs heuristic defender
with snapshotting, then train centralized defender (one brain, sees all,
bigger MLP) vs a population of attacker snapshots + heuristic.

Sweeps over k = 1..6 in parallel via multiprocessing.
"""
from __future__ import annotations

import argparse
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


def _setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("run_central")
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


def _cell(args: tuple) -> dict:
    """Train one (k, σ, p) cell with sequential_train, then evaluate."""
    cell_id, k, sigma, p, training, env_kwargs, n_eval, n_anim, seed, out_root = args

    threads = int(os.environ.get("WORKER_THREADS", "1"))
    os.environ.setdefault("OMP_NUM_THREADS", str(threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(threads))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(threads))
    import torch

    torch.set_num_threads(threads)

    from src.animate import run_episode, save_animation
    from src.env import PursuitEvasionEnv
    from src.policies import (
        CentralizedDefender,
        HeuristicAttacker,
        HeuristicDefender,
        NeuralAttacker,
    )
    from src.rollout import evaluate
    from src.train import PPOConfig, sequential_train

    cell_dir = Path(out_root) / "checkpoints" / f"k{k}_sigma{sigma}_p{p}"
    cell_dir.mkdir(parents=True, exist_ok=True)
    log_lines: list[str] = []

    def log(msg: str) -> None:
        log_lines.append(msg)

    cfg = PPOConfig(
        n_steps=training["n_steps"],
        n_epochs=training["n_epochs"],
        minibatch_size=training["minibatch_size"],
        lr=training["lr"],
        gamma=training["gamma"],
        gae_lambda=training["gae_lambda"],
        clip_coef=training["clip_coef"],
        ent_coef=training["ent_coef"],
        vf_coef=training["vf_coef"],
        max_grad_norm=training["max_grad_norm"],
    )

    t0 = time.perf_counter()
    res = sequential_train(
        k=k,
        sigma=sigma,
        p=p,
        cfg=cfg,
        n_envs=training["n_envs"],
        attacker_warmup_iters=training["attacker_warmup_iters"],
        central_defender_iters=training["central_defender_iters"],
        snapshot_fractions=tuple(training.get("snapshot_fractions", (0.4, 0.7, 1.0))),
        attacker_hidden=training.get("attacker_hidden", 256),
        attacker_n_layers=training.get("attacker_n_layers", 2),
        central_hidden=training.get("central_hidden", 256),
        central_n_layers=training.get("central_n_layers", 3),
        seed=seed,
        env_kwargs=env_kwargs,
        save_dir=cell_dir,
        log_fn=log,
    )
    train_elapsed = time.perf_counter() - t0
    log(f"train_elapsed={train_elapsed:.1f}s")

    env_meta = res["env_meta"]
    a_pol = NeuralAttacker(res["attacker_net"], env_meta["v_attacker"], deterministic=True)
    cd_pol = CentralizedDefender(
        res["central_defender_net"], env_meta["v_defender"], deterministic=True
    )

    # Headline eval: trained attacker vs centralized defender
    eval_main = evaluate(
        k=k, sigma=sigma, p=p,
        attacker_policy=a_pol, defender_policy=cd_pol,
        n_episodes=n_eval, seed=seed + 7919, env_kwargs=env_kwargs,
    )
    log(f"eval_main={eval_main}")

    # Sanity 1: trained attacker vs heuristic defender (should be high)
    heur_def = None
    eval_vs_heur_def = None
    try:
        env_meta_full = env_meta
        heur_def = HeuristicDefender(
            v_defender=env_meta_full["v_defender"],
            capture_radius=env_meta_full["capture_radius"],
        )
        eval_vs_heur_def = evaluate(
            k=k, sigma=sigma, p=p,
            attacker_policy=a_pol, defender_policy=heur_def,
            n_episodes=n_eval, seed=seed + 7920, env_kwargs=env_kwargs,
        )
        log(f"eval_attacker_vs_heuristic_def={eval_vs_heur_def}")
    except Exception as e:  # pragma: no cover
        log(f"sanity1 failed: {e}")

    # Sanity 2: heuristic attacker vs centralized defender (should be low)
    heur_att = HeuristicAttacker(
        target=np.asarray(env_meta["target_pos"], dtype=np.float32),
        v_attacker=env_meta["v_attacker"],
    )
    eval_vs_heur_att = evaluate(
        k=k, sigma=sigma, p=p,
        attacker_policy=heur_att, defender_policy=cd_pol,
        n_episodes=n_eval, seed=seed + 7921, env_kwargs=env_kwargs,
    )
    log(f"eval_heuristic_att_vs_cdef={eval_vs_heur_att}")

    # Animations: 3 episodes with the full RL pair
    anim_dir = Path(out_root) / "animations"
    anim_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n_anim):
        env_anim = PursuitEvasionEnv(
            batch_size=1, k=k, sigma=sigma, p=p, seed=seed + 1000 + i, **env_kwargs
        )
        a_anim = NeuralAttacker(res["attacker_net"], env_anim.v_attacker, deterministic=False)
        cd_anim = CentralizedDefender(
            res["central_defender_net"], env_anim.v_defender, deterministic=False
        )
        h = run_episode(env_anim, a_anim, cd_anim)
        out_anim = anim_dir / f"central_k{k}_sigma{sigma}_p{p}_ep{i}.mp4"
        save_animation(h, env_anim.metadata(), out_anim)
        log(f"saved anim → {out_anim.name} ({h['steps']} steps, outcome={h['outcome']})")

    log_path = cell_dir / "train.log"
    log_path.write_text("\n".join(log_lines))

    metrics_path = cell_dir / "metrics.json"
    metrics_path.write_text(json.dumps(_jsonable(res["metrics"]), indent=2))

    phase_summaries: list[dict] = []
    for phase_key in ("attacker", "central_defender"):
        ms = res["metrics"].get(phase_key) or []
        if ms:
            phase_summaries.append({
                "name": f"phase_{phase_key}", "n": len(ms),
                "first": ms[0], "last": ms[-1],
            })

    training_info = {
        "method": "sequential_train",
        "seed": seed,
        "n_envs": training["n_envs"],
        "ppo_config": asdict(cfg),
        "env_kwargs": env_kwargs,
        "training_cfg": dict(training),
        "schedule": res.get("schedule"),
        "phase_summaries": phase_summaries,
        "metrics_path": str(metrics_path),
    }

    return {
        "cell_id": cell_id,
        "k": k,
        "sigma": sigma,
        "p": p,
        "eval_main": eval_main,
        "eval_attacker_vs_heuristic_def": eval_vs_heur_def,
        "eval_heuristic_att_vs_cdef": eval_vs_heur_att,
        "train_seconds": train_elapsed,
        "log_path": str(log_path),
        "training_info": training_info,
    }


def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(_REPO_ROOT / "config_central.yaml"))
    parser.add_argument("--workers", type=int, default=0,
                        help="0 = auto = min(n_cells, cpu_count)")
    parser.add_argument("--out", type=str, default=str(_REPO_ROOT / "outputs_central"))
    parser.add_argument("--threads-per-worker", type=int, default=2,
                        help="BLAS threads per worker process.")
    args = parser.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    logger = _setup_logger(out_root / "run.log")

    config = yaml.safe_load(Path(args.config).read_text())
    logger.info(f"config: {json.dumps(config)}")

    k_values = list(config["sweep"]["k_values"])
    sigma = float(config["sweep"]["sigma"])
    p = float(config["sweep"]["p"])
    cells = [(k, sigma, p) for k in k_values]
    logger.info(f"cells: {len(cells)}  ({k_values=}, {sigma=}, {p=})")

    env_kwargs = {
        "dt": config["env"]["dt"],
        "max_steps": config["env"]["max_steps"],
        "capture_radius": config["env"]["capture_radius"],
        "target_radius": config["env"]["target_radius"],
        "target_pos": tuple(config["env"]["target_pos"]),
        "v_attacker": config["env"]["v_attacker"],
        "inner_radius": config["env"]["defender_inner_radius"],
        "outer_radius": config["env"]["defender_outer_radius"],
        "shaping": config["env"]["attacker_shaping"],
    }
    seed = int(config["seed"])
    n_eval = int(config["evaluation"]["n_episodes"])
    n_anim = int(config["evaluation"]["n_animations"])

    os.environ["WORKER_THREADS"] = str(args.threads_per_worker)

    cell_args = [
        (i, k, sigma, p, config["training"], env_kwargs, n_eval, n_anim, seed, str(out_root))
        for i, (k, _, _) in enumerate(cells)
    ]
    n_workers = args.workers if args.workers > 0 else min(len(cell_args), os.cpu_count() or 1)
    logger.info(
        f"n_workers={n_workers}  threads_per_worker={args.threads_per_worker}  "
        f"total_cores_in_use={n_workers * args.threads_per_worker}"
    )

    t0 = time.perf_counter()
    if n_workers == 1:
        results = [_cell(a) for a in cell_args]
    else:
        with mp.Pool(n_workers) as pool:
            results = []
            for r in pool.imap_unordered(_cell, cell_args):
                results.append(r)
                em = r["eval_main"]
                em_h = r.get("eval_attacker_vs_heuristic_def") or {}
                em_ch = r.get("eval_heuristic_att_vs_cdef") or {}
                logger.info(
                    f"  k={r['k']:2d}  σ={r['sigma']:.2f} p={r['p']:.2f}  "
                    f"main_succ={em['attacker_success_rate']:.3f}  "
                    f"att_vs_heur_def={em_h.get('attacker_success_rate', float('nan')):.3f}  "
                    f"heur_att_vs_cdef={em_ch.get('attacker_success_rate', float('nan')):.3f}  "
                    f"train={r['train_seconds']:.0f}s"
                )
    elapsed = time.perf_counter() - t0
    logger.info(f"sweep wall time: {elapsed:.0f}s")

    results.sort(key=lambda r: r["k"])
    out_json = out_root / "results.json"
    out_json.write_text(json.dumps({
        "config": config,
        "git": _git_info(_REPO_ROOT),
        "cells": [_jsonable(r) for r in results],
    }, indent=2))
    logger.info(f"results → {out_json}")

    # Phase curve plot
    from src.plot import phase_curve
    series = [{
        "label": f"trained attacker vs centralized defender (σ={sigma}, p={p})",
        "k": [r["k"] for r in results],
        "rate": [r["eval_main"]["attacker_success_rate"] for r in results],
        "ci_lo": [r["eval_main"]["ci_95"][0] for r in results],
        "ci_hi": [r["eval_main"]["ci_95"][1] for r in results],
    }]
    if all(r.get("eval_attacker_vs_heuristic_def") for r in results):
        series.append({
            "label": "trained attacker vs heuristic defender",
            "k": [r["k"] for r in results],
            "rate": [r["eval_attacker_vs_heuristic_def"]["attacker_success_rate"] for r in results],
            "ci_lo": [r["eval_attacker_vs_heuristic_def"]["ci_95"][0] for r in results],
            "ci_hi": [r["eval_attacker_vs_heuristic_def"]["ci_95"][1] for r in results],
        })
    series.append({
        "label": "heuristic attacker vs centralized defender",
        "k": [r["k"] for r in results],
        "rate": [r["eval_heuristic_att_vs_cdef"]["attacker_success_rate"] for r in results],
        "ci_lo": [r["eval_heuristic_att_vs_cdef"]["ci_95"][0] for r in results],
        "ci_hi": [r["eval_heuristic_att_vs_cdef"]["ci_95"][1] for r in results],
    })
    phase_curve(series, out_path=out_root / "plots" / "phase_curve_central.png",
                title=f"Centralized-defender phase curve (σ={sigma}, p={p})")
    logger.info(f"plot → {out_root / 'plots' / 'phase_curve_central.png'}")


if __name__ == "__main__":
    main()
