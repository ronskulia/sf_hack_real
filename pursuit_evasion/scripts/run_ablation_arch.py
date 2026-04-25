"""Ablation over centralized-defender NN size at fixed (k, sigma, p).

Each (hidden, n_layers) combo is one cell, run in parallel via
multiprocessing. Uses the same alternating-self-play schedule as
``alternating_train_centralized``. Independent of run_experiment.py.

Usage::
    python scripts/run_ablation_arch.py --k 4 --sigma 0.5 --p 0.7 \\
        --archs 128:2,192:3,256:3,384:3 --out outputs_ablation_arch
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import yaml  # noqa: F401  (keep parity with other runners)

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _parse_archs(s: str) -> list[tuple[int, int]]:
    archs: list[tuple[int, int]] = []
    for tok in s.split(","):
        h, n = tok.split(":")
        archs.append((int(h), int(n)))
    return archs


def _ablation_cell(args: tuple) -> dict:
    cell_id, hidden, n_layers, k, sigma, p, training_cfg, env_kwargs, n_eval, n_anim, seed, out_root = args
    threads = int(os.environ.get("WORKER_THREADS", "1"))
    os.environ.setdefault("OMP_NUM_THREADS", str(threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(threads))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(threads))
    import torch

    torch.set_num_threads(threads)

    from src.animate import run_episode, save_animation
    from src.env import PursuitEvasionEnv
    from src.policies import CentralizedDefender, NeuralAttacker
    from src.rollout import evaluate
    from src.train import PPOConfig, alternating_train_centralized

    cell_tag = f"h{hidden}_L{n_layers}"
    cell_dir = Path(out_root) / "checkpoints" / cell_tag
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

    t0 = time.perf_counter()
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
        central_hidden=hidden,
        central_n_layers=n_layers,
        env_kwargs=env_kwargs,
        save_dir=cell_dir,
        log_fn=log,
    )
    train_elapsed = time.perf_counter() - t0
    log(f"train_elapsed={train_elapsed:.1f}s")

    env_meta = res["env_meta"]
    a_pol = NeuralAttacker(res["attacker_net"], env_meta["v_attacker"], deterministic=True)
    d_pol = CentralizedDefender(
        res["central_defender_net"], env_meta["v_defender"], deterministic=True
    )
    eval_res = evaluate(
        k=k, sigma=sigma, p=p,
        attacker_policy=a_pol, defender_policy=d_pol,
        n_episodes=n_eval, seed=seed + 7919, env_kwargs=env_kwargs,
    )
    log(f"eval={eval_res}")

    anim_dir = Path(out_root) / "animations"
    anim_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n_anim):
        env_anim = PursuitEvasionEnv(
            batch_size=1, k=k, sigma=sigma, p=p, seed=seed + 1000 + i, **env_kwargs
        )
        a_anim = NeuralAttacker(res["attacker_net"], env_anim.v_attacker, deterministic=False)
        d_anim = CentralizedDefender(
            res["central_defender_net"], env_anim.v_defender, deterministic=False
        )
        h = run_episode(env_anim, a_anim, d_anim)
        out_anim = anim_dir / f"{cell_tag}_ep{i}.mp4"
        save_animation(h, env_anim.metadata(), out_anim)
        log(f"saved anim → {out_anim.name} ({h['steps']} steps, outcome={h['outcome']})")

    # full per-iter metrics dump
    metrics_path = cell_dir / "metrics.json"
    metrics_path.write_text(json.dumps(_jsonable(res["metrics"]), indent=2))

    phase_summaries: list[dict] = []
    if res["metrics"].get("warmup"):
        ms = res["metrics"]["warmup"]
        phase_summaries.append({
            "name": "warmup", "n": len(ms), "first": ms[0], "last": ms[-1],
        })
    for ri, rd in enumerate(res["metrics"].get("rounds", [])):
        for side in ("defender", "attacker"):
            ms = rd.get(side) or []
            if ms:
                phase_summaries.append({
                    "name": f"round_{ri+1}_{side}", "n": len(ms),
                    "first": ms[0], "last": ms[-1],
                })

    log_path = cell_dir / "train.log"
    log_path.write_text("\n".join(log_lines))

    return {
        "cell_id": cell_id,
        "central_hidden": hidden,
        "central_n_layers": n_layers,
        "k": k, "sigma": sigma, "p": p,
        "eval": eval_res,
        "train_seconds": train_elapsed,
        "log_path": str(log_path),
        "training_info": {
            "method": "alternating_train_centralized",
            "defender_type": "centralized",
            "central_hidden": hidden,
            "central_n_layers": n_layers,
            "seed": seed,
            "n_envs": training_cfg["n_envs"],
            "ppo_config": asdict(cfg),
            "env_kwargs": env_kwargs,
            "training_cfg": dict(training_cfg),
            "schedule": res.get("schedule"),
            "phase_summaries": phase_summaries,
            "metrics_path": str(metrics_path),
        },
    }


def _jsonable(r):
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
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--sigma", type=float, default=0.5)
    parser.add_argument("--p", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--archs", type=str, default="128:2,192:3,256:3,384:3",
                        help="Comma-separated 'hidden:n_layers' tuples.")
    parser.add_argument("--out", type=str, default=str(_REPO_ROOT / "outputs_ablation_arch"))
    parser.add_argument("--workers", type=int, default=0,
                        help="0 = auto = min(n_archs, cpu_count)")
    parser.add_argument("--threads-per-worker", type=int, default=2)
    parser.add_argument("--n-envs", type=int, default=256)
    parser.add_argument("--n-steps", type=int, default=128)
    parser.add_argument("--attacker-warmup-iters", type=int, default=50)
    parser.add_argument("--attacker-iters", type=int, default=100)
    parser.add_argument("--defender-iters", type=int, default=100)
    parser.add_argument("--n-alternations", type=int, default=3)
    parser.add_argument("--n-eval", type=int, default=2000)
    parser.add_argument("--n-anim", type=int, default=2)
    args = parser.parse_args()

    os.environ["WORKER_THREADS"] = str(args.threads_per_worker)

    archs = _parse_archs(args.archs)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    env_kwargs = {
        "dt": 0.02, "max_steps": 500, "capture_radius": 0.03,
        "target_radius": 0.10, "target_pos": (0.5, 0.5), "v_attacker": 1.0,
        "inner_radius": 0.15, "outer_radius": 0.35, "shaping": 0.01,
    }
    training_cfg = {
        "n_envs": args.n_envs,
        "n_steps": args.n_steps,
        "n_epochs": 4,
        "minibatch_size": 512,
        "lr": 3e-4,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_coef": 0.2,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "attacker_warmup_iters": args.attacker_warmup_iters,
        "attacker_iters": args.attacker_iters,
        "defender_iters": args.defender_iters,
        "n_alternations": args.n_alternations,
    }

    print(f"=== ablation arch  k={args.k} sigma={args.sigma} p={args.p}  archs={archs} ===")
    print(f"out={out_root}")

    cell_args = [
        (i, h, n, args.k, args.sigma, args.p, training_cfg, env_kwargs,
         args.n_eval, args.n_anim, args.seed, str(out_root))
        for i, (h, n) in enumerate(archs)
    ]
    n_workers = args.workers if args.workers > 0 else min(len(cell_args), os.cpu_count() or 1)
    print(f"n_workers={n_workers}  threads_per_worker={args.threads_per_worker}")

    t0 = time.perf_counter()
    if n_workers == 1:
        results = [_ablation_cell(a) for a in cell_args]
    else:
        with mp.Pool(n_workers) as pool:
            results = []
            for r in pool.imap_unordered(_ablation_cell, cell_args):
                results.append(r)
                ev = r["eval"]
                ci = ev["ci_95"]
                print(
                    f"  [arch h={r['central_hidden']} L={r['central_n_layers']}]  "
                    f"succ={ev['attacker_success_rate']:.3f}  "
                    f"CI=[{ci[0]:.3f},{ci[1]:.3f}]  "
                    f"train={r['train_seconds']:.0f}s"
                )
    elapsed = time.perf_counter() - t0
    print(f"sweep wall: {elapsed:.0f}s")

    results.sort(key=lambda r: (r["central_hidden"], r["central_n_layers"]))
    out_json = out_root / "results.json"
    out_json.write_text(json.dumps({
        "experiment": "centralized_defender_arch_ablation",
        "fixed": {"k": args.k, "sigma": args.sigma, "p": args.p, "seed": args.seed},
        "archs": [{"central_hidden": h, "central_n_layers": n} for h, n in archs],
        "git": _git_info(_REPO_ROOT),
        "cells": [_jsonable(r) for r in results],
    }, indent=2))
    print(f"results → {out_json}")


if __name__ == "__main__":
    main()
