"""Per-k pipeline: BC warm-start (imitate HeuristicOrbitalDefender)
then alternating-self-play RL fine-tune.

For each k in --k-values:
  Stage 1 (BC):  Train CentralizedDefenderNet via MSE regression on
                 (sorted_obs, sorted_action) pairs from HeuristicOrbitalDefender.
                 Stop when val_mae < --bc-target-mae or --bc-max-epochs reached.
  Stage 2 (RL):  alternating_train_centralized starting from BC weights.
                 The attacker's warmup phase uses the BC defender as opponent
                 (instead of HeuristicDefender) so the warm-start carries through.
  Eval + animations rendered against trained attacker.

Usage::
    python scripts/run_bc_then_rl.py --k-values 1 2 3 4 5 6 \\
        --sigma 0.8 --p 1.0 --workers 4 --threads-per-worker 2 \\
        --out outputs_bc_then_rl
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

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


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


# ============================================================================
# Per-cell worker (one k value)
# ============================================================================


def _cell(args: tuple) -> dict:
    cell_id, k, sigma, p, settings, env_kwargs, out_root = args
    threads = int(os.environ.get("WORKER_THREADS", "1"))
    os.environ.setdefault("OMP_NUM_THREADS", str(threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(threads))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(threads))
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", str(threads))
    import torch
    import torch.nn as nn_torch
    import torch.optim as optim

    torch.set_num_threads(threads)

    from src.animate import run_episode, save_animation
    from src.env import PursuitEvasionEnv
    from src.policies import (
        CentralizedDefender,
        CentralizedDefenderNet,
        HeuristicAttacker,
        HeuristicOrbitalDefender,
        NeuralAttacker,
    )
    from src.rollout import evaluate
    from src.train import PPOConfig, alternating_train_centralized

    cell_dir = Path(out_root) / "checkpoints" / f"k{k}_sigma{sigma}_p{p}"
    cell_dir.mkdir(parents=True, exist_ok=True)
    log_lines: list[str] = []

    def log(msg: str) -> None:
        log_lines.append(msg)
        print(f"[k={k}] {msg}", flush=True)

    seed = settings["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    # ---------- Stage 1: BC ----------
    bc_t0 = time.perf_counter()
    log(f"=== stage 1: BC (imitate HeuristicOrbitalDefender, sort=True) ===")

    env = PursuitEvasionEnv(
        batch_size=settings["n_envs"], k=k, sigma=sigma, p=p,
        seed=seed, **env_kwargs,
    )
    heuristic = HeuristicOrbitalDefender(
        v_defender=env.v_defender,
        target_pos=env_kwargs["target_pos"],
        orbit_radius=settings["orbit_radius"],
        capture_radius=env.capture_radius,
    )
    n_obs = 4 + 4 * k
    n_act = 2 * k
    total_samples = settings["bc_n_envs"] * settings["bc_n_rollouts"] * settings["bc_rollout_steps"]
    obs_arr = np.empty((total_samples, n_obs), dtype=np.float32)
    act_arr = np.empty((total_samples, n_act), dtype=np.float32)
    write = 0
    bc_env = PursuitEvasionEnv(
        batch_size=settings["bc_n_envs"], k=k, sigma=sigma, p=p,
        seed=seed, **env_kwargs,
    )
    for _ in range(settings["bc_n_rollouts"]):
        bc_env.reset()
        for _ in range(settings["bc_rollout_steps"]):
            obs_s, perm = CentralizedDefenderNet.build_sorted_obs(bc_env)
            d_act = heuristic.act(bc_env)  # env-ordered (B, k, 2)
            d_act_sorted = np.take_along_axis(d_act, perm[..., None], axis=1)
            obs_arr[write : write + bc_env.B] = obs_s
            act_arr[write : write + bc_env.B] = d_act_sorted.reshape(bc_env.B, n_act)
            write += bc_env.B
            theta = rng.uniform(-np.pi, np.pi, size=(bc_env.B,)).astype(np.float32)
            r_mag = rng.uniform(0.0, bc_env.v_attacker, size=(bc_env.B,)).astype(np.float32)
            a_vel = np.stack([r_mag * np.cos(theta), r_mag * np.sin(theta)], axis=-1)
            res = bc_env.step(a_vel, d_act)
            if res.just_done.any():
                bc_env.reset_idxs(res.just_done)

    n_total = obs_arr.shape[0]
    n_val = max(1, int(n_total * 0.1))
    perm_idx = rng.permutation(n_total)
    val_idx, train_idx = perm_idx[:n_val], perm_idx[n_val:]
    obs_train_t = torch.from_numpy(obs_arr[train_idx])
    act_train_t = torch.from_numpy(act_arr[train_idx])
    obs_val_t = torch.from_numpy(obs_arr[val_idx])
    act_val_t = torch.from_numpy(act_arr[val_idx])
    log(f"BC dataset: {n_total} samples (train={len(train_idx)}, val={n_val})")

    bc_net = CentralizedDefenderNet(
        k=k, hidden=settings["hidden"], n_layers=settings["n_layers"]
    )
    bc_opt = optim.Adam(bc_net.parameters(), lr=settings["bc_lr"])
    bc_history: list[dict] = []
    n_train = obs_train_t.shape[0]
    target_mae = settings["bc_target_mae"]
    max_epochs = settings["bc_max_epochs"]
    early_stopped_at = None
    for ep in range(max_epochs):
        idxs = np.random.permutation(n_train)
        running, count = 0.0, 0
        for start in range(0, n_train, settings["bc_batch_size"]):
            mb = idxs[start : start + settings["bc_batch_size"]]
            mean, _, _ = bc_net(obs_train_t[mb])
            loss = nn_torch.functional.mse_loss(mean, act_train_t[mb])
            bc_opt.zero_grad()
            loss.backward()
            bc_opt.step()
            running += loss.item() * len(mb)
            count += len(mb)
        train_mse = running / count
        with torch.no_grad():
            v_mean, _, _ = bc_net(obs_val_t)
            val_mse = nn_torch.functional.mse_loss(v_mean, act_val_t).item()
            val_mae = (v_mean - act_val_t).abs().mean().item()
        bc_history.append({
            "epoch": ep, "train_mse": train_mse,
            "val_mse": val_mse, "val_mae": val_mae,
        })
        if (ep % 5 == 0) or ep == max_epochs - 1:
            log(f"  BC ep {ep:3d}  train_mse={train_mse:.5f}  "
                f"val_mse={val_mse:.5f}  val_mae={val_mae:.4f}")
        if val_mae < target_mae:
            early_stopped_at = ep
            log(f"  BC early-stop: val_mae={val_mae:.4f} < {target_mae} at epoch {ep}")
            break
    bc_train_seconds = time.perf_counter() - bc_t0
    bc_state_dict = {kk: v.clone() for kk, v in bc_net.state_dict().items()}
    torch.save(bc_state_dict, cell_dir / "bc_defender.pt")
    log(f"BC done in {bc_train_seconds:.1f}s, final val_mae={bc_history[-1]['val_mae']:.4f}")

    # BC sanity animations (heuristic vs BC NN)
    bc_anim_dir = Path(out_root) / "animations" / f"k{k}_bc"
    bc_anim_dir.mkdir(parents=True, exist_ok=True)
    bc_caps = 0
    for i in range(settings["n_anim"]):
        env_h = PursuitEvasionEnv(
            batch_size=1, k=k, sigma=sigma, p=p,
            seed=seed + 5000 + i, **env_kwargs,
        )
        heur_att = HeuristicAttacker(target=env_h.target, v_attacker=env_h.v_attacker)
        heur_def = HeuristicOrbitalDefender(
            v_defender=env_h.v_defender, target_pos=env_kwargs["target_pos"],
            orbit_radius=settings["orbit_radius"], capture_radius=env_h.capture_radius,
        )
        h_h = run_episode(env_h, heur_att, heur_def)
        save_animation(h_h, env_h.metadata(),
                       bc_anim_dir / f"compare_{i}_heuristic.mp4")
        env_n = PursuitEvasionEnv(
            batch_size=1, k=k, sigma=sigma, p=p,
            seed=seed + 5000 + i, **env_kwargs,
        )
        heur_att2 = HeuristicAttacker(target=env_n.target, v_attacker=env_n.v_attacker)
        bc_def_pol = CentralizedDefender(
            bc_net, env_n.v_defender, deterministic=True, sort_by_distance=True,
        )
        h_n = run_episode(env_n, heur_att2, bc_def_pol)
        save_animation(h_n, env_n.metadata(),
                       bc_anim_dir / f"compare_{i}_bc_nn.mp4")
        if h_n["outcome"] == 2:
            bc_caps += 1
    log(f"BC sanity: NN captures {bc_caps}/{settings['n_anim']} (vs HeuristicAttacker)")

    # ---------- Stage 2: RL fine-tune ----------
    rl_t0 = time.perf_counter()
    log(f"=== stage 2: RL fine-tune (alternating, BC warm-start) ===")
    cfg = PPOConfig(
        n_steps=settings["n_steps"],
        n_epochs=4, minibatch_size=512,
        lr=settings["rl_lr"],
        gamma=0.99, gae_lambda=0.95, clip_coef=0.2,
        ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5,
    )
    res = alternating_train_centralized(
        k=k, sigma=sigma, p=p, cfg=cfg,
        n_envs=settings["n_envs"],
        attacker_warmup_iters=settings["rl_warmup"],
        defender_iters=settings["rl_def"],
        attacker_iters=settings["rl_att"],
        n_alternations=settings["rl_alts"],
        seed=seed,
        central_hidden=settings["hidden"],
        central_n_layers=settings["n_layers"],
        sort_by_distance=True,
        init_central_defender_state_dict=bc_state_dict,
        use_attacker_population=settings["use_population"],
        env_kwargs=env_kwargs,
        save_dir=cell_dir,  # writes attacker.pt + central_defender.pt
        log_fn=log,
    )
    rl_train_seconds = time.perf_counter() - rl_t0
    log(f"RL done in {rl_train_seconds:.1f}s")

    # ---------- Eval ----------
    env_meta = res["env_meta"]
    a_pol = NeuralAttacker(res["attacker_net"], env_meta["v_attacker"], deterministic=True)
    d_pol = CentralizedDefender(
        res["central_defender_net"], env_meta["v_defender"],
        deterministic=True, sort_by_distance=True,
    )
    eval_main = evaluate(
        k=k, sigma=sigma, p=p,
        attacker_policy=a_pol, defender_policy=d_pol,
        n_episodes=settings["n_eval"], seed=seed + 7919, env_kwargs=env_kwargs,
    )
    log(f"eval_main (trained_att vs trained_def): {eval_main}")

    heur_att = HeuristicAttacker(
        target=np.asarray(env_meta["target_pos"], dtype=np.float32),
        v_attacker=env_meta["v_attacker"],
    )
    eval_vs_heur_att = evaluate(
        k=k, sigma=sigma, p=p,
        attacker_policy=heur_att, defender_policy=d_pol,
        n_episodes=settings["n_eval"], seed=seed + 7921, env_kwargs=env_kwargs,
    )
    log(f"eval_heur_att_vs_trained_def: {eval_vs_heur_att}")

    # RL animations
    rl_anim_dir = Path(out_root) / "animations" / f"k{k}_rl"
    rl_anim_dir.mkdir(parents=True, exist_ok=True)
    for i in range(settings["n_anim"]):
        env_anim = PursuitEvasionEnv(
            batch_size=1, k=k, sigma=sigma, p=p,
            seed=seed + 1000 + i, **env_kwargs,
        )
        a_anim = NeuralAttacker(res["attacker_net"], env_anim.v_attacker, deterministic=False)
        d_anim = CentralizedDefender(
            res["central_defender_net"], env_anim.v_defender,
            deterministic=False, sort_by_distance=True,
        )
        h = run_episode(env_anim, a_anim, d_anim)
        save_animation(h, env_anim.metadata(), rl_anim_dir / f"rl_ep{i}.mp4")

    # Persist
    log_path = cell_dir / "train.log"
    log_path.write_text("\n".join(log_lines))
    metrics_path = cell_dir / "rl_metrics.json"
    metrics_path.write_text(json.dumps(_jsonable(res["metrics"]), indent=2))

    rl_phase_summaries: list[dict] = []
    warmup_ms = res["metrics"].get("warmup") or []
    if warmup_ms:
        rl_phase_summaries.append({
            "name": "warmup", "n": len(warmup_ms),
            "first": warmup_ms[0], "last": warmup_ms[-1],
        })
    for ri, rd in enumerate(res["metrics"].get("rounds", [])):
        for side in ("defender", "attacker"):
            ms = rd.get(side) or []
            if ms:
                rl_phase_summaries.append({
                    "name": f"round_{ri+1}_{side}", "n": len(ms),
                    "first": ms[0], "last": ms[-1],
                })

    return {
        "cell_id": cell_id,
        "k": k, "sigma": sigma, "p": p,
        "bc": {
            "n_samples": int(n_total),
            "train_seconds": bc_train_seconds,
            "epochs_run": len(bc_history),
            "early_stopped_at": early_stopped_at,
            "final_val_mae": bc_history[-1]["val_mae"],
            "final_val_mse": bc_history[-1]["val_mse"],
            "history": bc_history,
            "sanity_captures": bc_caps,
            "sanity_n": settings["n_anim"],
        },
        "rl": {
            "train_seconds": rl_train_seconds,
            "schedule": res.get("schedule"),
            "phase_summaries": rl_phase_summaries,
            "metrics_path": str(metrics_path),
            "eval_main": eval_main,
            "eval_heur_att_vs_trained_def": eval_vs_heur_att,
        },
        "log_path": str(log_path),
        "settings": dict(settings),
    }


# ============================================================================
# Driver
# ============================================================================


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k-values", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6])
    parser.add_argument("--sigma", type=float, default=0.8)
    parser.add_argument("--p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--orbit-radius", type=float, default=0.20)
    parser.add_argument("--out", type=str, default=str(_REPO_ROOT / "outputs_bc_then_rl"))
    parser.add_argument("--workers", type=int, default=0,
                        help="0 = auto = min(n_cells, cpu_count)")
    parser.add_argument("--threads-per-worker", type=int, default=2)

    # net
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=3)

    # BC
    parser.add_argument("--bc-target-mae", type=float, default=0.06)
    parser.add_argument("--bc-max-epochs", type=int, default=60)
    parser.add_argument("--bc-n-envs", type=int, default=128)
    parser.add_argument("--bc-n-rollouts", type=int, default=80)
    parser.add_argument("--bc-rollout-steps", type=int, default=32)
    parser.add_argument("--bc-batch-size", type=int, default=1024)
    parser.add_argument("--bc-lr", type=float, default=5e-4)

    # RL
    parser.add_argument("--n-envs", type=int, default=256)
    parser.add_argument("--n-steps", type=int, default=128)
    parser.add_argument("--rl-warmup", type=int, default=50)
    parser.add_argument("--rl-att", type=int, default=100)
    parser.add_argument("--rl-def", type=int, default=100)
    parser.add_argument("--rl-alts", type=int, default=3)
    parser.add_argument("--rl-lr", type=float, default=3e-4)

    # eval / animation
    parser.add_argument("--n-eval", type=int, default=2000)
    parser.add_argument("--n-anim", type=int, default=3)

    # extra defender shaping (defaults 0 → existing behaviour)
    parser.add_argument("--c-progress-target", type=float, default=0.0)
    parser.add_argument("--c-chase", type=float, default=0.0)
    parser.add_argument("--c-block", type=float, default=0.0)
    parser.add_argument("--c-block-threshold", type=float, default=0.05)
    parser.add_argument("--c-pressure", type=float, default=0.0)
    parser.add_argument("--c-pressure-radius", type=float, default=0.15)
    parser.add_argument("--c-cluster", type=float, default=0.0)
    parser.add_argument("--c-cluster-scale", type=float, default=0.10)
    parser.add_argument("--c-timeout", type=float, default=0.0)
    parser.add_argument("--c-line-of-sight", type=float, default=0.0)
    parser.add_argument("--c-los-threshold", type=float, default=0.05)

    # population training
    parser.add_argument("--use-population", action="store_true",
                        help="During each defender phase, mix in heuristic attacker + "
                        "previous attacker snapshots in addition to the current attacker.")

    args = parser.parse_args()
    os.environ["WORKER_THREADS"] = str(args.threads_per_worker)

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    env_kwargs = {
        "dt": 0.02, "max_steps": 500, "capture_radius": 0.03,
        "target_radius": 0.10, "target_pos": (0.5, 0.5), "v_attacker": 1.0,
        "inner_radius": 0.15, "outer_radius": 0.35, "shaping": 0.01,
        "c_progress_target": args.c_progress_target,
        "c_chase": args.c_chase,
        "c_block": args.c_block,
        "c_block_threshold": args.c_block_threshold,
        "c_pressure": args.c_pressure,
        "c_pressure_radius": args.c_pressure_radius,
        "c_cluster": args.c_cluster,
        "c_cluster_scale": args.c_cluster_scale,
        "c_timeout": args.c_timeout,
        "c_line_of_sight": args.c_line_of_sight,
        "c_los_threshold": args.c_los_threshold,
    }
    settings = {
        "seed": args.seed,
        "hidden": args.hidden, "n_layers": args.n_layers,
        "orbit_radius": args.orbit_radius,
        "bc_target_mae": args.bc_target_mae,
        "bc_max_epochs": args.bc_max_epochs,
        "bc_n_envs": args.bc_n_envs,
        "bc_n_rollouts": args.bc_n_rollouts,
        "bc_rollout_steps": args.bc_rollout_steps,
        "bc_batch_size": args.bc_batch_size,
        "bc_lr": args.bc_lr,
        "n_envs": args.n_envs, "n_steps": args.n_steps,
        "rl_warmup": args.rl_warmup,
        "rl_att": args.rl_att, "rl_def": args.rl_def,
        "rl_alts": args.rl_alts, "rl_lr": args.rl_lr,
        "use_population": args.use_population,
        "n_eval": args.n_eval, "n_anim": args.n_anim,
    }

    cell_args = [
        (i, k, args.sigma, args.p, settings, env_kwargs, str(out_root))
        for i, k in enumerate(args.k_values)
    ]
    n_workers = args.workers if args.workers > 0 else min(len(cell_args), os.cpu_count() or 1)
    print(f"=== bc_then_rl  k_values={args.k_values}  sigma={args.sigma} p={args.p} ===")
    print(f"  bc: target_mae={args.bc_target_mae}  max_epochs={args.bc_max_epochs}")
    print(f"  rl: warmup={args.rl_warmup} alt={args.rl_alts}x"
          f"(def={args.rl_def}, att={args.rl_att})")
    print(f"  n_workers={n_workers}  threads_per_worker={args.threads_per_worker}")

    t0 = time.perf_counter()
    if n_workers == 1:
        results = [_cell(a) for a in cell_args]
    else:
        with mp.Pool(n_workers) as pool:
            results = []
            for r in pool.imap_unordered(_cell, cell_args):
                results.append(r)
                em = r["rl"]["eval_main"]
                ci = em["ci_95"]
                bc = r["bc"]
                print(
                    f"  [k={r['k']}] BC: ep={bc['epochs_run']} mae={bc['final_val_mae']:.4f} "
                    f"({bc['train_seconds']:.0f}s)  |  "
                    f"RL: succ={em['attacker_success_rate']:.3f} CI=[{ci[0]:.3f},{ci[1]:.3f}] "
                    f"({r['rl']['train_seconds']:.0f}s)",
                    flush=True,
                )
    elapsed = time.perf_counter() - t0
    print(f"sweep wall: {elapsed:.0f}s")

    results.sort(key=lambda r: r["k"])
    out_json = out_root / "results.json"
    out_json.write_text(json.dumps({
        "experiment": "bc_then_rl_centralized",
        "fixed": {"sigma": args.sigma, "p": args.p, "seed": args.seed},
        "k_values": args.k_values,
        "settings": settings,
        "git": _git_info(_REPO_ROOT),
        "cells": [_jsonable(r) for r in results],
    }, indent=2))
    print(f"results → {out_json}")


if __name__ == "__main__":
    main()
