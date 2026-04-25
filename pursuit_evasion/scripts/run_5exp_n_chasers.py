"""5-experiment pipeline at fixed k=5, sweeping over n_chasers ∈ {1..5}.

For each n_chasers:
  Stage 1 (BC):   imitate HeuristicOrbitalDefender(n_chasers) into a
                  CentralizedDefenderNet (sort_by_distance=True).
  Stage 2 (att1): train RL attacker vs the BC-init defender (frozen).
                  Save mid-snapshot at iter ≈ stage2_iters // 2.
  Stage 3 (def):  train RL defender vs PopulationAttacker[heuristic_att,
                  attacker_mid, attacker_final].
  Stage 4 (att2): continue training the attacker vs the new defender.

Final eval (n=2000 each, deterministic):
  - Heuristic def       vs Heuristic att
  - Heuristic def       vs Trained att (post-Stage 4)
  - Trained NN def      vs Heuristic att
  - Trained NN def      vs Trained att

Saves results.json, animations, checkpoints, plus a 5×4 bar plot.

Usage::
  python scripts/run_5exp_n_chasers.py --workers 5 --threads-per-worker 3
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
# Per-cell worker
# ============================================================================


def _cell(args: tuple) -> dict:
    cell_id, n_chasers, k, sigma, p, settings, env_kwargs, out_root = args
    threads = int(os.environ.get("WORKER_THREADS", "1"))
    os.environ.setdefault("OMP_NUM_THREADS", str(threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(threads))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(threads))
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", str(threads))
    import torch
    import torch.nn.functional as Fnn
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
        NeuralAttackerNet,
        PopulationAttacker,
    )
    from src.rollout import evaluate
    from src.train import (
        PPOConfig,
        train_attacker,
        train_defender_centralized,
    )

    cell_dir = Path(out_root) / "checkpoints" / f"n{n_chasers}"
    cell_dir.mkdir(parents=True, exist_ok=True)
    log_lines: list[str] = []

    def log(msg: str) -> None:
        log_lines.append(msg)
        print(f"[n={n_chasers}] {msg}", flush=True)

    seed = settings["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    cfg = PPOConfig(
        n_steps=settings["n_steps"],
        n_epochs=4, minibatch_size=512,
        lr=settings["rl_lr"],
        gamma=0.99, gae_lambda=0.95, clip_coef=0.2,
        ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5,
    )

    # ---- Stage 1: BC ----
    log(f"=== stage 1: BC imitate HeuristicOrbitalDefender(n_chasers={n_chasers}) ===")
    bc_t0 = time.perf_counter()
    bc_env = PursuitEvasionEnv(
        batch_size=settings["bc_n_envs"], k=k, sigma=sigma, p=p,
        seed=seed, **env_kwargs,
    )
    heur_def = HeuristicOrbitalDefender(
        v_defender=bc_env.v_defender,
        target_pos=env_kwargs["target_pos"],
        orbit_radius=settings["orbit_radius"],
        capture_radius=bc_env.capture_radius,
        n_chasers=n_chasers,
    )
    n_obs = 4 + 4 * k
    n_act = 2 * k
    total_samples = settings["bc_n_envs"] * settings["bc_n_rollouts"] * settings["bc_rollout_steps"]
    obs_arr = np.empty((total_samples, n_obs), dtype=np.float32)
    act_arr = np.empty((total_samples, n_act), dtype=np.float32)
    write = 0
    for _ in range(settings["bc_n_rollouts"]):
        bc_env.reset()
        for _ in range(settings["bc_rollout_steps"]):
            obs_s, perm = CentralizedDefenderNet.build_sorted_obs(bc_env)
            d_act = heur_def.act(bc_env)
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

    bc_net = CentralizedDefenderNet(
        k=k, hidden=settings["hidden"], n_layers=settings["n_layers"]
    )
    bc_opt = optim.Adam(bc_net.parameters(), lr=settings["bc_lr"])
    n_train = obs_train_t.shape[0]
    bc_history: list[dict] = []
    for ep in range(settings["bc_max_epochs"]):
        idxs = np.random.permutation(n_train)
        for start in range(0, n_train, settings["bc_batch_size"]):
            mb = idxs[start : start + settings["bc_batch_size"]]
            mean, _, _ = bc_net(obs_train_t[mb])
            loss = Fnn.mse_loss(mean, act_train_t[mb])
            bc_opt.zero_grad(); loss.backward(); bc_opt.step()
        with torch.no_grad():
            v_mean, _, _ = bc_net(obs_val_t)
            val_mae = (v_mean - act_val_t).abs().mean().item()
            val_mse = Fnn.mse_loss(v_mean, act_val_t).item()
        bc_history.append({"epoch": ep, "val_mae": val_mae, "val_mse": val_mse})
        if (ep % 5 == 0) or ep == settings["bc_max_epochs"] - 1:
            log(f"  BC ep {ep:3d}  val_mae={val_mae:.4f}")
        if val_mae < settings["bc_target_mae"]:
            log(f"  BC early-stop at ep {ep}  val_mae={val_mae:.4f}")
            break
    torch.save(bc_net.state_dict(), cell_dir / "bc_defender.pt")
    bc_seconds = time.perf_counter() - bc_t0
    log(f"BC done in {bc_seconds:.1f}s  val_mae={bc_history[-1]['val_mae']:.4f}")

    # Build a working centralized defender net seeded from BC for the RL stages.
    central_def_net = CentralizedDefenderNet(
        k=k, hidden=settings["hidden"], n_layers=settings["n_layers"]
    )
    central_def_net.load_state_dict(bc_net.state_dict())

    # ---- Stage 2: train attacker vs BC defender, save mid + final ----
    log(f"=== stage 2: train attacker vs BC defender ({settings['stage2_att_iters']} iters) ===")
    s2_t0 = time.perf_counter()
    rl_env = PursuitEvasionEnv(
        batch_size=settings["n_envs"], k=k, sigma=sigma, p=p,
        seed=seed, **env_kwargs,
    )
    attacker_net = NeuralAttackerNet(rl_env.attacker_obs_dim)
    bc_def_frozen = CentralizedDefender(
        central_def_net, rl_env.v_defender, deterministic=False, sort_by_distance=True,
    )
    mid_iter = max(0, settings["stage2_att_iters"] // 2 - 1)
    a_log = train_attacker(
        rl_env, attacker_net, bc_def_frozen,
        n_iters=settings["stage2_att_iters"], cfg=cfg, log_fn=log,
        snapshot_iters=[mid_iter],
    )
    attacker_mid_state = a_log["snapshots"][0] if a_log["snapshots"] else None
    attacker_final_state = {kk: v.detach().cpu().clone() for kk, v in attacker_net.state_dict().items()}
    if attacker_mid_state is not None:
        torch.save(attacker_mid_state, cell_dir / "attacker_stage2_mid.pt")
    torch.save(attacker_final_state, cell_dir / "attacker_stage2_final.pt")
    s2_seconds = time.perf_counter() - s2_t0
    log(f"stage 2 done in {s2_seconds:.1f}s")

    # ---- Stage 3: train defender vs population [heur, att_mid, att_final] ----
    log(f"=== stage 3: train defender vs population ({settings['stage3_def_iters']} iters) ===")
    s3_t0 = time.perf_counter()
    rl_env_d = PursuitEvasionEnv(
        batch_size=settings["n_envs"], k=k, sigma=sigma, p=p,
        seed=seed + 1, **env_kwargs,
    )
    pop_policies = [
        HeuristicAttacker(target=rl_env_d.target, v_attacker=rl_env_d.v_attacker)
    ]
    if attacker_mid_state is not None:
        mid_net = NeuralAttackerNet(rl_env_d.attacker_obs_dim)
        mid_net.load_state_dict(attacker_mid_state)
        mid_net.eval()
        pop_policies.append(NeuralAttacker(mid_net, rl_env_d.v_attacker, deterministic=True))
    final_net_for_pop = NeuralAttackerNet(rl_env_d.attacker_obs_dim)
    final_net_for_pop.load_state_dict(attacker_final_state)
    final_net_for_pop.eval()
    pop_policies.append(NeuralAttacker(final_net_for_pop, rl_env_d.v_attacker, deterministic=False))
    pop_policies.append(NeuralAttacker(final_net_for_pop, rl_env_d.v_attacker, deterministic=True))
    log(f"  population size = {len(pop_policies)} (heur + att_mid + att_final×2)")
    population = PopulationAttacker(pop_policies, rl_env_d.B, rng_seed=seed + 100)
    train_defender_centralized(
        rl_env_d, central_def_net, pop_policies[0],
        n_iters=settings["stage3_def_iters"], cfg=cfg, log_fn=log,
        sort_by_distance=True, population=population,
    )
    torch.save(central_def_net.state_dict(), cell_dir / "central_defender.pt")
    s3_seconds = time.perf_counter() - s3_t0
    log(f"stage 3 done in {s3_seconds:.1f}s")

    # ---- Stage 4: continue attacker training vs new defender ----
    log(f"=== stage 4: train attacker vs new defender ({settings['stage4_att_iters']} iters) ===")
    s4_t0 = time.perf_counter()
    rl_env_a2 = PursuitEvasionEnv(
        batch_size=settings["n_envs"], k=k, sigma=sigma, p=p,
        seed=seed + 2, **env_kwargs,
    )
    new_def_frozen = CentralizedDefender(
        central_def_net, rl_env_a2.v_defender, deterministic=False, sort_by_distance=True,
    )
    train_attacker(
        rl_env_a2, attacker_net, new_def_frozen,
        n_iters=settings["stage4_att_iters"], cfg=cfg, log_fn=log,
    )
    torch.save(attacker_net.state_dict(), cell_dir / "attacker.pt")
    s4_seconds = time.perf_counter() - s4_t0
    log(f"stage 4 done in {s4_seconds:.1f}s")

    # ---- Final eval: 4 matchups ----
    log(f"=== final eval (n={settings['n_eval']} each) ===")
    env_meta = rl_env_a2.metadata()
    heur_att_pol = HeuristicAttacker(
        target=np.asarray(env_meta["target_pos"], dtype=np.float32),
        v_attacker=env_meta["v_attacker"],
    )
    trained_att_pol = NeuralAttacker(attacker_net, env_meta["v_attacker"], deterministic=True)
    nn_def_pol = CentralizedDefender(
        central_def_net, env_meta["v_defender"], deterministic=True, sort_by_distance=True,
    )
    eval_kwargs = dict(k=k, sigma=sigma, p=p, n_episodes=settings["n_eval"], env_kwargs=env_kwargs)

    e_hh = evaluate(attacker_policy=heur_att_pol,    defender_policy=heur_def, seed=seed + 7919, **eval_kwargs)
    e_ht = evaluate(attacker_policy=trained_att_pol, defender_policy=heur_def, seed=seed + 7920, **eval_kwargs)
    e_nh = evaluate(attacker_policy=heur_att_pol,    defender_policy=nn_def_pol, seed=seed + 7921, **eval_kwargs)
    e_nt = evaluate(attacker_policy=trained_att_pol, defender_policy=nn_def_pol, seed=seed + 7922, **eval_kwargs)
    log(f"heur_def vs heur_att:    succ={e_hh['attacker_success_rate']:.3f}")
    log(f"heur_def vs trained_att: succ={e_ht['attacker_success_rate']:.3f}")
    log(f"NN_def   vs heur_att:    succ={e_nh['attacker_success_rate']:.3f}")
    log(f"NN_def   vs trained_att: succ={e_nt['attacker_success_rate']:.3f}")

    # ---- Animations: 2 of NN-def vs trained-att, 2 of NN-def vs heur-att ----
    anim_dir = Path(out_root) / "animations" / f"n{n_chasers}"
    anim_dir.mkdir(parents=True, exist_ok=True)
    for i in range(settings["n_anim"]):
        env_anim = PursuitEvasionEnv(
            batch_size=1, k=k, sigma=sigma, p=p, seed=seed + 1000 + i, **env_kwargs,
        )
        a_anim = NeuralAttacker(attacker_net, env_anim.v_attacker, deterministic=False)
        d_anim = CentralizedDefender(
            central_def_net, env_anim.v_defender, deterministic=False, sort_by_distance=True,
        )
        h = run_episode(env_anim, a_anim, d_anim)
        save_animation(h, env_anim.metadata(), anim_dir / f"nn_vs_trained_att_ep{i}.mp4")
    for i in range(settings["n_anim"]):
        env_anim = PursuitEvasionEnv(
            batch_size=1, k=k, sigma=sigma, p=p, seed=seed + 2000 + i, **env_kwargs,
        )
        heur_a2 = HeuristicAttacker(target=env_anim.target, v_attacker=env_anim.v_attacker)
        d_anim = CentralizedDefender(
            central_def_net, env_anim.v_defender, deterministic=True, sort_by_distance=True,
        )
        h = run_episode(env_anim, heur_a2, d_anim)
        save_animation(h, env_anim.metadata(), anim_dir / f"nn_vs_heur_att_ep{i}.mp4")

    log_path = cell_dir / "train.log"
    log_path.write_text("\n".join(log_lines))

    return {
        "cell_id": cell_id,
        "n_chasers": n_chasers,
        "k": k, "sigma": sigma, "p": p,
        "bc": {
            "n_samples": int(n_total),
            "epochs_run": len(bc_history),
            "final_val_mae": bc_history[-1]["val_mae"],
            "history": bc_history,
            "seconds": bc_seconds,
        },
        "stage_seconds": {
            "stage2": s2_seconds,
            "stage3": s3_seconds,
            "stage4": s4_seconds,
        },
        "eval": {
            "heur_def_vs_heur_att": e_hh,
            "heur_def_vs_trained_att": e_ht,
            "nn_def_vs_heur_att": e_nh,
            "nn_def_vs_trained_att": e_nt,
        },
        "log_path": str(log_path),
        "settings": dict(settings),
    }


# ============================================================================
# Driver + plot
# ============================================================================


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--sigma", type=float, default=0.8)
    parser.add_argument("--p", type=float, default=1.0)
    parser.add_argument("--n-chasers-list", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--orbit-radius", type=float, default=0.20)
    parser.add_argument("--out", type=str,
                        default=str(_REPO_ROOT / "outputs_5exp_n_chasers"))
    parser.add_argument("--workers", type=int, default=0,
                        help="0 = auto = min(n_cells, cpu_count)")
    parser.add_argument("--threads-per-worker", type=int, default=2)

    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--bc-target-mae", type=float, default=0.06)
    parser.add_argument("--bc-max-epochs", type=int, default=60)
    parser.add_argument("--bc-n-envs", type=int, default=128)
    parser.add_argument("--bc-n-rollouts", type=int, default=80)
    parser.add_argument("--bc-rollout-steps", type=int, default=32)
    parser.add_argument("--bc-batch-size", type=int, default=1024)
    parser.add_argument("--bc-lr", type=float, default=5e-4)

    parser.add_argument("--n-envs", type=int, default=256)
    parser.add_argument("--n-steps", type=int, default=128)
    parser.add_argument("--stage2-att-iters", type=int, default=200)
    parser.add_argument("--stage3-def-iters", type=int, default=200)
    parser.add_argument("--stage4-att-iters", type=int, default=200)
    parser.add_argument("--rl-lr", type=float, default=3e-4)

    parser.add_argument("--n-eval", type=int, default=2000)
    parser.add_argument("--n-anim", type=int, default=2)

    args = parser.parse_args()
    os.environ["WORKER_THREADS"] = str(args.threads_per_worker)

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    env_kwargs = {
        "dt": 0.02, "max_steps": 500, "capture_radius": 0.03,
        "target_radius": 0.10, "target_pos": (0.5, 0.5), "v_attacker": 1.0,
        "inner_radius": 0.15, "outer_radius": 0.35, "shaping": 0.01,
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
        "stage2_att_iters": args.stage2_att_iters,
        "stage3_def_iters": args.stage3_def_iters,
        "stage4_att_iters": args.stage4_att_iters,
        "rl_lr": args.rl_lr,
        "n_eval": args.n_eval, "n_anim": args.n_anim,
    }

    cell_args = [
        (i, n, args.k, args.sigma, args.p, settings, env_kwargs, str(out_root))
        for i, n in enumerate(args.n_chasers_list)
    ]
    n_workers = args.workers if args.workers > 0 else min(len(cell_args), os.cpu_count() or 1)
    print(f"=== 5exp_n_chasers  k={args.k}  sigma={args.sigma} p={args.p} ===")
    print(f"  n_chasers list: {args.n_chasers_list}")
    print(f"  schedule per cell: BC + stage2={args.stage2_att_iters} + "
          f"stage3={args.stage3_def_iters} + stage4={args.stage4_att_iters}")
    print(f"  n_workers={n_workers}  threads_per_worker={args.threads_per_worker}")

    t0 = time.perf_counter()
    if n_workers == 1:
        results = [_cell(a) for a in cell_args]
    else:
        with mp.Pool(n_workers) as pool:
            results = []
            for r in pool.imap_unordered(_cell, cell_args):
                results.append(r)
                ev = r["eval"]
                print(
                    f"  [n={r['n_chasers']}]  "
                    f"heur_def-vs-heur_att={ev['heur_def_vs_heur_att']['attacker_success_rate']:.3f}  "
                    f"heur_def-vs-trained={ev['heur_def_vs_trained_att']['attacker_success_rate']:.3f}  "
                    f"NN_def-vs-heur={ev['nn_def_vs_heur_att']['attacker_success_rate']:.3f}  "
                    f"NN_def-vs-trained={ev['nn_def_vs_trained_att']['attacker_success_rate']:.3f}",
                    flush=True,
                )
    elapsed = time.perf_counter() - t0
    print(f"sweep wall: {elapsed:.0f}s")

    results.sort(key=lambda r: r["n_chasers"])
    out_json = out_root / "results.json"
    out_json.write_text(json.dumps({
        "experiment": "5exp_n_chasers_centralized_BC_then_RL_then_RL_again",
        "fixed": {"k": args.k, "sigma": args.sigma, "p": args.p, "seed": args.seed},
        "n_chasers_list": args.n_chasers_list,
        "settings": settings,
        "git": _git_info(_REPO_ROOT),
        "cells": [_jsonable(r) for r in results],
    }, indent=2))
    print(f"results → {out_json}")

    # ---- Plot 5×4 grid ----
    plots_dir = out_root / "plots"
    plots_dir.mkdir(exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_chasers_axis = [r["n_chasers"] for r in results]
    matchups = [
        ("heur_def_vs_heur_att", "Heur def vs Heur att", "tab:gray"),
        ("heur_def_vs_trained_att", "Heur def vs Trained att", "tab:purple"),
        ("nn_def_vs_heur_att", "NN def vs Heur att", "tab:orange"),
        ("nn_def_vs_trained_att", "NN def vs Trained att", "tab:blue"),
    ]
    fig, ax = plt.subplots(figsize=(12, 5.5))
    x = np.arange(len(n_chasers_axis))
    w = 0.2
    for i, (key, label, color) in enumerate(matchups):
        vals = [r["eval"][key]["attacker_success_rate"] for r in results]
        ax.bar(x + (i - 1.5) * w, vals, w, label=label, color=color)
        for xi, v in zip(x + (i - 1.5) * w, vals):
            ax.text(xi, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"n_chasers={n}" for n in n_chasers_axis])
    ax.set_ylabel("attacker success rate (lower = stronger defender)")
    ax.set_title(
        f"5-experiment sweep over n_chasers  (k={args.k}, σ={args.sigma}, p={args.p}, "
        f"n_eval={args.n_eval})"
    )
    ax.legend(loc="best")
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(plots_dir / "n_chasers_sweep.png", dpi=130)
    print(f"plot → {plots_dir / 'n_chasers_sweep.png'}")


if __name__ == "__main__":
    main()
