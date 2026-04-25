"""Single-cell alternating centralized-defender trainer, optimized for Apple
Silicon (M2 MacBook Air) with periodic animation snapshots during training.

Independent of ``run_experiment.py`` and ``run_central.py`` — does not use
multiprocessing (one cell, no benefit), drives the train_* functions in
chunks so we can render an animation between chunks. Saves a timelapse of
``animations/anim_iter{N:04d}_{phase}.mp4`` files plus the usual checkpoints.

Defaults:
  k=4, sigma=0.3, p=0.7, seed=0, anim_every=25 PPO iters, 4 BLAS threads.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# --- M2 BLAS knobs: must be set BEFORE torch import ---
_DEFAULT_THREADS = 4
_threads_env = os.environ.get("M2_THREADS", str(_DEFAULT_THREADS))
os.environ.setdefault("OMP_NUM_THREADS", _threads_env)
os.environ.setdefault("MKL_NUM_THREADS", _threads_env)
os.environ.setdefault("OPENBLAS_NUM_THREADS", _threads_env)
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", _threads_env)  # Apple Accelerate
os.environ.setdefault("NUMEXPR_NUM_THREADS", _threads_env)

import numpy as np
import torch

torch.set_num_threads(int(_threads_env))

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.animate import run_episode, save_animation
from src.env import PursuitEvasionEnv
from src.policies import (
    CentralizedDefender,
    CentralizedDefenderNet,
    HeuristicDefender,
    HeuristicOrbitalDefender,
    NeuralAttacker,
    NeuralAttackerNet,
)
from src.train import PPOConfig, train_attacker, train_defender_centralized


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--sigma", type=float, default=0.3)
    parser.add_argument("--p", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default=str(_REPO_ROOT / "outputs_local_m2"))
    parser.add_argument("--anim-every", type=int, default=25,
                        help="Render an animation every N PPO iters (within a phase).")
    parser.add_argument("--threads", type=int, default=int(_threads_env),
                        help="BLAS / torch threads. M2 Air has 4 P-cores + 4 E-cores; "
                        "4 is usually the sweet spot for small MLPs.")
    parser.add_argument("--n-envs", type=int, default=256)
    parser.add_argument("--n-steps", type=int, default=128)
    parser.add_argument("--attacker-warmup-iters", type=int, default=50)
    parser.add_argument("--attacker-iters", type=int, default=100)
    parser.add_argument("--defender-iters", type=int, default=100)
    parser.add_argument("--n-alternations", type=int, default=3)
    parser.add_argument("--central-hidden", type=int, default=256)
    parser.add_argument("--central-n-layers", type=int, default=3)
    parser.add_argument("--sort-defenders", action="store_true",
                        help="Reorder defenders by distance to attacker before each "
                        "forward pass (slot 0 = closest). Makes the centralized "
                        "policy permutation-invariant.")
    parser.add_argument("--bc-warm-start", action="store_true",
                        help="Before RL, train the centralized defender via BC "
                        "to imitate HeuristicOrbitalDefender. Then use BC defender "
                        "as the attacker's warmup opponent.")
    parser.add_argument("--bc-target-mae", type=float, default=0.06)
    parser.add_argument("--bc-max-epochs", type=int, default=60)
    parser.add_argument("--bc-n-rollouts", type=int, default=80)
    parser.add_argument("--bc-rollout-steps", type=int, default=32)
    parser.add_argument("--bc-batch-size", type=int, default=1024)
    parser.add_argument("--bc-lr", type=float, default=5e-4)
    parser.add_argument("--orbit-radius", type=float, default=0.20,
                        help="Guard ring radius for HeuristicOrbitalDefender (BC only).")
    args = parser.parse_args()

    if args.threads != int(_threads_env):
        torch.set_num_threads(args.threads)
        for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
            os.environ[var] = str(args.threads)

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    anim_dir = out_root / "animations"
    anim_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_root / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== M2 single-cell centralized alternating ===")
    print(f"  k={args.k}  sigma={args.sigma}  p={args.p}  seed={args.seed}")
    print(f"  threads={torch.get_num_threads()}  n_envs={args.n_envs}  n_steps={args.n_steps}")
    print(f"  schedule: warmup={args.attacker_warmup_iters}  alt={args.n_alternations}x"
          f"(def={args.defender_iters}, att={args.attacker_iters})")
    print(f"  anim_every={args.anim_every}  out={out_root}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = PPOConfig(n_steps=args.n_steps)

    env_kwargs = {
        "dt": 0.02, "max_steps": 500, "capture_radius": 0.03,
        "target_radius": 0.10, "target_pos": (0.5, 0.5), "v_attacker": 1.0,
        "inner_radius": 0.15, "outer_radius": 0.35, "shaping": 0.01,
    }

    env = PursuitEvasionEnv(
        batch_size=args.n_envs, k=args.k, sigma=args.sigma, p=args.p,
        seed=args.seed, **env_kwargs,
    )

    attacker_net = NeuralAttackerNet(env.attacker_obs_dim)
    defender_net = CentralizedDefenderNet(
        k=args.k, hidden=args.central_hidden, n_layers=args.central_n_layers,
    )
    a_opt = torch.optim.Adam(attacker_net.parameters(), lr=cfg.lr, eps=1e-5)
    d_opt = torch.optim.Adam(defender_net.parameters(), lr=cfg.lr, eps=1e-5)

    state = {"cumulative_iters": 0, "anim_idx": 0, "t_start": time.perf_counter()}

    def snapshot(label: str) -> None:
        env_anim = PursuitEvasionEnv(
            batch_size=1, k=args.k, sigma=args.sigma, p=args.p,
            seed=args.seed + 31337, **env_kwargs,
        )
        a_pol = NeuralAttacker(attacker_net, env_anim.v_attacker, deterministic=False)
        d_pol = CentralizedDefender(
            defender_net, env_anim.v_defender, deterministic=False,
            sort_by_distance=args.sort_defenders,
        )
        h = run_episode(env_anim, a_pol, d_pol)
        out_anim = (
            anim_dir
            / f"anim_{state['anim_idx']:03d}_iter{state['cumulative_iters']:04d}_{label}.mp4"
        )
        save_animation(h, env_anim.metadata(), out_anim)
        elapsed = time.perf_counter() - state["t_start"]
        print(
            f"  → anim #{state['anim_idx']:03d} iter={state['cumulative_iters']:4d} "
            f"{label}: {h['steps']} steps outcome={h['outcome']} "
            f"[wall {elapsed:.1f}s]"
        )
        state["anim_idx"] += 1

    def train_chunked(side: str, n_iters: int, opponent, label: str) -> None:
        chunk = max(1, args.anim_every)
        done = 0
        while done < n_iters:
            todo = min(chunk, n_iters - done)
            t0 = time.perf_counter()
            if side == "attacker":
                train_attacker(
                    env, attacker_net, opponent,
                    n_iters=todo, cfg=cfg, optimizer=a_opt, log_every=0,
                )
            else:
                train_defender_centralized(
                    env, defender_net, opponent,
                    n_iters=todo, cfg=cfg, optimizer=d_opt, log_every=0,
                    sort_by_distance=args.sort_defenders,
                )
            done += todo
            state["cumulative_iters"] += todo
            chunk_dt = time.perf_counter() - t0
            print(
                f"[{label}] +{todo} iters "
                f"({done}/{n_iters} in phase, {state['cumulative_iters']} total) "
                f"in {chunk_dt:.1f}s ({chunk_dt/todo:.2f}s/iter)"
            )
            snapshot(f"{label}_phase{done}")

    # Baseline animation: random nets vs each other
    snapshot("baseline_random")

    # ---- Optional BC warm-start of the centralized defender ----
    if args.bc_warm_start:
        import torch.nn.functional as Fnn
        bc_t0 = time.perf_counter()
        print("\n[BC] training defender to imitate HeuristicOrbitalDefender")
        heur_orb = HeuristicOrbitalDefender(
            v_defender=env.v_defender,
            target_pos=env_kwargs["target_pos"],
            orbit_radius=args.orbit_radius,
            capture_radius=env.capture_radius,
        )
        bc_rng = np.random.default_rng(args.seed + 7)
        n_obs = 4 + 4 * args.k
        n_act = 2 * args.k
        total_samples = args.n_envs * args.bc_n_rollouts * args.bc_rollout_steps
        obs_arr = np.empty((total_samples, n_obs), dtype=np.float32)
        act_arr = np.empty((total_samples, n_act), dtype=np.float32)
        write = 0
        for _ in range(args.bc_n_rollouts):
            env.reset()
            for _ in range(args.bc_rollout_steps):
                if args.sort_defenders:
                    obs_s, perm = CentralizedDefenderNet.build_sorted_obs(env)
                else:
                    obs_s = CentralizedDefenderNet.build_obs(env)
                    perm = None
                d_act = heur_orb.act(env)
                if perm is not None:
                    d_act_rec = np.take_along_axis(d_act, perm[..., None], axis=1)
                else:
                    d_act_rec = d_act
                obs_arr[write : write + env.B] = obs_s
                act_arr[write : write + env.B] = d_act_rec.reshape(env.B, n_act)
                write += env.B
                theta = bc_rng.uniform(-np.pi, np.pi, size=(env.B,)).astype(np.float32)
                r_mag = bc_rng.uniform(0.0, env.v_attacker, size=(env.B,)).astype(np.float32)
                a_vel = np.stack([r_mag * np.cos(theta), r_mag * np.sin(theta)], axis=-1)
                res = env.step(a_vel, d_act)
                if res.just_done.any():
                    env.reset_idxs(res.just_done)
        n_total = obs_arr.shape[0]
        n_val = max(1, int(n_total * 0.1))
        idx_perm = bc_rng.permutation(n_total)
        val_idx, train_idx = idx_perm[:n_val], idx_perm[n_val:]
        obs_train_t = torch.from_numpy(obs_arr[train_idx])
        act_train_t = torch.from_numpy(act_arr[train_idx])
        obs_val_t = torch.from_numpy(obs_arr[val_idx])
        act_val_t = torch.from_numpy(act_arr[val_idx])
        bc_opt = torch.optim.Adam(defender_net.parameters(), lr=args.bc_lr)
        n_train = obs_train_t.shape[0]
        for ep in range(args.bc_max_epochs):
            idxs = np.random.permutation(n_train)
            for start in range(0, n_train, args.bc_batch_size):
                mb = idxs[start : start + args.bc_batch_size]
                mean, _, _ = defender_net(obs_train_t[mb])
                loss = Fnn.mse_loss(mean, act_train_t[mb])
                bc_opt.zero_grad(); loss.backward(); bc_opt.step()
            with torch.no_grad():
                v_mean, _, _ = defender_net(obs_val_t)
                val_mae = (v_mean - act_val_t).abs().mean().item()
            if (ep % 5 == 0) or ep == args.bc_max_epochs - 1:
                print(f"  BC ep {ep:3d}  val_mae={val_mae:.4f}")
            if val_mae < args.bc_target_mae:
                print(f"  BC early-stop at epoch {ep}, val_mae={val_mae:.4f}")
                break
        torch.save(defender_net.state_dict(), ckpt_dir / "bc_defender.pt")
        print(f"[BC] done in {time.perf_counter()-bc_t0:.1f}s, val_mae={val_mae:.4f}")
        snapshot("after_BC")

    # ---- Phase 0: attacker warmup ----
    if args.bc_warm_start:
        print("\n[phase 0] attacker warmup vs BC-init centralized defender")
        warmup_opp = CentralizedDefender(
            defender_net, env.v_defender, deterministic=False,
            sort_by_distance=args.sort_defenders,
        )
    else:
        print("\n[phase 0] attacker warmup vs HeuristicDefender")
        warmup_opp = HeuristicDefender(
            v_defender=env.v_defender, capture_radius=env.capture_radius
        )
    train_chunked("attacker", args.attacker_warmup_iters, warmup_opp, "warmup")

    for r in range(args.n_alternations):
        print(f"\n[round {r+1}/{args.n_alternations} A] train centralized defender vs frozen attacker")
        att_frozen = NeuralAttacker(attacker_net, env.v_attacker, deterministic=False)
        env.reset()
        train_chunked("defender", args.defender_iters, att_frozen, f"r{r+1}_def")

        print(f"\n[round {r+1}/{args.n_alternations} B] train attacker vs frozen central defender")
        def_frozen = CentralizedDefender(
            defender_net, env.v_defender, deterministic=False,
            sort_by_distance=args.sort_defenders,
        )
        env.reset()
        train_chunked("attacker", args.attacker_iters, def_frozen, f"r{r+1}_att")

    torch.save(attacker_net.state_dict(), ckpt_dir / "attacker.pt")
    torch.save(defender_net.state_dict(), ckpt_dir / "central_defender.pt")
    summary = {
        "k": args.k, "sigma": args.sigma, "p": args.p, "seed": args.seed,
        "threads": torch.get_num_threads(),
        "schedule": {
            "attacker_warmup_iters": args.attacker_warmup_iters,
            "attacker_iters": args.attacker_iters,
            "defender_iters": args.defender_iters,
            "n_alternations": args.n_alternations,
            "total_iters": state["cumulative_iters"],
        },
        "anim_every": args.anim_every,
        "n_animations": state["anim_idx"],
        "wall_seconds": time.perf_counter() - state["t_start"],
    }
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nDone. {state['anim_idx']} animations, total wall {summary['wall_seconds']:.1f}s")
    print(f"checkpoints → {ckpt_dir}")
    print(f"animations  → {anim_dir}")
    print(f"summary     → {out_root / 'summary.json'}")


if __name__ == "__main__":
    main()
