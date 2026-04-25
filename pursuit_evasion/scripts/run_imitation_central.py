"""Behavioral cloning: train a CentralizedDefenderNet to imitate the
HeuristicOrbitalDefender. No PPO, no rewards — pure supervised regression
of (centralized obs) -> (heuristic action), MSE on the policy mean.

Pipeline:
  1. Build a dataset of (obs, heuristic_action) pairs by repeatedly
     resetting envs and stepping with random attacker actions + heuristic
     defender actions. The env is just a state generator; rewards are not
     used. State diversity comes from spawn randomness + rollout drift.
  2. Train CentralizedDefenderNet by MSE between its actor mean and the
     heuristic action (continuous targets, scaled to be in v_defender-units
     before clipping).
  3. Evaluate on a held-out batch of (obs, action) pairs.
  4. Save checkpoint and (optionally) render two side-by-side animations:
     heuristic-driven episode and NN-driven episode at the same seed.

Usage::
    python scripts/run_imitation_central.py --k 4 --sigma 1.0 --p 0.7 \\
        --orbit-radius 0.20 --n-rollouts 200 --rollout-steps 64 \\
        --hidden 256 --n-layers 3 --epochs 30 --batch-size 1024 \\
        --out outputs_imitation_central
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# M2 BLAS knobs before torch import
_DEFAULT_THREADS = 4
os.environ.setdefault("OMP_NUM_THREADS", str(_DEFAULT_THREADS))
os.environ.setdefault("MKL_NUM_THREADS", str(_DEFAULT_THREADS))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(_DEFAULT_THREADS))
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", str(_DEFAULT_THREADS))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

torch.set_num_threads(_DEFAULT_THREADS)

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.animate import run_episode, save_animation
from src.env import PursuitEvasionEnv
from src.policies import (
    CentralizedDefender,
    CentralizedDefenderNet,
    HeuristicAttacker,
    HeuristicOrbitalDefender,
)


# ----------------------------------------------------------------- dataset ----


def collect_dataset(
    *,
    env: PursuitEvasionEnv,
    heuristic: HeuristicOrbitalDefender,
    n_rollouts: int,
    rollout_steps: int,
    rng: np.random.Generator,
    sort_by_distance: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect (obs, action) pairs by stepping the env with the heuristic.

    The heuristic drives the defenders; the attacker takes random velocity
    actions (uniform on the speed disc) for state diversity. Rewards are
    not used. Each rollout contributes ``B * rollout_steps`` samples; we run
    ``n_rollouts`` rollouts.

    If ``sort_by_distance`` is True, both the recorded obs and action are
    permuted so that defender slot 0 is always the closest to the attacker.
    The env step still uses the original (env-ordered) action.
    """
    B, k = env.B, env.k
    n_obs = 4 + 4 * k
    n_act = 2 * k
    total = n_rollouts * rollout_steps * B
    obs_arr = np.empty((total, n_obs), dtype=np.float32)
    act_arr = np.empty((total, n_act), dtype=np.float32)
    write = 0
    for r in range(n_rollouts):
        env.reset()
        for t in range(rollout_steps):
            if sort_by_distance:
                obs, perm = CentralizedDefenderNet.build_sorted_obs(env)
            else:
                obs = CentralizedDefenderNet.build_obs(env)
                perm = None
            d_act = heuristic.act(env)  # (B, k, 2), env-ordered
            if perm is not None:
                # Permute action to match sorted obs ordering.
                d_act_sorted = np.take_along_axis(d_act, perm[..., None], axis=1)
                act_record = d_act_sorted.reshape(B, n_act)
            else:
                act_record = d_act.reshape(B, n_act)
            obs_arr[write : write + B] = obs
            act_arr[write : write + B] = act_record
            write += B
            # Random attacker velocity, magnitude ~ U(0, v_attacker).
            theta = rng.uniform(-np.pi, np.pi, size=(B,)).astype(np.float32)
            r_mag = rng.uniform(0.0, env.v_attacker, size=(B,)).astype(np.float32)
            a_vel = np.stack([r_mag * np.cos(theta), r_mag * np.sin(theta)], axis=-1)
            res = env.step(a_vel, d_act)  # env-ordered action
            if res.just_done.any():
                env.reset_idxs(res.just_done)
    assert write == total
    return obs_arr, act_arr


# ---------------------------------------------------------------- training ----


def train_imitation(
    *,
    net: CentralizedDefenderNet,
    obs_train: np.ndarray,
    act_train: np.ndarray,
    obs_val: np.ndarray,
    act_val: np.ndarray,
    epochs: int,
    batch_size: int,
    lr: float,
    log_fn=print,
) -> dict:
    """MSE regression of (obs -> heuristic action) on the actor mean head."""
    opt = optim.Adam(net.parameters(), lr=lr)
    n = obs_train.shape[0]
    obs_train_t = torch.from_numpy(obs_train)
    act_train_t = torch.from_numpy(act_train)
    obs_val_t = torch.from_numpy(obs_val)
    act_val_t = torch.from_numpy(act_val)

    history: list[dict] = []
    t0 = time.perf_counter()
    for ep in range(epochs):
        idxs = np.random.permutation(n)
        running = 0.0
        count = 0
        for start in range(0, n, batch_size):
            mb = idxs[start : start + batch_size]
            obs_mb = obs_train_t[mb]
            act_mb = act_train_t[mb]
            mean, _logstd, _value = net(obs_mb)
            loss = nn.functional.mse_loss(mean, act_mb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item() * len(mb)
            count += len(mb)
        train_mse = running / count
        with torch.no_grad():
            val_mean, _, _ = net(obs_val_t)
            val_mse = nn.functional.mse_loss(val_mean, act_val_t).item()
            # per-dim mean abs error in same units as v_defender
            val_mae = (val_mean - act_val_t).abs().mean().item()
        history.append({
            "epoch": ep, "train_mse": train_mse,
            "val_mse": val_mse, "val_mae": val_mae,
        })
        elapsed = time.perf_counter() - t0
        log_fn(
            f"  ep {ep:3d}  train_mse={train_mse:.6f}  "
            f"val_mse={val_mse:.6f}  val_mae={val_mae:.4f}  [{elapsed:.1f}s]"
        )
    return {"history": history}


# ------------------------------------------------------------- entry point ----


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--p", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--orbit-radius", type=float, default=0.20)
    parser.add_argument("--out", type=str,
                        default=str(_REPO_ROOT / "outputs_imitation_central"))
    # dataset
    parser.add_argument("--n-envs", type=int, default=128)
    parser.add_argument("--n-rollouts", type=int, default=80)
    parser.add_argument("--rollout-steps", type=int, default=32)
    parser.add_argument("--val-frac", type=float, default=0.1)
    # net + training
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    # eval / animation
    parser.add_argument("--n-anim", type=int, default=2,
                        help="Number of (heuristic, NN) animation pairs to render.")
    parser.add_argument("--sort-defenders", action="store_true",
                        help="Reorder defenders by distance to attacker before each "
                        "forward pass (slot 0 = closest). Permutation-invariant; "
                        "typically halves required NN capacity.")
    args = parser.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    anim_dir = out_root / "animations"
    anim_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_root / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== imitation: HeuristicOrbitalDefender → CentralizedDefenderNet ===")
    print(f"  k={args.k}  sigma={args.sigma}  p={args.p}  orbit_r={args.orbit_radius}")
    print(f"  net: hidden={args.hidden} n_layers={args.n_layers}  "
          f"sort_defenders={args.sort_defenders}")
    print(f"  data: n_envs={args.n_envs} n_rollouts={args.n_rollouts} "
          f"rollout_steps={args.rollout_steps}  → "
          f"{args.n_envs * args.n_rollouts * args.rollout_steps} samples")
    print(f"  train: epochs={args.epochs} batch={args.batch_size} lr={args.lr}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    env_kwargs = {
        "dt": 0.02, "max_steps": 500, "capture_radius": 0.03,
        "target_radius": 0.10, "target_pos": (0.5, 0.5), "v_attacker": 1.0,
        "inner_radius": 0.15, "outer_radius": 0.35, "shaping": 0.01,
    }
    env = PursuitEvasionEnv(
        batch_size=args.n_envs, k=args.k, sigma=args.sigma, p=args.p,
        seed=args.seed, **env_kwargs,
    )
    heuristic = HeuristicOrbitalDefender(
        v_defender=env.v_defender, target_pos=env_kwargs["target_pos"],
        orbit_radius=args.orbit_radius, capture_radius=env.capture_radius,
    )

    # ---- 1. dataset ----
    t0 = time.perf_counter()
    obs_arr, act_arr = collect_dataset(
        env=env, heuristic=heuristic,
        n_rollouts=args.n_rollouts, rollout_steps=args.rollout_steps, rng=rng,
        sort_by_distance=args.sort_defenders,
    )
    print(f"  dataset: {obs_arr.shape[0]} samples in {time.perf_counter()-t0:.1f}s")
    n_total = obs_arr.shape[0]
    n_val = int(n_total * args.val_frac)
    perm = rng.permutation(n_total)
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    obs_train, act_train = obs_arr[train_idx], act_arr[train_idx]
    obs_val, act_val = obs_arr[val_idx], act_arr[val_idx]
    print(f"  train={obs_train.shape[0]}  val={obs_val.shape[0]}")

    # ---- 2. train ----
    net = CentralizedDefenderNet(k=args.k, hidden=args.hidden, n_layers=args.n_layers)
    log = train_imitation(
        net=net,
        obs_train=obs_train, act_train=act_train,
        obs_val=obs_val, act_val=act_val,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
    )

    # ---- 3. final eval ----
    with torch.no_grad():
        net.eval()
        # batched MSE over the whole val set (already small)
        v_mean, _, _ = net(torch.from_numpy(obs_val))
        final_val_mse = nn.functional.mse_loss(v_mean, torch.from_numpy(act_val)).item()
        final_val_mae = (v_mean - torch.from_numpy(act_val)).abs().mean().item()
        # per-component breakdown (each defender's vx, vy)
        per_dim_mae = (v_mean - torch.from_numpy(act_val)).abs().mean(dim=0).numpy().tolist()
    print(f"\nfinal val: mse={final_val_mse:.6f}  mae={final_val_mae:.4f}")
    print(f"per-dim mae: {[f'{x:.3f}' for x in per_dim_mae]}")

    torch.save(net.state_dict(), ckpt_dir / "imitation_defender.pt")

    # ---- 4. comparison animations ----
    print(f"\nrendering {args.n_anim} comparison animations (heuristic vs imitation NN)...")
    for i in range(args.n_anim):
        # heuristic episode
        env_h = PursuitEvasionEnv(
            batch_size=1, k=args.k, sigma=args.sigma, p=args.p,
            seed=args.seed + 5000 + i, **env_kwargs,
        )
        heur_att = HeuristicAttacker(target=env_h.target, v_attacker=env_h.v_attacker)
        heur_def = HeuristicOrbitalDefender(
            v_defender=env_h.v_defender, target_pos=env_kwargs["target_pos"],
            orbit_radius=args.orbit_radius, capture_radius=env_h.capture_radius,
        )
        h_h = run_episode(env_h, heur_att, heur_def)
        save_animation(h_h, env_h.metadata(),
                       anim_dir / f"compare_{i}_heuristic.mp4")
        # NN episode at the same seed
        env_n = PursuitEvasionEnv(
            batch_size=1, k=args.k, sigma=args.sigma, p=args.p,
            seed=args.seed + 5000 + i, **env_kwargs,
        )
        heur_att2 = HeuristicAttacker(target=env_n.target, v_attacker=env_n.v_attacker)
        nn_def = CentralizedDefender(
            net, env_n.v_defender, deterministic=True,
            sort_by_distance=args.sort_defenders,
        )
        h_n = run_episode(env_n, heur_att2, nn_def)
        save_animation(h_n, env_n.metadata(),
                       anim_dir / f"compare_{i}_imitation_nn.mp4")
        print(f"  pair {i}: heur outcome={h_h['outcome']} steps={h_h['steps']}  |  "
              f"nn outcome={h_n['outcome']} steps={h_n['steps']}")

    summary = {
        "k": args.k, "sigma": args.sigma, "p": args.p,
        "orbit_radius": args.orbit_radius, "seed": args.seed,
        "sort_defenders": args.sort_defenders,
        "net": {"hidden": args.hidden, "n_layers": args.n_layers},
        "dataset": {
            "n_envs": args.n_envs, "n_rollouts": args.n_rollouts,
            "rollout_steps": args.rollout_steps,
            "n_train": int(obs_train.shape[0]), "n_val": int(obs_val.shape[0]),
        },
        "training": {
            "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
            "history": log["history"],
        },
        "final": {
            "val_mse": final_val_mse, "val_mae": final_val_mae,
            "per_dim_mae": per_dim_mae,
        },
    }
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\ncheckpoint → {ckpt_dir / 'imitation_defender.pt'}")
    print(f"summary    → {out_root / 'summary.json'}")
    print(f"animations → {anim_dir}")


if __name__ == "__main__":
    main()
