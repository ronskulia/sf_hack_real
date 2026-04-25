# Project: Pursuit-Evasion Phase Transition Study

Build a Python project that studies phase transitions in a multi-agent pursuit-evasion game. One attacker drone tries to reach a fixed target while k defender drones try to intercept it. We sweep k from 1 to N and measure the attacker success rate to find the critical k\* where the defender gains a winning strategy.

## Repository Structure

```
pursuit_evasion/
├── README.md
├── requirements.txt
├── config.yaml                  # all hyperparameters live here
├── src/
│   ├── __init__.py
│   ├── env.py                   # vectorized pursuit-evasion environment
│   ├── policies.py              # heuristic + neural policies
│   ├── train.py                 # PPO training loop
│   ├── rollout.py               # evaluation rollouts (parallel)
│   ├── animate.py               # trajectory animations
│   └── plot.py                  # phase transition plots
├── scripts/
│   ├── run_experiment.py        # main entrypoint
│   └── quick_test.py            # smoke test (heuristics only, fast)
└── outputs/
    ├── checkpoints/
    ├── animations/
    └── plots/
```

## Environment Specification (`src/env.py`)

Implement a 2D continuous pursuit-evasion environment:

- **World:** unit square `[0, 1] × [0, 1]`, target at `(0.5, 0.5)`.
- **Attacker:** spawned uniformly on the boundary of the square. Max speed `v_a = 1.0` (in normalized units).
- **Defenders:** k drones, spawned uniformly inside an annulus around the target (inner radius 0.15, outer 0.35). Max speed `v_d = sigma * v_a`.
- **Dynamics:** discrete time, `dt = 0.02`. Each agent picks a velocity vector, magnitude is clipped to its max speed, position updates as `x += v * dt`. No acceleration limits (keep it simple for the hackathon; add later if there's time).
- **Action space:** continuous 2D velocity, clipped to max speed.
- **Observation:** for the attacker — own position, target position, all k defender positions and velocities. For each defender — own position/velocity, attacker position/velocity, teammate positions/velocities.
- **Termination:**
  - Attacker reaches target (within radius 0.02) → attacker wins.
  - Any defender within capture radius `r_capture = 0.03` of the attacker → with probability `p`, attacker dies (defender wins). With probability `1-p`, nothing happens (attacker survives the encounter and continues).
  - Step limit reached (default 500 steps) → defender wins by timeout.
- **Rewards:**
  - Attacker: `+1` for reaching target, `-1` for being killed, small shaping reward `-0.01 * distance_to_target` per step.
  - Defenders (shared team reward): `-1` for attacker reaching target, `+1` for killing attacker, small shaping `-0.01 * min_distance_from_any_defender_to_attacker`.

**Critical:** the environment must be vectorized — accept a batch of B environments and step them all in parallel as NumPy array operations. No Python for-loops over batch dimension. This is what makes the whole project tractable.

Use plain NumPy (not JAX) for portability — the environment is fast enough vectorized that we don't need GPU. Use PyTorch for the neural networks.

## Policies (`src/policies.py`)

Implement four policies:

1. **`HeuristicAttacker`:** head straight toward the target, but if a defender is within "danger radius" 0.15, add a repulsive component perpendicular to the defender direction (steer around the closest defender).
2. **`HeuristicDefender`:** Apollonius-style pursuit — head toward the predicted intercept point assuming the attacker continues at current velocity. For multiple defenders, assign each defender to a different angular sector around the attacker (split `2π` into k slices, each defender targets its sector's intercept).
3. **`NeuralAttacker`:** small MLP (2 hidden layers, 128 units, tanh), outputs mean and log-std of a 2D Gaussian over velocity. Input is the attacker's observation vector; pad/mask to handle variable k.
4. **`NeuralDefender`:** **shared parameter** policy used by all k defenders. Each defender runs the same network on its own observation. Use mean-pooling over teammate features so the architecture is permutation-invariant and works for any k. Same MLP size as attacker.

## Training (`src/train.py`)

PPO with the following protocol — I want **alternating best-response training**, not naive simultaneous self-play, because the latter oscillates:

1. **Phase 0:** train neural attacker against heuristic defenders for `attacker_warmup_iters` PPO iterations.
2. **Phase 1:** freeze attacker, train neural defenders against it for `defender_iters` iterations.
3. **Phase 2:** freeze defenders, retrain attacker against them for `attacker_iters` iterations.
4. Repeat phases 1–2 for `n_alternations` rounds (default 3).

Use a clean PPO implementation — fork CleanRL's `ppo_continuous_action.py` style (single-file, no framework). Standard hyperparameters: `lr=3e-4`, `gamma=0.99`, `gae_lambda=0.95`, `clip_coef=0.2`, `ent_coef=0.01`, `n_envs=256`, `n_steps=128`, `n_epochs=4`, `minibatch_size=512`. Save checkpoints to `outputs/checkpoints/k{k}_sigma{sigma}_p{p}/` after each phase.

Train a separate set of policies for each `(k, sigma, p)` cell in the sweep. Yes, this is expensive; that's why we vectorize aggressively and keep networks small.

## Rollout & Evaluation (`src/rollout.py`)

Function `evaluate(attacker_policy, defender_policy, k, sigma, p, n_episodes=2000) -> dict` that:

- Runs `n_episodes` rollouts in parallel (using the vectorized env).
- Returns `{"attacker_success_rate": float, "mean_episode_length": float, "ci_95": (low, high)}` with bootstrapped confidence intervals.

## Animations (`src/animate.py`)

After training each `(k, sigma, p)` cell, save **3 example episodes** as MP4 animations (use Matplotlib's `FuncAnimation` + `ffmpeg` writer):

- Show the unit square, target as a red star, attacker as a red triangle, defenders as blue circles.
- Plot fading trajectories (last 50 steps) for each agent.
- Capture radii drawn as faint circles around defenders.
- Title shows `k, sigma, p, outcome (attacker won / defender won)`.
- Save as `outputs/animations/k{k}_sigma{sigma}_p{p}_ep{i}.mp4`.

Pick the 3 episodes deterministically: one attacker win, one defender capture, one timeout (if available).

## Phase Transition Plot (`src/plot.py`)

Two plots:

1. **`phase_curve.png`:** for fixed `(sigma, p)`, attacker success rate vs k (with 95% CI error bars). Should show the transition.
2. **`phase_heatmap.png`:** if user sweeps over multiple `(sigma, p)` values, show a 2D heatmap with k on x-axis, `(sigma, p)` combinations on y-axis, color = attacker success rate. Use `viridis` colormap.

Also produce `phase_curve_heuristic.png` using only the heuristic policies (no RL) as a baseline — this is a critical sanity check and a fallback if RL training is flaky.

## Configuration (`config.yaml`)

```yaml
sweep:
  k_values: [1, 2, 3, 4, 5, 6]
  sigma: 0.7          # defender speed ratio (single value or list)
  p: 0.8              # kill probability (single value or list)

env:
  dt: 0.02
  max_steps: 500
  capture_radius: 0.03
  target_radius: 0.02
  target_pos: [0.5, 0.5]

training:
  attacker_warmup_iters: 50
  attacker_iters: 100
  defender_iters: 100
  n_alternations: 3
  n_envs: 256
  n_steps: 128
  lr: 3.0e-4
  gamma: 0.99
  gae_lambda: 0.95
  clip_coef: 0.2
  ent_coef: 0.01

evaluation:
  n_episodes: 2000
  n_animations: 3

mode: "rl"            # "rl" or "heuristic" — heuristic skips training entirely
seed: 0
```

## Main Entrypoint (`scripts/run_experiment.py`)

```bash
python scripts/run_experiment.py --config config.yaml
```

Should:
1. Parse config.
2. Run heuristic baseline first (always — it's fast and gives us a fallback plot).
3. If `mode == "rl"`, train policies for each cell in the sweep.
4. Evaluate all cells.
5. Save animations.
6. Save plots.
7. Print a summary table to stdout (k, success rate, CI).

Use `tqdm` for progress bars and log to both stdout and `outputs/run.log`.

## Quick Test (`scripts/quick_test.py`)

A smoke test that runs heuristic-vs-heuristic for k=1 and k=4 with 200 episodes each and produces one animation. Should complete in under 30 seconds. I'll run this first to verify the environment is correct before kicking off real training.

## Implementation Order

Please implement and verify in this order, stopping after each step to confirm it works:

1. Vectorized environment + heuristic policies + `quick_test.py`. Verify the animation looks physically plausible.
2. Heuristic baseline sweep — produce `phase_curve_heuristic.png`. Confirm it shows a sensible curve (attacker win rate decreasing in k).
3. PPO training for the **attacker only** against heuristic defenders, single (k, sigma, p) cell. Verify learning curve goes up.
4. Full alternating training loop, single cell. Verify it converges.
5. Full sweep + final plots.

## Requirements

```
numpy
torch
matplotlib
pyyaml
tqdm
imageio
imageio-ffmpeg
```

Use Python 3.11+. Type-hint everything. Add docstrings to every public function. Keep dependencies minimal — no Gymnasium, no Stable-Baselines3, no Hydra.
