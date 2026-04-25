"""Smoke test for the vectorized env + heuristic policies.

Runs heuristic-vs-heuristic for ``k = 1`` and ``k = 4``, 200 episodes each,
then writes one animation. Should finish in well under 30 seconds on a CPU.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

# Allow ``python scripts/quick_test.py`` from project root by adding it to path.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.animate import run_episode, save_animation  # noqa: E402
from src.env import (  # noqa: E402
    OUTCOME_ATTACKER_WIN,
    OUTCOME_DEFENDER_CAPTURE,
    OUTCOME_DEFENDER_TIMEOUT,
    PursuitEvasionEnv,
)
from src.policies import HeuristicAttacker, HeuristicDefender  # noqa: E402


def evaluate_heuristic(
    *, k: int, sigma: float, p: float, n_episodes: int, seed: int
) -> dict[str, float | int]:
    """Run ``n_episodes`` heuristic-vs-heuristic episodes in parallel."""
    env = PursuitEvasionEnv(batch_size=n_episodes, k=k, sigma=sigma, p=p, seed=seed)
    attacker = HeuristicAttacker(target=env.target, v_attacker=env.v_attacker)
    defender = HeuristicDefender(
        v_defender=env.v_defender, capture_radius=env.capture_radius
    )

    # Outer cap is generous; inner break exits as soon as every env is done.
    for _ in range(env.max_steps + 5):
        a = attacker.act(env)
        d = defender.act(env)
        env.step(a, d)
        if bool(env.done.all()):
            break

    n_wins = int((env.outcome == OUTCOME_ATTACKER_WIN).sum())
    n_capture = int((env.outcome == OUTCOME_DEFENDER_CAPTURE).sum())
    n_timeout = int((env.outcome == OUTCOME_DEFENDER_TIMEOUT).sum())
    return {
        "k": k,
        "n_episodes": n_episodes,
        "attacker_success_rate": n_wins / n_episodes,
        "n_attacker_wins": n_wins,
        "n_capture": n_capture,
        "n_timeout": n_timeout,
        "mean_steps": float(env.step_count.mean()),
    }


def make_animation(*, k: int, sigma: float, p: float, seed: int, out_path: Path) -> Path:
    """Render one episode to MP4 and return the saved path."""
    env = PursuitEvasionEnv(batch_size=1, k=k, sigma=sigma, p=p, seed=seed)
    attacker = HeuristicAttacker(target=env.target, v_attacker=env.v_attacker)
    defender = HeuristicDefender(
        v_defender=env.v_defender, capture_radius=env.capture_radius
    )
    history = run_episode(env, attacker, defender)
    return save_animation(history, env.metadata(), out_path)


def main() -> None:
    print("=== quick_test: heuristic vs heuristic ===")
    t0 = time.time()

    results = []
    for k in (1, 4):
        r = evaluate_heuristic(k=k, sigma=0.7, p=0.8, n_episodes=200, seed=k)
        results.append(r)
        print(
            f"  k={r['k']}  succ={r['attacker_success_rate']:.3f}  "
            f"wins={r['n_attacker_wins']}  cap={r['n_capture']}  "
            f"timeout={r['n_timeout']}  mean_steps={r['mean_steps']:.1f}"
        )
    eval_elapsed = time.time() - t0
    print(f"  evaluation: {eval_elapsed:.2f}s")

    # Find a seed that gives an interesting k=4 episode (defender capture is most fun to watch).
    out_dir = _REPO_ROOT / "outputs" / "animations"
    out_path = out_dir / "quick_test_k4.mp4"
    print(f"\n=== rendering animation: k=4, seed=0 -> {out_path.relative_to(_REPO_ROOT)} ===")
    t1 = time.time()
    make_animation(k=4, sigma=0.7, p=0.8, seed=0, out_path=out_path)
    print(f"  animation: {time.time() - t1:.2f}s")

    # Also a k=1 animation so we can see the simple case.
    out_path_k1 = out_dir / "quick_test_k1.mp4"
    print(f"=== rendering animation: k=1, seed=2 -> {out_path_k1.relative_to(_REPO_ROOT)} ===")
    t1 = time.time()
    make_animation(k=1, sigma=0.7, p=0.8, seed=2, out_path=out_path_k1)
    print(f"  animation: {time.time() - t1:.2f}s")

    total = time.time() - t0
    print(f"\n=== total elapsed: {total:.2f}s ===")

    # Sanity-check expected qualitative behavior. We don't fail the script on
    # this; just print a warning.
    if results[0]["attacker_success_rate"] <= results[1]["attacker_success_rate"]:
        print(
            "WARNING: attacker success rate did NOT decrease from k=1 to k=4; "
            "heuristic curve is suspicious."
        )
    else:
        print("OK: attacker success rate decreases from k=1 to k=4 as expected.")


if __name__ == "__main__":
    main()
