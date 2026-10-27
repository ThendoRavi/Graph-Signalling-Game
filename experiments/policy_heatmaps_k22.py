"""Greedy-policy heatmaps for the Results chapter's "Exhaustive Greedy-Policy
Heatmaps" subsection.

Generates, for K_{2,2}, the per-agent greedy-policy grid
(:func:`gsg.visualization.render_policy_heatmap`) for:

  * the oracle (perfect by construction, R = 1.00);
  * the random baseline (no stable policy -- an arbitrary single draw);
  * one perfect (R = 1.00) run from each of the three IQL variants;
  * one "exploiting joint channel" (R = 0.875) run, as a contrast.

Perfect / 0.875 seeds are not hard-coded: for each variant the script trains
seeds in order, computes each run's exact mean reward by exhaustive
enumeration, and picks the first run that lands in the target band -- so the
saved figures are guaranteed to show the reward stated in their filename and
title.

Run:  python experiments/policy_heatmaps_k22.py
"""

from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from gsg.baselines.oracle_baseline import k22_oracle
from gsg.baselines.random_baseline import RandomGuesser, RandomSignaller
from gsg.environment.graphs import SignallingGraph
from gsg.environment.pettingzoo_env import GraphSignallingParallelEnv
from gsg.evaluation.conventions import exhaustive_joint_table, extract_all_policies
from gsg.training.trainer import train_independent_iql, train_shared_iql, train_two_brain_iql
from gsg.visualization import render_policy_heatmap

TRAIN_EPISODES = 4_000
EPSILON_DECAY = 2_500
MAX_SEEDS = 12

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "question1", "figures", "policy_heatmaps")


def make_env(seed=None):
    return GraphSignallingParallelEnv(SignallingGraph.complete_bipartite(2, 2), seed=seed)


def _train(variant, env, seed):
    if variant == "shared":
        _n, sig, gue, _h = train_shared_iql(env, num_episodes=TRAIN_EPISODES,
                                            epsilon_decay_episodes=EPSILON_DECAY, seed=seed)
    elif variant == "two_brain":
        _s, _g, sig, gue, _h = train_two_brain_iql(env, num_episodes=TRAIN_EPISODES,
                                                   epsilon_decay_episodes=EPSILON_DECAY, seed=seed)
    elif variant == "independent":
        _s, _g, sig, gue, _h = train_independent_iql(env, num_episodes=TRAIN_EPISODES,
                                                     epsilon_decay_episodes=EPSILON_DECAY, seed=seed)
    else:
        raise ValueError(variant)
    return sig, gue


def reward_of(env, sig, gue):
    outcomes = exhaustive_joint_table(env.graph, env.item_space.n, sig, gue)
    return sum(o.reward for o in outcomes) / len(outcomes)


def save_heatmap(env, sig, gue, title, fname):
    policies = extract_all_policies(env.graph, env.item_space.n, sig, gue)
    path = os.path.join(FIG_DIR, fname)
    fig = render_policy_heatmap(policies, title=title, save_path=path)
    plt.close(fig)
    print(f"  saved -> {os.path.normpath(path)}")


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    print("Generating greedy-policy heatmaps, K_2,2 ...", flush=True)

    # --- oracle ---
    env = make_env(seed=0)
    sig, gue = k22_oracle()
    r = reward_of(env, sig, gue)
    print(f"oracle: R={r:.3f}")
    save_heatmap(env, sig, gue, f"oracle (R = {r:.2f})", "conventions_oracle.png")

    # --- random baseline (arbitrary single draw; no stable policy) ---
    env = make_env(seed=0)
    sig = {0: RandomSignaller(seed=1), 1: RandomSignaller(seed=2)}
    gue = {0: RandomGuesser(seed=3), 1: RandomGuesser(seed=4)}
    save_heatmap(env, sig, gue,
                 "random baseline (no stable policy; one arbitrary draw)",
                 "conventions_random.png")

    # --- one perfect run per variant, + first 0.875 run seen ---
    exploit_saved = False
    for variant in ("shared", "two_brain", "independent"):
        perfect_saved = False
        for seed in range(MAX_SEEDS):
            env = make_env(seed=seed)
            sig, gue = _train(variant, env, seed)
            r = reward_of(env, sig, gue)
            print(f"{variant} seed {seed}: R={r:.3f}", flush=True)

            if (not perfect_saved) and r == 1.0:
                save_heatmap(env, sig, gue,
                             f"{variant}: perfect coordination (seed {seed}, R = 1.00)",
                             f"conventions_{variant}_perfect.png")
                perfect_saved = True

            if (not exploit_saved) and abs(r - 0.875) < 1e-6:
                save_heatmap(env, sig, gue,
                             f"{variant}: exploiting joint channel (seed {seed}, R = 0.875)",
                             "conventions_exploiting_0875.png")
                exploit_saved = True

            if perfect_saved and exploit_saved:
                break
        if not perfect_saved:
            print(f"  WARNING: no perfect run found for {variant} in {MAX_SEEDS} seeds")

    if not exploit_saved:
        print("  WARNING: no 0.875 run found across the variants scanned")
    print("done.", flush=True)


if __name__ == "__main__":
    main()
