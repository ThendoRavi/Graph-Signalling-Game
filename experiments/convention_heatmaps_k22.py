"""Convergence analysis: action-response heatmaps for K_{2,2}.

For a handful of seeds per IQL variant, trains a population, then instead of
just reporting a reward number, extracts and *shows* the actual convention
each agent converged to:

  [1] The exhaustive joint table -- every one of the 4 possible hidden-item
      combinations, walked through deterministically (exploration off), with
      the resulting signals/guesses/correctness for each. Possible because
      K_{2,2}'s state space is small enough to enumerate completely rather
      than sample (Section 4.6.1).
  [2] A saved PNG heatmap per run: one small 2x2 grid per agent showing its
      *entire* greedy policy (see gsg.evaluation.conventions and
      gsg.visualization.render_policy_heatmap) -- e.g. whether S0 and S1
      converged to the same rule (redundant, caps reward around 0.75-0.89)
      or complementary ones (S0 tracks G0, S1 tracks G1 -- reaches 1.00).
  [3] A canonical id for each signaller's policy, so repeated conventions
      across seeds can be spotted directly, not just inferred from reward.

Deliberately a *small* batch (NUM_SEEDS_PER_VARIANT below), not the full
P=30 protocol -- this is for looking closely at a handful of concrete
examples, not for estimating a convergence rate (compare_iql_variants_k22.py
already does that, at the P=30 scale).

Run:  python experiments/convention_heatmaps_k22.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from gsg.environment.graphs import SignallingGraph
from gsg.environment.pettingzoo_env import GraphSignallingParallelEnv
from gsg.evaluation.conventions import (
    canonical_id,
    convention_summary,
    exhaustive_joint_table,
    extract_all_policies,
)
from gsg.training.trainer import train_independent_iql, train_shared_iql, train_two_brain_iql
from gsg.visualization import ITEM_NAMES, SIGNAL_NAMES, render_policy_heatmap


def _item_name(value: int) -> str:
    return ITEM_NAMES.get(value, str(value))


def _signal_name(value: int) -> str:
    return SIGNAL_NAMES.get(value, str(value))

NUM_SEEDS_PER_VARIANT = 4
NUM_EPISODES = 3000
EPSILON_DECAY_EPISODES = 1800

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "question1", "figures")

VARIANTS = {
    "shared": lambda env, seed: train_shared_iql(
        env, num_episodes=NUM_EPISODES, epsilon_decay_episodes=EPSILON_DECAY_EPISODES, seed=seed,
    )[1:3],  # (network, signaller_agents, guesser_agents, history) -> (signaller_agents, guesser_agents)
    "two_brain": lambda env, seed: train_two_brain_iql(
        env, num_episodes=NUM_EPISODES, epsilon_decay_episodes=EPSILON_DECAY_EPISODES, seed=seed,
    )[2:4],  # (sig_net, gue_net, signaller_agents, guesser_agents, history) -> (..., ...)
    "independent": lambda env, seed: train_independent_iql(
        env, num_episodes=NUM_EPISODES, epsilon_decay_episodes=EPSILON_DECAY_EPISODES, seed=seed,
    )[2:4],  # (sig_nets, gue_nets, signaller_agents, guesser_agents, history) -> (..., ...)
}


def make_env(seed=None) -> GraphSignallingParallelEnv:
    graph = SignallingGraph.complete_bipartite(2, 2)
    return GraphSignallingParallelEnv(graph, seed=seed)


def print_joint_table(outcomes) -> None:
    for o in outcomes:
        items_str = ", ".join(f"G{g}={_item_name(v)}" for g, v in sorted(o.items.items()))
        signals_str = ", ".join(f"S{s}={_signal_name(v)}" for s, v in sorted(o.signals.items()))
        guesses_str = ", ".join(
            f"G{g}={_item_name(v)}{'(OK)' if o.correct[g] else '(XX)'}"
            for g, v in sorted(o.guesses.items())
        )
        print(f"    items[{items_str}] -> signals[{signals_str}] -> guesses[{guesses_str}]  R={o.reward:.2f}")


def main() -> None:
    os.makedirs(FIGURES_DIR, exist_ok=True)
    print("=" * 72)
    print("Convergence analysis: action-response heatmaps, K_{2,2}")
    print(f"{NUM_SEEDS_PER_VARIANT} seeds per variant, {NUM_EPISODES} training episodes each")
    print("=" * 72)

    tallies = {name: {"perfect": 0, "other": 0} for name in VARIANTS}

    for variant_name, train_fn in VARIANTS.items():
        print(f"\n{'#' * 72}")
        print(f"# Variant: {variant_name}")
        print(f"{'#' * 72}")

        for seed in range(NUM_SEEDS_PER_VARIANT):
            env = make_env(seed=seed)
            signaller_agents, guesser_agents = train_fn(env, seed)

            outcomes = exhaustive_joint_table(env.graph, env.item_space.n, signaller_agents, guesser_agents)
            summary = convention_summary(outcomes)
            policies = extract_all_policies(env.graph, env.item_space.n, signaller_agents, guesser_agents)

            s0_id = canonical_id(policies["S0"], action_range=2, value_range=2, neighbourhood_size=2)
            s1_id = canonical_id(policies["S1"], action_range=2, value_range=2, neighbourhood_size=2)

            print(f"\n--- {variant_name}, seed {seed}: {summary} "
                  f"(S0 convention id={s0_id}, S1 convention id={s1_id}) ---")
            print_joint_table(outcomes)

            save_path = os.path.join(FIGURES_DIR, f"conventions_{variant_name}_seed{seed}.png")
            render_policy_heatmap(
                policies,
                title=f"{variant_name} -- seed {seed} -- {summary}",
                save_path=save_path,
            )
            print(f"    saved heatmap -> {os.path.normpath(save_path)}")

            key = "perfect" if summary.startswith("perfect") else "other"
            tallies[variant_name][key] += 1

    print(f"\n{'=' * 72}")
    print("Summary: how many of these seeds found a perfect (complementary) convention")
    print("=" * 72)
    for variant_name, counts in tallies.items():
        total = counts["perfect"] + counts["other"]
        print(f"  {variant_name:12s}: {counts['perfect']}/{total} perfect")


if __name__ == "__main__":
    main()
