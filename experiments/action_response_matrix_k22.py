"""Action Response Matrix: empirical action/response correlation, K_{2,2}.

For each of the three IQL variants, plus the oracle (a known-perfect
reference point) and random (a known-*no*-rule reference point at the
opposite end), trains/builds the population, lets it settle, then plays
1000 *real* episodes and tallies -- for every (signaller, guesser) edge --
how often each (signaller's emitted signal, guesser's produced guess) pair
actually occurred. This is different from the earlier convention heatmaps
(gsg.evaluation.conventions.exhaustive_joint_table /
extract_all_policies / render_policy_heatmap), which show what a policy
*would* do for every possible input, assuming it's a fixed rule. This one
shows what actually came out, how often, from real sampled play -- in the
spirit of the cross-play matrices in the other-play / any-play literature.

The two tools agree when a policy is genuinely deterministic (a converged
greedy agent, or the oracle): the empirical matrix will show ~100%/0% splits
matching the deterministic table exactly. They diverge, informatively, when
a policy isn't a clean fixed rule -- most notably the random baseline, whose
enumerated 4-cell table earlier could look "perfect" purely by the luck of
one query per cell, but whose 1000-episode empirical matrix reliably shows
close to a flat 25% everywhere, because there is no rule to have gotten
lucky about.

Run:  python experiments/action_response_matrix_k22.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from gsg.baselines.oracle_baseline import k22_oracle
from gsg.baselines.random_baseline import RandomGuesser, RandomSignaller
from gsg.environment.graphs import SignallingGraph
from gsg.environment.pettingzoo_env import GraphSignallingParallelEnv
from gsg.evaluation.conventions import all_edges_action_response_frequencies
from gsg.training.trainer import train_independent_iql, train_shared_iql, train_two_brain_iql
from gsg.visualization import render_action_response_matrix

NUM_TRAINING_EPISODES = 3000
EPSILON_DECAY_EPISODES = 1800
NUM_MEASUREMENT_EPISODES = 1000
TRAIN_SEED = 0

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "question1", "figures")


def make_env(seed=None) -> GraphSignallingParallelEnv:
    graph = SignallingGraph.complete_bipartite(2, 2)
    return GraphSignallingParallelEnv(graph, seed=seed)


def print_frequencies(freqs) -> None:
    for (s, g), table in sorted(freqs.items()):
        print(f"    S{s} -> G{g}:")
        for (signal, guess), frac in sorted(table.items()):
            light = "OFF" if signal == 0 else "ON"
            item = "CAT" if guess == 0 else "DOG"
            print(f"        signal={light:3s}, guess={item:3s} -> {frac * 100:5.1f}%")


def run_variant(name: str, signaller_agents, guesser_agents, measure_env) -> None:
    freqs = all_edges_action_response_frequencies(
        measure_env, signaller_agents, guesser_agents, num_episodes=NUM_MEASUREMENT_EPISODES,
    )
    print(f"\n--- {name} ---")
    print_frequencies(freqs)

    save_path = os.path.join(FIGURES_DIR, f"action_response_{name}.png")
    render_action_response_matrix(
        freqs,
        title=f"{name} -- action/response frequencies over {NUM_MEASUREMENT_EPISODES} episodes",
        save_path=save_path,
    )
    print(f"    saved -> {os.path.normpath(save_path)}")


def main() -> None:
    os.makedirs(FIGURES_DIR, exist_ok=True)
    print("=" * 72)
    print("Action Response Matrix (empirical), K_{2,2}")
    print(f"{NUM_TRAINING_EPISODES} training episodes, then {NUM_MEASUREMENT_EPISODES} "
          f"measurement episodes per variant")
    print("=" * 72)

    # Oracle: no training needed, deterministic by construction -- included
    # as the "what does 100%/0% actually look like" reference point.
    env = make_env(seed=TRAIN_SEED)
    signaller_agents, guesser_agents = k22_oracle()
    run_variant("oracle", signaller_agents, guesser_agents, env)

    # Random: no training, no rule at all -- ignores its observation and
    # answers by coin flip every call. Included as the "what does flat 25%
    # everywhere actually look like" reference point, at the opposite end
    # from the oracle.
    env = make_env(seed=TRAIN_SEED)
    signaller_agents = {0: RandomSignaller(seed=100), 1: RandomSignaller(seed=101)}
    guesser_agents = {0: RandomGuesser(seed=102), 1: RandomGuesser(seed=103)}
    run_variant("random", signaller_agents, guesser_agents, env)

    # Shared-brain
    env = make_env(seed=TRAIN_SEED)
    _network, signaller_agents, guesser_agents, _history = train_shared_iql(
        env, num_episodes=NUM_TRAINING_EPISODES, epsilon_decay_episodes=EPSILON_DECAY_EPISODES,
        seed=TRAIN_SEED,
    )
    run_variant("shared", signaller_agents, guesser_agents, env)

    # Two-brain
    env = make_env(seed=TRAIN_SEED)
    _sig_net, _gue_net, signaller_agents, guesser_agents, _history = train_two_brain_iql(
        env, num_episodes=NUM_TRAINING_EPISODES, epsilon_decay_episodes=EPSILON_DECAY_EPISODES,
        seed=TRAIN_SEED,
    )
    run_variant("two_brain", signaller_agents, guesser_agents, env)

    # Independent
    env = make_env(seed=TRAIN_SEED)
    _sig_nets, _gue_nets, signaller_agents, guesser_agents, _history = train_independent_iql(
        env, num_episodes=NUM_TRAINING_EPISODES, epsilon_decay_episodes=EPSILON_DECAY_EPISODES,
        seed=TRAIN_SEED,
    )
    run_variant("independent", signaller_agents, guesser_agents, env)


if __name__ == "__main__":
    main()
