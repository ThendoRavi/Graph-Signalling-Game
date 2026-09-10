"""Action Response Matrix built from *live training play*, K_{2,2}.

Companion to action_response_matrix_k22.py, which trained each variant to
convergence and then measured 1000 episodes with exploration *off* -- so the
trained agents were fully deterministic (exact 0% cells) while the random
baseline was fully stochastic (no 0% cells anywhere). That's a real
structural difference, but it's mostly explained by "one is a frozen
function, the other is a coin flip", which makes the comparison a bit
apples-to-oranges.

This version instead tallies the (signal, guess) pairs from the *actual
training episodes* -- exploration on, epsilon decaying on its normal
schedule -- split into a few windows across training. So you see the
behaviour of an agent that is *still learning and still exploring*, which
is the same kind of thing the random baseline is (stochastic), just with a
structure that emerges as training proceeds. Early windows (epsilon near 1)
should look close to the random baseline -- roughly flat; later windows
should show the load-bearing / vestigial edge structure sharpening, without
ever hitting exact 0% because there's always some residual exploration.

Run:  python experiments/action_response_matrix_during_training_k22.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from gsg.baselines.random_baseline import RandomGuesser, RandomSignaller
from gsg.environment.graphs import SignallingGraph
from gsg.environment.pettingzoo_env import GraphSignallingParallelEnv
from gsg.evaluation.conventions import ActionResponseTally
from gsg.training.trainer import (
    linear_epsilon,
    train_independent_iql,
    train_shared_iql,
    train_two_brain_iql,
)
from gsg.visualization import render_action_response_matrix

NUM_TRAINING_EPISODES = 3000
EPSILON_DECAY_EPISODES = 1800
NUM_WINDOWS = 4
WINDOW_SIZE = NUM_TRAINING_EPISODES // NUM_WINDOWS
TRAIN_SEED = 0

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "question1", "figures")


def make_env(seed=None) -> GraphSignallingParallelEnv:
    graph = SignallingGraph.complete_bipartite(2, 2)
    return GraphSignallingParallelEnv(graph, seed=seed)


def windowed_tallies(graph) -> list[ActionResponseTally]:
    return [ActionResponseTally(graph) for _ in range(NUM_WINDOWS)]


def make_on_episode(tallies):
    def on_episode(episode: int, record) -> None:
        window = min(episode // WINDOW_SIZE, NUM_WINDOWS - 1)
        tallies[window].add(record)
    return on_episode


def print_and_render(name: str, tallies) -> None:
    print(f"\n--- {name} ---")
    for i, tally in enumerate(tallies):
        lo = i * WINDOW_SIZE
        hi = lo + tally.num_episodes - 1
        eps_lo = linear_epsilon(lo, EPSILON_DECAY_EPISODES)
        eps_hi = linear_epsilon(hi, EPSILON_DECAY_EPISODES)
        freqs = tally.frequencies()

        # A compact "how peaked is each edge" readout: max cell % per edge.
        peaks = ", ".join(
            f"S{s}->G{g}:{max(table.values()) * 100:.0f}%"
            for (s, g), table in sorted(freqs.items())
        )
        print(f"    window {i} (episodes {lo}-{hi}, epsilon {eps_lo:.2f}->{eps_hi:.2f}): "
              f"max-cell per edge = {peaks}")

        save_path = os.path.join(FIGURES_DIR, f"action_response_training_{name}_window{i}.png")
        render_action_response_matrix(
            freqs,
            title=f"{name} -- training episodes {lo}-{hi} "
                  f"(epsilon {eps_lo:.2f} -> {eps_hi:.2f})",
            save_path=save_path,
        )
    print(f"    saved {NUM_WINDOWS} window figures -> "
          f"{os.path.normpath(os.path.join(FIGURES_DIR, f'action_response_training_{name}_window*.png'))}")


def main() -> None:
    os.makedirs(FIGURES_DIR, exist_ok=True)
    print("=" * 72)
    print("Action Response Matrix from live training play, K_{2,2}")
    print(f"{NUM_TRAINING_EPISODES} training episodes, {NUM_WINDOWS} windows of "
          f"{WINDOW_SIZE} episodes each")
    print("=" * 72)

    # Random baseline: no training, but run the same number of episodes and
    # window them the same way -- expected to stay flat in every window.
    env = make_env(seed=TRAIN_SEED)
    tallies = windowed_tallies(env.graph)
    on_ep = make_on_episode(tallies)
    signaller_agents = {0: RandomSignaller(seed=0), 1: RandomSignaller(seed=1)}
    guesser_agents = {0: RandomGuesser(seed=2), 1: RandomGuesser(seed=3)}
    for episode in range(NUM_TRAINING_EPISODES):
        record = env.run_episode(signaller_agents, guesser_agents)
        on_ep(episode, record)
    print_and_render("random", tallies)

    # Shared-brain
    env = make_env(seed=TRAIN_SEED)
    tallies = windowed_tallies(env.graph)
    train_shared_iql(
        env, num_episodes=NUM_TRAINING_EPISODES, epsilon_decay_episodes=EPSILON_DECAY_EPISODES,
        on_episode=make_on_episode(tallies), seed=TRAIN_SEED,
    )
    print_and_render("shared", tallies)

    # Two-brain
    env = make_env(seed=TRAIN_SEED)
    tallies = windowed_tallies(env.graph)
    train_two_brain_iql(
        env, num_episodes=NUM_TRAINING_EPISODES, epsilon_decay_episodes=EPSILON_DECAY_EPISODES,
        on_episode=make_on_episode(tallies), seed=TRAIN_SEED,
    )
    print_and_render("two_brain", tallies)

    # Independent
    env = make_env(seed=TRAIN_SEED)
    tallies = windowed_tallies(env.graph)
    train_independent_iql(
        env, num_episodes=NUM_TRAINING_EPISODES, epsilon_decay_episodes=EPSILON_DECAY_EPISODES,
        on_episode=make_on_episode(tallies), seed=TRAIN_SEED,
    )
    print_and_render("independent", tallies)


if __name__ == "__main__":
    main()
