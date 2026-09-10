"""Cumulative action-response matrix, independent variant, K_{2,2}.

Same idea as action_response_matrix_during_training_k22.py, but the windows
*accumulate* instead of being disjoint slices:

    checkpoint 0  ->  episodes 0-749      (same as the disjoint window 0)
    checkpoint 1  ->  episodes 0-1499     (includes everything before it)
    checkpoint 2  ->  episodes 0-2249
    checkpoint 3  ->  episodes 0-2999     (the whole training run)

The point of running this is to *see the difference*: a cumulative matrix
at the final checkpoint still has ~25% of its mass coming from the
pure-exploration early episodes, so the settled load-bearing structure
looks weaker (more diluted) than it actually is by the end. The disjoint
version isolates "what the agent was doing in this slice"; the cumulative
version answers "what has the agent done on average across everything so
far", which drifts toward the true end-state only slowly.

Independent variant only, as requested.

Run:  python experiments/action_response_matrix_cumulative_independent_k22.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from gsg.environment.graphs import SignallingGraph
from gsg.environment.pettingzoo_env import GraphSignallingParallelEnv
from gsg.evaluation.conventions import ActionResponseTally
from gsg.training.trainer import linear_epsilon, train_independent_iql
from gsg.visualization import render_action_response_matrix

NUM_TRAINING_EPISODES = 3000
EPSILON_DECAY_EPISODES = 1800
CHECKPOINTS = [750, 1500, 2250, 3000]  # cumulative: episodes 0 .. (checkpoint - 1)
TRAIN_SEED = 0

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "question1", "figures")


def main() -> None:
    os.makedirs(FIGURES_DIR, exist_ok=True)
    print("=" * 72)
    print("Cumulative action-response matrix, independent variant, K_{2,2}")
    print(f"{NUM_TRAINING_EPISODES} training episodes; checkpoints at {CHECKPOINTS}")
    print("=" * 72)

    env = GraphSignallingParallelEnv(SignallingGraph.complete_bipartite(2, 2), seed=TRAIN_SEED)
    tally = ActionResponseTally(env.graph)
    snapshots: dict[int, dict] = {}

    def on_episode(episode: int, record) -> None:
        tally.add(record)
        if episode + 1 in CHECKPOINTS:
            snapshots[episode + 1] = tally.frequencies()

    train_independent_iql(
        env,
        num_episodes=NUM_TRAINING_EPISODES,
        epsilon_decay_episodes=EPSILON_DECAY_EPISODES,
        on_episode=on_episode,
        seed=TRAIN_SEED,
    )

    for i, checkpoint in enumerate(CHECKPOINTS):
        freqs = snapshots[checkpoint]
        eps_at_checkpoint = linear_epsilon(checkpoint - 1, EPSILON_DECAY_EPISODES)
        peaks = ", ".join(
            f"S{s}->G{g}:{max(table.values()) * 100:.0f}%"
            for (s, g), table in sorted(freqs.items())
        )
        print(f"\ncheckpoint {i}: cumulative over episodes 0-{checkpoint - 1} "
              f"(epsilon now {eps_at_checkpoint:.2f})")
        print(f"    max-cell per edge = {peaks}")

        save_path = os.path.join(
            FIGURES_DIR, f"action_response_cumulative_independent_checkpoint{i}.png"
        )
        render_action_response_matrix(
            freqs,
            title=f"independent (CUMULATIVE) -- episodes 0-{checkpoint - 1}",
            save_path=save_path,
        )
        print(f"    saved -> {os.path.normpath(save_path)}")


if __name__ == "__main__":
    main()
