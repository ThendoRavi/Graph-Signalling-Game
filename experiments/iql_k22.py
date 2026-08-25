"""Shared-parameter IQL learning K_{2,2} (two signallers, two guessers).

This file is identical to experiments/iql_single_pair.py except for one
line -- the graph built in make_env() below. Nothing about the network
shape, per-agent identity encoding, action masking, training loop, or
evaluation had to change: all of that is derived generically from
env.graph/env.item_space inside training/trainer.py and agents/networks.py.

Because K_{2,2} is fully connected, both signallers see the exact same
neighbourhood every episode -- reaching the 1.00 ceiling requires them to
spontaneously specialise into *complementary* conventions (one tracking
guesser 0, the other guesser 1), a genuinely harder, two-learner
coordination problem than the base case's trivial 1-bit mapping. Unlike
single_pair, convergence to 1.00 here is not guaranteed on every run -- see
the "expect inconsistent convergence" discussion this file's design is
based on. That's not a bug to route around; it's the actual phenomenon
Question 1 of the proposal studies.

Run:  python experiments/iql_k22.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from gsg.environment.graphs import SignallingGraph
from gsg.environment.pettingzoo_env import GraphSignallingParallelEnv
from gsg.evaluation.metrics import evaluate
from gsg.training.trainer import play_greedy_example, train_shared_iql
from gsg.visualization import render_text


def make_env(seed=None) -> GraphSignallingParallelEnv:
    graph = SignallingGraph.complete_bipartite(2, 2)   # <-- the one line that defines the game
    return GraphSignallingParallelEnv(graph, seed=seed)


def main() -> None:
    print("=" * 64)
    print("Shared-parameter IQL")
    print("=" * 64)
    print("[1] Training (live progress every 150 episodes):")

    env = make_env(seed=0)
    print(f"    Graph: {env.graph!r}")

    def on_log(episodes_so_far, window_rewards, network, signaller_agents, guesser_agents) -> None:
        window_start = episodes_so_far - len(window_rewards) + 1
        mean_reward = sum(window_rewards) / len(window_rewards)
        print(f"\n    episodes {window_start:>5}-{episodes_so_far:<5}: "
              f"mean reward = {mean_reward:.3f}")

        example = play_greedy_example(env, signaller_agents, guesser_agents)
        for line in render_text(env.graph, example).splitlines():
            print(f"    {line}")

    network, signaller_agents, guesser_agents, history = train_shared_iql(
        env,
        num_episodes=4000,
        epsilon_decay_episodes=2500,
        log_every=400,
        on_log=on_log,
        seed=0,
    )
    print(f"\n    Done: {len(history.episode_rewards)} episodes, "
          f"{len(history.losses)} gradient steps.")

    print("\n[2] Final greedy evaluation over 1000 episodes:")
    for agent in list(signaller_agents.values()) + list(guesser_agents.values()):
        agent.epsilon = 0.0  # fully exploit -- no more exploration
    eval_env = make_env(seed=99)
    result = evaluate(
        eval_env,
        signaller_agents,
        guesser_agents,
        episodes=1000,
        seed=99,
    )
    print("   ", result)


if __name__ == "__main__":
    main()
