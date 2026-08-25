"""Shared-parameter IQL actually *learning* the base case (one
signaller, one guesser), played through the PettingZoo adapter.

Sections, in order:

  [1] Train: 1500 episodes of shared-parameter IQL (one network, every
      agent, masked per role -- see agents/networks.py and algorithms/iql.py).
      Progress is reported *live*, every 150 episodes, as training runs --
      not collected silently and dumped all at once at the end. Each
      checkpoint prints the mean reward over that window, then plays one
      fully-greedy example episode (see play_greedy_example() in
      training/trainer.py) so you can watch not just the *number* improve,
      but the actual behaviour that produced it.
  [2] Final greedy evaluation over 1000 episodes, using the exact same
      evaluate() function used for the random/oracle baselines elsewhere in
      this project -- so the learned agent's score is directly comparable
      to the ~0.50 (random) and 1.00 (oracle) numbers from
      demo_pettingzoo_single_pair.py.

The only line in this file that's specific to the base case is the graph
built in make_env() below -- everything else (network shape, per-agent
identity, action masking, training loop, evaluation) is derived generically
from whatever graph and item count that line produces. See
experiments/iql_k22.py, which is this exact file with only that one line
different, for proof.

Run:  python experiments/iql_single_pair.py
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
    graph = SignallingGraph.single_pair()   # <-- the one line that defines the game
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
        num_episodes=1500,
        epsilon_decay_episodes=800,
        log_every=150,
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
