"""Demo: the Graph Signalling Game base case (one signaller, one guesser),
played through the PettingZoo ParallelEnv adapter.

Same four sections as demo_pettingzoo_k22.py, on the smaller graph:

  [1] PettingZoo's own compliance check (``parallel_api_test``).
  [2] One episode, played through env.run_episode() and rendered as text.
  [3] The oracle baseline over 1000 episodes -- the ceiling.
  [4] The random baseline over 1000 episodes -- the floor.

The only real differences from the K_{2,2} demo are the graph
(``SignallingGraph.single_pair()`` instead of ``complete_bipartite(2, 2)``),
the oracle (``single_pair_oracle()`` instead of ``k22_oracle()``), and that
each agent dict has one entry (id 0) instead of two -- everything else,
including the ``evaluate()`` call, is identical, because both graphs are
driven through the same generic environment and adapter code.

What "random" and "oracle" are
--------------------------------
Neither one *learns* -- they're fixed reference players used to sanity-check
the environment, not agents under study:

* **random** ignores whatever it observes and answers by coin flip. It marks
  the floor: with one guesser right half the time by pure chance, the
  expected team reward is 0.50.
* **oracle** is a hand-coded *perfect* strategy -- the signaller broadcasts
  the guesser's item verbatim, the guesser copies the bit. It marks the
  ceiling: confirms 1.00 is actually reachable in this environment.

Run:  python experiments/demo_pettingzoo_single_pair.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from gsg.baselines.oracle_baseline import single_pair_oracle
from gsg.baselines.random_baseline import RandomGuesser, RandomSignaller
from gsg.environment.graphs import SignallingGraph
from gsg.environment.pettingzoo_env import GraphSignallingParallelEnv
from gsg.evaluation.metrics import evaluate
from gsg.visualization import render_text


def make_env(seed=None) -> GraphSignallingParallelEnv:
    graph = SignallingGraph.single_pair()
    return GraphSignallingParallelEnv(graph, seed=seed)


def run_compliance_check() -> None:
    print("[1] PettingZoo compliance check")
    try:
        from pettingzoo.test import parallel_api_test
    except ImportError as exc:
        print(f"    Skipped: {exc}")
        return
    parallel_api_test(make_env(), num_cycles=200)
    print("    PASSED: env satisfies the PettingZoo ParallelEnv contract.")


def run_one_episode() -> None:
    print("\n[2] One episode, played through env.run_episode()")
    env = make_env(seed=0)
    signallers = {0: RandomSignaller(seed=0)}
    guessers = {0: RandomGuesser(seed=1)}
    record = env.run_episode(signallers, guessers)
    print(render_text(env.graph, record))


def run_oracle() -> None:
    print("\n[3] Oracle baseline over 1000 episodes (should be perfect):")
    signallers, guessers = single_pair_oracle()
    result = evaluate(make_env(seed=10), signallers, guessers, episodes=1000, seed=10)
    print("   ", result)


def run_random() -> None:
    print("\n[4] Random baseline over 1000 episodes (should be ~0.50):")
    signallers = {0: RandomSignaller(seed=100)}
    guessers = {0: RandomGuesser(seed=101)}
    result = evaluate(make_env(seed=20), signallers, guessers, episodes=1000, seed=20)
    print("   ", result)


if __name__ == "__main__":
    print("=" * 64)
    print("Graph Signalling Game via PettingZoo -- base case (1 signaller, 1 guesser)")
    print("=" * 64)
    run_compliance_check()
    run_one_episode()
    run_oracle()
    run_random()
