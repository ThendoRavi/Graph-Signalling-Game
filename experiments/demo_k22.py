"""Demo: the Graph Signalling Game on K_{2,2}, played through the PettingZoo
ParallelEnv adapter.

Four sections, in order:

  [1] PettingZoo's own compliance check (``parallel_api_test``) -- an
      independent, third-party confirmation that the adapter actually
      satisfies the ``ParallelEnv`` contract.
  [2] One episode, played through env.run_episode() and rendered with the
      same readable text view used everywhere else in this project.
  [3] The oracle baseline over 1000 episodes -- the ceiling.
  [4] The random baseline over 1000 episodes -- the floor.

Note there's no PettingZoo-specific evaluation code here: sections [3] and
[4] call the *exact same* ``gsg.evaluation.metrics.evaluate()`` used
elsewhere in this project against the raw ``GraphSignallingGame``. That
works because ``GraphSignallingParallelEnv`` exposes ``.graph``,
``.item_space``, and ``.run_episode(...)`` with the same names, shapes, and
return types as the engine it wraps -- see the "parity" section of
pettingzoo_env.py.

What "random" and "oracle" are
--------------------------------
Neither one *learns* -- they're fixed reference players used to sanity-check
the environment, not agents under study:

* **random** ignores whatever it observes and answers by coin flip. It marks
  the floor: with 2 guessers each independently right half the time by pure
  chance, the expected team reward is 0.50. Any real learner should clear
  this easily -- if it can't, something's wrong with training, not the game.
* **oracle** is a hand-coded *perfect* strategy (signaller 0 always tracks
  guesser 0's item, signaller 1 always tracks guesser 1's), not something
  that was learned. It marks the ceiling: confirms 1.00 is actually
  reachable in this environment, so if a future learner falls short of it,
  the gap is the learner's, not a flaw in the game itself.

Run:  python experiments/demo_pettingzoo_k22.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from gsg.baselines.oracle_baseline import k22_oracle
from gsg.baselines.random_baseline import RandomGuesser, RandomSignaller
from gsg.environment.graphs import SignallingGraph
from gsg.environment.pettingzoo_env import GraphSignallingParallelEnv
from gsg.evaluation.metrics import evaluate
from gsg.visualization import render_text


def make_env(seed=None) -> GraphSignallingParallelEnv:
    graph = SignallingGraph.complete_bipartite(2, 2)
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
    signallers = {0: RandomSignaller(seed=0), 1: RandomSignaller(seed=1)}
    guessers = {0: RandomGuesser(seed=2), 1: RandomGuesser(seed=3)}
    record = env.run_episode(signallers, guessers)
    print(render_text(env.graph, record))


def run_oracle() -> None:
    print("\n[3] Oracle baseline over 1000 episodes (should be perfect):")
    signallers, guessers = k22_oracle()
    result = evaluate(make_env(seed=10), signallers, guessers, episodes=1000, seed=10)
    print("   ", result)


def run_random() -> None:
    print("\n[4] Random baseline over 1000 episodes (should be ~0.50):")
    signallers = {0: RandomSignaller(seed=100), 1: RandomSignaller(seed=101)}
    guessers = {0: RandomGuesser(seed=102), 1: RandomGuesser(seed=103)}
    result = evaluate(make_env(seed=20), signallers, guessers, episodes=1000, seed=20)
    print("   ", result)


if __name__ == "__main__":
    print("=" * 64)
    print("Graph Signalling Game via PettingZoo -- K_{2,2}")
    print("=" * 64)
    run_compliance_check()
    run_one_episode()
    run_oracle()
    run_random()
