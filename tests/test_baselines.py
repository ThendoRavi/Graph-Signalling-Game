"""Tests for baselines (Section 4.5): random reward ~ 0.50,
oracle reward = 1.00, on K_{2,2}.

Runnable two ways:
    pytest tests/                   # if pytest is installed
    python tests/test_baselines.py   # falls back to a built-in runner
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from gsg.baselines.oracle_baseline import k22_oracle
from gsg.baselines.random_baseline import RandomGuesser, RandomSignaller
from gsg.environment.graph_signalling_game import GraphSignallingGame
from gsg.environment.graphs import SignallingGraph
from gsg.evaluation.metrics import evaluate


def _k22_env(seed: int) -> GraphSignallingGame:
    return GraphSignallingGame(
        SignallingGraph.complete_bipartite(2, 2), num_item_values=2, seed=seed
    )


def test_k22_oracle_reaches_ceiling():
    """Complementary tracking signallers give every guesser R = 1.00 (Sec 4.5.4)."""
    env = _k22_env(seed=100)
    signallers, guessers = k22_oracle()
    result = evaluate(env, signallers, guessers, episodes=2000, seed=100)
    assert result.mean_reward == 1.0, result
    for g, acc in result.per_guesser_accuracy.items():
        assert acc == 1.0, (g, acc)


def test_k22_random_near_floor():
    """Random signallers/guessers sit at the ~0.50 chance floor on K_{2,2}."""
    env = _k22_env(seed=101)
    signallers = {s: RandomSignaller(seed=100 + s) for s in env.graph.signallers}
    guessers = {g: RandomGuesser(num_item_values=2, seed=200 + g) for g in env.graph.guessers}
    result = evaluate(env, signallers, guessers, episodes=20000, seed=101)
    assert abs(result.mean_reward - 0.5) < 0.03, result


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for t in tests:
        t()
        print(f"  PASS {t.__name__}")
        passed += 1
    print(f"\n{passed}/{len(tests)} baseline tests passed.")


if __name__ == "__main__":
    _run_all()
