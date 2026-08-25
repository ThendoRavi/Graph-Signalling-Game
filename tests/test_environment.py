"""Tests for the Graph Signalling Game environment (Section 4.2).

These check that the *rules* are in place on the base one-signaller/one-guesser
graph: episode phase ordering, observation construction, the reward structure
(Eq. 4.4), and the two anchor performance points -- random ~= 0.50 (floor,
Section 4.5.1) and oracle == 1.00 (ceiling, Section 4.5.4).

Runnable two ways:
    pytest tests/                 # if pytest is installed
    python tests/test_environment.py   # falls back to a built-in runner
"""

from __future__ import annotations

import os
import sys

# Make ``src/`` importable when run directly (no install step needed).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from gsg.baselines.oracle_baseline import single_pair_oracle
from gsg.baselines.random_baseline import RandomGuesser, RandomSignaller
from gsg.environment.graph_signalling_game import GraphSignallingGame
from gsg.environment.graphs import SignallingGraph
from gsg.evaluation.metrics import evaluate


def _single_pair_env(num_item_values: int = 2, seed: int = 0) -> GraphSignallingGame:
    return GraphSignallingGame(
        SignallingGraph.single_pair(), num_item_values=num_item_values, seed=seed
    )


def test_graph_structure():
    """The base graph is one signaller connected to one guesser."""
    g = SignallingGraph.single_pair()
    g.validate()
    assert g.signallers == [0]
    assert g.guessers == [0]
    assert g.edges == [(0, 0)]
    assert g.out_neighbours(0) == [0]
    assert g.in_neighbours(0) == [0]
    assert g.has_full_guesser_coverage()


def test_coverage_constraint_enforced():
    """A guesser with no incoming signal must be rejected (Section 4.7.2)."""
    g = SignallingGraph()
    g.add_signaller()
    g.add_guesser()  # never connected
    try:
        GraphSignallingGame(g)
    except ValueError:
        return
    raise AssertionError("expected ValueError for uncovered guesser")


def test_observation_shapes():
    """Signaller sees the guesser's item; guesser sees the signal bit."""
    env = _single_pair_env(seed=1)
    signaller_obs = env.reset()
    # One neighbour -> length-1 observation tuple holding a binary item.
    assert set(signaller_obs) == {0}
    assert len(signaller_obs[0]) == 1
    assert signaller_obs[0][0] in (0, 1)

    guesser_obs = env.submit_signals({0: 1})
    assert guesser_obs[0] == (1,)  # the guesser heard the single signal "1"


def test_phase_ordering_enforced():
    """Methods must be called reset -> signals -> guesses, not out of order."""
    env = _single_pair_env(seed=2)
    # Cannot signal before reset.
    try:
        env.submit_signals({0: 0})
    except RuntimeError:
        pass
    else:
        raise AssertionError("submit_signals before reset should fail")

    env.reset()
    # Cannot guess before signalling.
    try:
        env.submit_guesses({0: 0})
    except RuntimeError:
        pass
    else:
        raise AssertionError("submit_guesses before signals should fail")


def test_invalid_action_rejected():
    """Out-of-range or wrong-agent actions are rejected."""
    env = _single_pair_env(seed=3)
    env.reset()
    try:
        env.submit_signals({0: 2})  # 2 is not a valid binary signal
    except ValueError:
        pass
    else:
        raise AssertionError("out-of-range signal should fail")


def test_reward_structure():
    """Reward is 1.0 iff the guess matches the hidden item (Eq. 4.4)."""
    env = _single_pair_env(seed=4)
    # Drive an episode manually so we know the true item.
    signaller_obs = env.reset(seed=4)
    true_item = signaller_obs[0][0]
    env.submit_signals({0: 0})
    reward_correct, rec_correct = env.submit_guesses({0: true_item})
    assert reward_correct == 1.0 and rec_correct.correct[0] is True

    signaller_obs = env.reset(seed=4)
    true_item = signaller_obs[0][0]
    env.submit_signals({0: 0})
    reward_wrong, rec_wrong = env.submit_guesses({0: 1 - true_item})
    assert reward_wrong == 0.0 and rec_wrong.correct[0] is False


def test_oracle_reaches_ceiling():
    """The hand-coded oracle scores a perfect 1.00 (Section 4.5.4)."""
    env = _single_pair_env(seed=5)
    signallers, guessers = single_pair_oracle()
    result = evaluate(env, signallers, guessers, episodes=2000, seed=5)
    assert result.mean_reward == 1.0, result


def test_random_near_floor():
    """Random agents sit at the ~0.50 chance floor (Section 4.5.1)."""
    env = _single_pair_env(seed=6)
    signallers = {0: RandomSignaller(seed=6)}
    guessers = {0: RandomGuesser(num_item_values=2, seed=7)}
    result = evaluate(env, signallers, guessers, episodes=20000, seed=6)
    assert abs(result.mean_reward - 0.5) < 0.03, result


# --- built-in runner (used when pytest is unavailable) ---------------------

def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for t in tests:
        t()
        print(f"  PASS {t.__name__}")
        passed += 1
    print(f"\n{passed}/{len(tests)} environment tests passed.")


if __name__ == "__main__":
    _run_all()
