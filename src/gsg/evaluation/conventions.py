"""Convention extraction and enumeration (Section 4.6.3).

This is the machinery behind "what convention did this run actually
converge to" -- not a reward number, but the literal lookup table each
agent settled on. Two levels of detail:

* :func:`greedy_policy` / :func:`extract_all_policies` -- extract *one
  agent's* (or every agent's) greedy mapping from observation to action,
  by directly querying ``agent.act()`` for every possible observation with
  exploration turned off. This is the concrete form of a signaller's
  ``pi_s`` or a guesser's ``pi_g`` (Section 4.6.3).
* :func:`exhaustive_joint_table` -- walk through *every possible episode*
  (every combination of hidden items) and report exactly what the whole
  trained population does for each, deterministically. This works because,
  with exploration off, a trained population's behaviour is a pure function
  of the items -- there's no need to sample episodes and hope to see every
  case; K_{2,2}'s state space is small enough to enumerate completely
  (Section 4.6.1's whole point).

:func:`canonical_id` turns a policy table into a single stable integer, so
two runs can be compared for having converged to the *literal same*
convention, not just a similarly-performing one -- Section 4.6.3 notes a
signaller has 16 possible mappings in the binary case; this is how you'd
tell which one a given run found.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Dict, List, Tuple

from ..environment.graphs import SignallingGraph

Observation = Tuple[int, ...]


def greedy_policy(agent, value_range: int, neighbourhood_size: int) -> Dict[Observation, int]:
    """Query ``agent``'s greedy (epsilon=0) action for every possible
    observation it could receive -- ``value_range`` possibilities in each of
    ``neighbourhood_size`` slots.

    Temporarily forces ``epsilon`` to 0 and ``recording`` (if the agent has
    it -- learning agents do, fixed baselines don't) to ``False``, restoring
    both afterward, so this extracts the agent's actual best policy without
    either sampling a random exploratory action or polluting its replay
    buffer with these probe queries.
    """
    saved_epsilon = getattr(agent, "epsilon", None)
    saved_recording = getattr(agent, "recording", None)
    if saved_epsilon is not None:
        agent.epsilon = 0.0
    if saved_recording is not None:
        agent.recording = False
    try:
        table: Dict[Observation, int] = {}
        for obs in itertools.product(range(value_range), repeat=neighbourhood_size):
            table[obs] = agent.act(obs)
        return table
    finally:
        if saved_epsilon is not None:
            agent.epsilon = saved_epsilon
        if saved_recording is not None:
            agent.recording = saved_recording


def extract_all_policies(
    graph: SignallingGraph,
    num_item_values: int,
    signaller_agents: Dict[int, object],
    guesser_agents: Dict[int, object],
) -> Dict[str, Dict[Observation, int]]:
    """Extract every agent's greedy policy table, keyed by a readable label
    (``"S0"``, ``"G1"``, ...).

    A signaller's observation slots range over the item space
    (``num_item_values``); a guesser's observation slots range over the
    signal space, which is always binary regardless of item count.
    """
    policies: Dict[str, Dict[Observation, int]] = {}
    for s in graph.signallers:
        policies[f"S{s}"] = greedy_policy(signaller_agents[s], num_item_values, graph.out_degree(s))
    for g in graph.guessers:
        policies[f"G{g}"] = greedy_policy(guesser_agents[g], 2, graph.in_degree(g))
    return policies


def canonical_id(policy: Dict[Observation, int], action_range: int, value_range: int, neighbourhood_size: int) -> int:
    """A stable integer id for a policy table -- lets two runs be compared
    for having converged to the *literal same* convention (Section 4.6.3:
    a signaller has ``2 ** 4 = 16`` possible mappings in the binary K_{2,2}
    case), not just a similarly-performing one.

    Observations are visited in the same fixed order ``greedy_policy()``
    used to build the table, so the same convention always produces the
    same id regardless of which run or which agent object produced it.
    """
    obs_order = list(itertools.product(range(value_range), repeat=neighbourhood_size))
    digit_value = 1
    total = 0
    for obs in obs_order:
        total += policy[obs] * digit_value
        digit_value *= action_range
    return total


@dataclass
class WorldOutcome:
    """One fully-enumerated possible episode, played through fixed (greedy)
    policies rather than sampled -- see :func:`exhaustive_joint_table`."""

    items: Dict[int, int]
    signals: Dict[int, int]
    guesses: Dict[int, int]
    correct: Dict[int, bool]
    reward: float


def exhaustive_joint_table(
    graph: SignallingGraph,
    num_item_values: int,
    signaller_agents: Dict[int, object],
    guesser_agents: Dict[int, object],
) -> List[WorldOutcome]:
    """Enumerate *every* possible combination of hidden items and report
    exactly what the (greedy) trained population does for each.

    With exploration off, the whole game is a deterministic function of the
    items, so every possible world can be walked through directly instead
    of sampled -- the "exhaustively characterise the convention space" idea
    from Section 4.6.1, made concrete.
    """
    guesser_ids = graph.guessers
    outcomes: List[WorldOutcome] = []

    all_agents = list(signaller_agents.values()) + list(guesser_agents.values())
    saved = [(getattr(a, "epsilon", None), getattr(a, "recording", None)) for a in all_agents]
    for a in all_agents:
        if hasattr(a, "epsilon"):
            a.epsilon = 0.0
        if hasattr(a, "recording"):
            a.recording = False
    try:
        for items_combo in itertools.product(range(num_item_values), repeat=len(guesser_ids)):
            items = dict(zip(guesser_ids, items_combo))

            signals: Dict[int, int] = {}
            for s in graph.signallers:
                obs = tuple(items[g] for g in graph.out_neighbours(s))
                signals[s] = signaller_agents[s].act(obs)

            guesses: Dict[int, int] = {}
            for g in graph.guessers:
                obs = tuple(signals[s] for s in graph.in_neighbours(g))
                guesses[g] = guesser_agents[g].act(obs)

            correct = {g: guesses[g] == items[g] for g in guesser_ids}
            reward = sum(correct.values()) / len(guesser_ids)
            outcomes.append(WorldOutcome(items, signals, guesses, correct, reward))
    finally:
        for a, (epsilon, recording) in zip(all_agents, saved):
            if epsilon is not None:
                a.epsilon = epsilon
            if recording is not None:
                a.recording = recording

    return outcomes


def convention_summary(outcomes: List[WorldOutcome]) -> str:
    """One-line classification of a joint table, using Section 4.5.6's bands."""
    mean_reward = sum(o.reward for o in outcomes) / len(outcomes)
    if mean_reward >= 0.999:
        return "perfect (complementary conventions)"
    if mean_reward <= 0.501:
        return "at floor (no working convention)"
    if mean_reward < 0.749:
        return f"partial ({mean_reward:.3f})"
    if mean_reward <= 0.751:
        return f"single-bit ceiling ({mean_reward:.3f})"
    return f"exploiting joint channel ({mean_reward:.3f})"
