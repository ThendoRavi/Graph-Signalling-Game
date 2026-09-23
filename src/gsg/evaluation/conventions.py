"""Convention extraction and enumeration (Section 4.6.3).

This is the machinery behind "what convention did this run actually
converge to" -- not a reward number, but the literal lookup table each
agent settled on. Two different methods, answering two different questions:

* :func:`greedy_policy` / :func:`extract_all_policies` / :func:`exhaustive_joint_table`
  -- **what would the agent do, for every possible input, assuming it's
  deterministic?** Extracted by directly querying ``agent.act()`` with
  exploration forced off, for every possible observation -- the concrete
  form of a signaller's ``pi_s`` or a guesser's ``pi_g`` (Section 4.6.3).
  This is exact and exhaustive, but it assumes the policy *is* a fixed
  function of the observation -- which is true for a converged, greedy IQL
  agent, but not for a genuinely stochastic one (a policy still exploring,
  or a random baseline that never looks at its input at all). For those,
  querying each input once can show a "clean-looking" result purely by
  luck of the draw, not because a real rule exists.
* :func:`all_edges_action_response_frequencies` / :class:`ActionResponseTally`
  -- **what did the agent actually do, how often, across real play?** Built
  empirically by *playing* real episodes (sampling items the normal way,
  not enumerating them) and tallying how often each (signaller's signal,
  guesser's guess) pair actually occurred, per edge. This is the "Action
  Response Matrix" idea (in the spirit of the cross-play matrices in the
  other-play / any-play literature): a converged, information-carrying edge
  concentrates most of its probability mass into one or two cells; an edge
  carrying no real information spreads close to an even 25% across all four
  -- and because it's built from many samples rather than one query per
  cell, that flatness shows up robustly instead of looking accidentally
  "clean" the way a single-query table can. The *function* runs its own
  dedicated batch with exploration forced off (the settled greedy policy);
  the *class* is fed episodes from outside -- e.g. a trainer's ``on_episode``
  hook -- so it can measure behaviour *while the agent is still learning*,
  exploration and all, which is what makes a like-for-like comparison
  against a (also-stochastic) random baseline meaningful.

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
Edge = Tuple[int, int]


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


def joint_signaller_convention_id(
    graph: SignallingGraph,
    num_item_values: int,
    signaller_agents: Dict[int, object],
) -> Tuple[int, ...]:
    """A single hashable id for the *whole population's* signaller-side
    convention: one :func:`canonical_id` per signaller, in ``graph.signallers``
    order.

    Where :func:`canonical_id` answers "which of the 16 (or more, at higher
    item counts) possible mappings did *this one* signaller find", this is
    the generalisation to an entire population: two independently-trained
    runs get the *same* joint id iff every signaller found the literal same
    greedy mapping, not merely an equally-rewarding one. This is what makes
    "how many distinct conventions did P runs land on, and how often does
    each recur" (Section~4.6.3's O1.4 consistency question) a well-posed,
    countable question rather than a matter of eyeballing reward numbers or
    heatmaps one run at a time.
    """
    ids = []
    for s in graph.signallers:
        k = graph.out_degree(s)
        policy = greedy_policy(signaller_agents[s], num_item_values, k)
        ids.append(
            canonical_id(policy, action_range=2, value_range=num_item_values, neighbourhood_size=k)
        )
    return tuple(ids)


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


def all_edges_action_response_frequencies(
    env,
    signaller_agents: Dict[int, object],
    guesser_agents: Dict[int, object],
    num_episodes: int = 1000,
) -> Dict[Edge, Dict[Tuple[int, int], float]]:
    """Play ``num_episodes`` *real* episodes and, for every (signaller,
    guesser) edge in ``env.graph``, tally how often each (signal, guess)
    pair actually occurred -- built from one shared batch of episodes, so
    every edge's table reflects the same underlying sample rather than
    independently-resampled runs.

    This is the empirical "Action Response Matrix": not what a fixed policy
    *would* do for a given input (see :func:`exhaustive_joint_table`), but
    what actually came out, how often, across genuine play. Forces every
    agent greedy (``epsilon=0``) and non-recording first, restoring both
    afterward, for the same reason :func:`greedy_policy` does -- this
    measures the *settled* behaviour, not a mix of exploration and
    exploitation, and it must not pollute any agent's replay buffer with
    these measurement episodes. Baseline agents with no ``epsilon`` (the
    oracle, or a genuinely random policy that never looks at its input at
    all) are left exactly as they are -- there's no "settled" state to force
    for them, which is itself the point: a random baseline's table will
    stay close to a flat 25% in every cell no matter how long you sample it,
    because there was never a rule to settle into.

    Returns ``{(signaller_id, guesser_id): {(signal, guess): fraction}}``,
    each inner dict's four fractions summing to 1.0.
    """
    edges = env.graph.edges
    tallies: Dict[Edge, Dict[Tuple[int, int], int]] = {
        edge: {(a, r): 0 for a in (0, 1) for r in (0, 1)} for edge in edges
    }

    all_agents = list(signaller_agents.values()) + list(guesser_agents.values())
    saved = [(getattr(a, "epsilon", None), getattr(a, "recording", None)) for a in all_agents]
    for a in all_agents:
        if hasattr(a, "epsilon"):
            a.epsilon = 0.0
        if hasattr(a, "recording"):
            a.recording = False
    try:
        for _ in range(num_episodes):
            record = env.run_episode(signaller_agents, guesser_agents)
            for s, g in edges:
                tallies[(s, g)][(record.signals[s], record.guesses[g])] += 1
    finally:
        for a, (epsilon, recording) in zip(all_agents, saved):
            if epsilon is not None:
                a.epsilon = epsilon
            if recording is not None:
                a.recording = recording

    return {
        edge: {key: count / num_episodes for key, count in tally.items()}
        for edge, tally in tallies.items()
    }


def edge_action_response_frequencies(
    env,
    signaller_id: int,
    guesser_id: int,
    signaller_agents: Dict[int, object],
    guesser_agents: Dict[int, object],
    num_episodes: int = 1000,
) -> Dict[Tuple[int, int], float]:
    """Like :func:`all_edges_action_response_frequencies`, for a single
    (``signaller_id``, ``guesser_id``) edge only."""
    return all_edges_action_response_frequencies(
        env, signaller_agents, guesser_agents, num_episodes
    )[(signaller_id, guesser_id)]


class ActionResponseTally:
    """Accumulates (signaller's signal, guesser's guess) counts per edge,
    one episode at a time.

    Where :func:`all_edges_action_response_frequencies` runs its own
    dedicated batch of episodes with exploration forced *off* (measuring the
    settled greedy policy), this is fed :class:`EpisodeRecord`\\ s from
    *outside* -- e.g. from a trainer's ``on_episode`` hook, one per training
    episode, with exploration still on and epsilon at whatever the schedule
    currently says. That's the difference between "what did the converged
    agent do" and "what did the agent do *while it was still learning*" --
    and the latter is what makes a like-for-like comparison against a
    (also-stochastic) random baseline meaningful.

    :meth:`frequencies` returns the same ``{edge: {(signal, guess):
    fraction}}`` shape the batch function does, ready to hand straight to
    :func:`gsg.visualization.render_action_response_matrix`.
    """

    def __init__(self, graph: SignallingGraph) -> None:
        self._edges: List[Edge] = list(graph.edges)
        self._counts: Dict[Edge, Dict[Tuple[int, int], int]] = {
            edge: {(a, r): 0 for a in (0, 1) for r in (0, 1)} for edge in self._edges
        }
        self._total = 0

    def add(self, record) -> None:
        """Fold one played episode's record into the running tally."""
        for s, g in self._edges:
            self._counts[(s, g)][(record.signals[s], record.guesses[g])] += 1
        self._total += 1

    @property
    def num_episodes(self) -> int:
        return self._total

    def frequencies(self) -> Dict[Edge, Dict[Tuple[int, int], float]]:
        if self._total == 0:
            raise ValueError("frequencies() called before any episodes were add()ed")
        return {
            edge: {key: count / self._total for key, count in counts.items()}
            for edge, counts in self._counts.items()
        }
