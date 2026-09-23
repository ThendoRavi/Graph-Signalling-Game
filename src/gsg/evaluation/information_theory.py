"""Information-theoretic metrics (Section~sec:eval-criteria's "information-theoretic
metrics" group): mutual information, channel utilisation, and entropy reduction.

Every quantity here is computed *exactly*, not estimated from samples. The
trick is the same one :func:`gsg.evaluation.conventions.exhaustive_joint_table`
already uses: because hidden items are i.i.d. uniform and a converged greedy
policy is a deterministic function of its input, walking every one of the
``num_item_values ** num_guessers`` possible worlds (each equally likely)
gives the *exact* joint distribution of any (signal, item) or (observation,
item) pair a policy induces -- no Monte Carlo noise, no confidence interval,
just arithmetic on a small, exhaustively-enumerated table. This module takes
the ``List[WorldOutcome]`` that :func:`exhaustive_joint_table` already
produces and reduces it to three families of number:

* :func:`signaller_info_metrics` -- for every signaller ``i``: its channel
  utilisation ``H(M_i)`` (a binary channel's capacity is exactly 1 bit, so
  "H(M_i) / log2(|M|)" reduces to ``H(M_i)`` itself, already normalised), and
  for every guesser ``j`` it serves, the mutual information ``I(M_i; x_j)``
  between the bit it emits and that guesser's hidden item.
* :func:`guesser_info_metrics` -- for every guesser ``j``: the entropy
  reduction ``I(o_j; x_j)`` its received signal vector achieves about its own
  hidden item -- how much of the guesser's own uncertainty its observation
  actually resolves, out of the ``log2(num_item_values)`` bits available.

A signaller with ``I(M_i; x_j) approx 0`` for every ``j`` is not
communicating about ``x_j`` at all, regardless of what its channel
utilisation says; a guesser with ``I(o_j; x_j) approx 0`` is not using its
observation, regardless of how informative the signals reaching it actually
are (Section~sec:results-scaling-crosscutting's "failing guesser" pattern is
exactly this: entropy reduction near 0 for that guesser, even when the
edges feeding it individually carry real information about *other*
guessers' items).
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Hashable, List, Tuple

from ..environment.graphs import SignallingGraph
from .conventions import WorldOutcome


def entropy_bits(counts: Counter, total: int) -> float:
    """Shannon entropy, in bits, of the distribution ``counts / total``."""
    if total == 0:
        raise ValueError("entropy_bits() called with total == 0")
    h = 0.0
    for count in counts.values():
        if count == 0:
            continue
        p = count / total
        h -= p * math.log2(p)
    return h


def mutual_information_bits(joint_counts: "Counter[Tuple[Hashable, Hashable]]", total: int) -> float:
    """``I(X; Y)`` in bits, from joint counts of ``(x, y)`` pairs, via
    ``I(X;Y) = H(X) + H(Y) - H(X,Y)`` -- numerically simpler and just as
    exact as the log-ratio form for the small, exhaustively-enumerated
    tables this module works with.
    """
    x_counts: Counter = Counter()
    y_counts: Counter = Counter()
    for (x, y), count in joint_counts.items():
        x_counts[x] += count
        y_counts[y] += count
    h_xy = entropy_bits(joint_counts, total)
    h_x = entropy_bits(x_counts, total)
    h_y = entropy_bits(y_counts, total)
    return h_x + h_y - h_xy


@dataclass
class SignallerInfoMetrics:
    """Per-signaller information-theoretic summary."""

    channel_utilisation: float                 # H(M_i), bits, out of 1 max
    mutual_information: Dict[int, float]        # guesser_id -> I(M_i; x_j), bits


@dataclass
class GuesserInfoMetrics:
    """Per-guesser information-theoretic summary."""

    entropy_reduction: float                    # I(o_j; x_j), bits


def signaller_info_metrics(
    graph: SignallingGraph, outcomes: List[WorldOutcome]
) -> Dict[int, SignallerInfoMetrics]:
    """Channel utilisation and per-edge mutual information for every signaller,
    computed exactly from an :func:`exhaustive_joint_table` batch."""
    total = len(outcomes)
    result: Dict[int, SignallerInfoMetrics] = {}
    for s in graph.signallers:
        signal_counts = Counter(o.signals[s] for o in outcomes)
        channel_utilisation = entropy_bits(signal_counts, total)

        mutual_information: Dict[int, float] = {}
        for j in graph.out_neighbours(s):
            joint = Counter((o.signals[s], o.items[j]) for o in outcomes)
            mutual_information[j] = mutual_information_bits(joint, total)

        result[s] = SignallerInfoMetrics(
            channel_utilisation=channel_utilisation, mutual_information=mutual_information
        )
    return result


def guesser_info_metrics(
    graph: SignallingGraph, outcomes: List[WorldOutcome]
) -> Dict[int, GuesserInfoMetrics]:
    """Entropy reduction ``I(o_j; x_j)`` for every guesser, computed exactly
    from an :func:`exhaustive_joint_table` batch."""
    total = len(outcomes)
    result: Dict[int, GuesserInfoMetrics] = {}
    for g in graph.guessers:
        in_signallers = graph.in_neighbours(g)
        joint = Counter(
            (tuple(o.signals[s] for s in in_signallers), o.items[g]) for o in outcomes
        )
        result[g] = GuesserInfoMetrics(entropy_reduction=mutual_information_bits(joint, total))
    return result
