"""Signaller agents (Sections 4.2, 4.3).

A signaller observes the hidden items of its neighbouring guessers and emits a
single binary signal m_i in {0,1}. This module provides the reusable, non-random
signaller policies:

* :class:`DeterministicSignaller` -- a lookup table from observation to signal.
  This is the concrete form of a signaller *convention* pi_s (Section 4.6.3) and
  will be how learned/enumerated conventions are represented and replayed.
* :class:`TrackingSignaller` -- emits the item of one specific neighbour. Used to
  build the oracle (Section 4.5.4): a signaller "tracking" a guesser transmits
  that guesser's item verbatim.

The random signaller baseline lives in :mod:`gsg.baselines.random_baseline`.
"""

from __future__ import annotations

from typing import Dict, Tuple

from .base_agent import Observation, SignallerAgent


class DeterministicSignaller(SignallerAgent):
    """A fixed mapping from observation to binary signal (a convention pi_s).

    Parameters
    ----------
    policy:
        Dict mapping each possible observation tuple to a signal in {0,1}.
        Observations not present in the table fall back to ``default``.
    default:
        Signal emitted for unseen observations (defaults to 0).
    """

    def __init__(self, policy: Dict[Observation, int], default: int = 0) -> None:
        self.policy = dict(policy)
        self.default = default

    def act(self, observation: Observation) -> int:
        return self.policy.get(tuple(observation), self.default)


class TrackingSignaller(SignallerAgent):
    """Emit the item of a single tracked neighbour (an oracle building block).

    Parameters
    ----------
    index:
        Position within the observation tuple to transmit. For a signaller with
        one neighbour, ``index = 0`` means "broadcast that guesser's item
        directly" -- the perfect encoding for the one-signaller/one-guesser base
        case (Section 4.5.4).
    """

    def __init__(self, index: int = 0) -> None:
        self.index = index

    def act(self, observation: Observation) -> int:
        return observation[self.index]
