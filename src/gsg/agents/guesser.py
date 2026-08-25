"""Guesser agents (Sections 4.2, 4.3).

A guesser observes the binary signals of its neighbouring signallers and emits a
guess x̂_j of its own hidden item. This module provides the reusable, non-random
guesser policies:

* :class:`DeterministicGuesser` -- a lookup table from received-signal tuple to a
  guess. This is the concrete form of a guesser convention pi_g (Section 4.6.3).
* :class:`CopySignalGuesser` -- guesses whatever a specific incoming signal says.
  Paired with a :class:`~gsg.agents.signaller.TrackingSignaller`, this decodes
  the oracle encoding (Section 4.5.4): if the signaller broadcast the item as-is,
  copying the bit recovers it exactly.

The random guesser baseline lives in :mod:`gsg.baselines.random_baseline`.
"""

from __future__ import annotations

from typing import Dict

from .base_agent import GuesserAgent, Observation


class DeterministicGuesser(GuesserAgent):
    """A fixed mapping from received signals to a guess (a convention pi_g).

    Parameters
    ----------
    policy:
        Dict mapping each possible signal tuple to a guess. Observations not in
        the table fall back to ``default``.
    default:
        Guess emitted for unseen observations (defaults to 0).
    """

    def __init__(self, policy: Dict[Observation, int], default: int = 0) -> None:
        self.policy = dict(policy)
        self.default = default

    def act(self, observation: Observation) -> int:
        return self.policy.get(tuple(observation), self.default)


class CopySignalGuesser(GuesserAgent):
    """Guess the value of a single incoming signal (an oracle building block).

    Parameters
    ----------
    index:
        Position within the received-signal tuple to copy. For a guesser hearing
        one signaller, ``index = 0`` means "guess whatever the light says" -- the
        matching decoder for :class:`~gsg.agents.signaller.TrackingSignaller` in
        the base case (Section 4.5.4).
    """

    def __init__(self, index: int = 0) -> None:
        self.index = index

    def act(self, observation: Observation) -> int:
        return observation[self.index]
