"""Random baseline (Section 4.5.1).

Signallers emit m ~ Bernoulli(0.5) and guessers emit a uniformly random guess,
both ignoring their observations entirely. Because items are i.i.d. uniform and
guesses are independent of them, the expected team reward is exactly the chance
floor ``R = 1 / num_item_values`` (0.50 in the binary case). Any trained agent
must clear this floor to have learned anything at all -- so this baseline is the
"no convention" reference the environment check compares against.

Each agent owns its own seeded RNG so runs are reproducible and different agents
don't share a random stream.
"""

from __future__ import annotations

import random
from typing import Optional

from ..agents.base_agent import GuesserAgent, Observation, SignallerAgent


class RandomSignaller(SignallerAgent):
    """Emit a uniformly random bit, ignoring the observation (Section 4.5.1)."""

    def __init__(self, seed: Optional[int] = None) -> None:
        self._rng = random.Random(seed)

    def act(self, observation: Observation) -> int:
        return self._rng.randrange(2)


class RandomGuesser(GuesserAgent):
    """Guess uniformly at random, ignoring the signals (Section 4.5.1).

    Parameters
    ----------
    num_item_values:
        Size of the guess space (2 = binary, 3 = ternary). The guess is drawn
        uniformly from ``{0, ..., num_item_values - 1}``.
    """

    def __init__(self, num_item_values: int = 2, seed: Optional[int] = None) -> None:
        self.num_item_values = num_item_values
        self._rng = random.Random(seed)

    def act(self, observation: Observation) -> int:
        return self._rng.randrange(self.num_item_values)
