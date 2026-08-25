"""Observation and action space definitions.

A single tiny value type, :class:`DiscreteSpace`

* a **hidden item**   x_j  lives in ``{0, 1}``     (binary)``
* a **binary signal** m_i  lives in ``{0, 1}``     (the one-bit channel)
* a **guess**         x̂_j  lives in the item space (the guesser names its item)

"""

from __future__ import annotations

import random


class DiscreteSpace:
    """A finite set of integer values ``{0, 1, ..., n-1}``.

    Parameters
    ----------
    n:
        Number of distinct values. ``n = 2`` gives the binary space used for
        signals and binary items; ``n = 3`` gives the ternary item space.
    name:
        Human-readable label, used only in error messages / rendering.
    """

    def __init__(self, n: int, name: str = "discrete") -> None:
        if n < 1:
            raise ValueError(f"DiscreteSpace needs n >= 1, got {n}")
        self.n = n
        self.name = name

    def contains(self, value: object) -> bool:
        """True iff ``value`` is a valid element of this space."""
        return isinstance(value, int) and 0 <= value < self.n

    def sample(self, rng: random.Random) -> int:
        """Draw one value uniformly at random.

        """
        return rng.randrange(self.n)

    def __len__(self) -> int:
        return self.n

    def __repr__(self) -> str:
        return f"DiscreteSpace({self.name}, n={self.n})"


# Convenient shared instance for the common binary case (signals / binary items).
BINARY = DiscreteSpace(2, name="binary")
