"""Abstract base agent (Section 4.3).

Agents are the *decision-makers*, kept completely separate from the graph nodes
they sit on. The environment only ever calls ``act(observation)``, so any policy
-- random, hand-coded oracle, or (later) a learned Q-network -- plugs into the
same slot. This separation is what makes the system OOP-extensible: adding an
agent type never touches the environment, and adding a graph node never touches
the agents.

Roles are fixed and non-interchangeable (Section 4.3): a signaller's observation
is a tuple of *items* and its action is a *binary signal*; a guesser's
observation is a tuple of *signals* and its action is a *guess*. The two base
classes below exist to make that distinction explicit in the type system, even
though they share the same interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple

Observation = Tuple[int, ...]


class Agent(ABC):
    """Common interface for every agent in the game.

    Subclasses implement :meth:`act`. The learning hooks (:meth:`observe_reward`,
    :meth:`reset_episode`) are no-ops by default so that non-learning agents
    (random, oracle) need not implement them; the future IQL/QMIX agents will
    override them.
    """

    #: Overridden by the role base classes below.
    role: str = "agent"

    @abstractmethod
    def act(self, observation: Observation) -> int:
        """Map a local observation to a discrete action."""

    def observe_reward(self, reward: float) -> None:
        """Learning hook: receive the shared team reward. No-op by default."""

    def reset_episode(self) -> None:
        """Per-episode hook (e.g. clear recurrent state). No-op by default."""

    def __repr__(self) -> str:
        return f"{type(self).__name__}(role={self.role})"


class SignallerAgent(Agent):
    """Base class for signallers: observe items, emit a binary signal (Sec 4.2)."""

    role = "signaller"


class GuesserAgent(Agent):
    """Base class for guessers: observe signals, emit a guess (Sec 4.2)."""

    role = "guesser"
