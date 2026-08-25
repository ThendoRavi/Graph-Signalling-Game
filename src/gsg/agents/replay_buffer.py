"""Experience replay buffer (Section 4.4.1, Table 4.1).

Stores single-step transitions and hands back random minibatches for
training -- the two standard ingredients of "experience replay": training on
a *shuffled* sample of past experience, rather than only the most recent
transition, is what keeps a small batch of updates from over-fitting to
whatever happened to occur in the last episode or two.

Simplification specific to gamma = 0 (worth understanding, not hiding)
--------------------------------------------------------------------------
A standard DQN transition is ``(state, action, reward, next_state, done)``,
because the training target needs ``next_state`` to bootstrap:

    target = reward + gamma * max_a' Q(next_state, a')

Table 4.1 fixes ``gamma = 0`` (the game is single-step: nothing happens
*after* a guess, so there's nothing to bootstrap from). With gamma = 0, that
whole second term is multiplied by zero regardless of what ``next_state`` or
the target network say, so the target simplifies to exactly:

    target = reward

which needs no ``next_state`` at all. So a transition here is just
``(input_vector, action_index, reward)`` -- storing ``next_state``/``done``
fields that the loss would never use would be storing dead weight, not
faithfulness to a "real" DQN buffer.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Tuple

import torch


@dataclass
class Transition:
    """One recorded decision: what was seen, what was chosen, what it earned."""

    input_vector: torch.Tensor  # shape (INPUT_DIM,) -- see agents/networks.py
    action_index: int           # 0-3, the combined-action index that was taken
    reward: float                # the team reward received for that episode


class ReplayBuffer:
    """A fixed-capacity ring buffer of :class:`Transition` objects.

    Parameters
    ----------
    capacity:
        Maximum number of transitions kept. Table 4.1 specifies 5e4; that
        number was sized for the full K_{2,2}/topology experiments, and is
        far larger than the single-pair base case could ever need (there are
        only 4 distinct (role, observed_bit) combinations in this game!) --
        it's kept as the default here for consistency with the rest of the
        project's hyperparameters, not because this tiny game requires it.
    """

    def __init__(self, capacity: int = 50_000) -> None:
        self.capacity = capacity
        self._data: List[Transition] = []
        self._next_index = 0

    def add(self, transition: Transition) -> None:
        """Insert a transition, overwriting the oldest one once full."""
        if len(self._data) < self.capacity:
            self._data.append(transition)
        else:
            # Ring-buffer overwrite: oldest transitions are replaced first,
            # so the buffer always holds the most recent `capacity` experiences.
            self._data[self._next_index] = transition
        self._next_index = (self._next_index + 1) % self.capacity

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Draw ``batch_size`` transitions uniformly at random, stacked into tensors.

        Returns ``(inputs, action_indices, rewards)`` with shapes
        ``(batch_size, INPUT_DIM)``, ``(batch_size,)``, ``(batch_size,)``.
        """
        batch = random.sample(self._data, batch_size)
        inputs = torch.stack([t.input_vector for t in batch])
        action_indices = torch.tensor([t.action_index for t in batch], dtype=torch.int64)
        rewards = torch.tensor([t.reward for t in batch], dtype=torch.float32)
        return inputs, action_indices, rewards

    def __len__(self) -> int:
        return len(self._data)
