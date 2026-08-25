"""Independent Q-Learning (IQL), three parameter-sharing variants (Section 4.4.1).

All three variants below are still *Independent* Q-Learning in the sense
that matters for the research question: each agent still picks its own
action from only its own local observation, with no communication and no
coordination at decision-time -- a signaller never sees what a guesser will
do, and vice versa. What differs between them -- and from the literal "each
agent maintains its own Q-function" description in Section 4.4.1 -- is *how
much gets shared*:

* :class:`IQLSignaller` / :class:`IQLGuesser` -- **one brain for everyone**.
  Every agent's Q-values come out of the *same* ``SharedQNetwork``
  instance, whichever role it plays, distinguished by an identity flag in
  the input and a mask on the output (see ``agents/networks.py``).
* :class:`TwoBrainIQLSignaller` / :class:`TwoBrainIQLGuesser` -- **one brain
  per role**. Signallers share one ``SharedQNetwork``; guessers share a
  *separate* one. The more conventional MARL choice -- parameter sharing
  within a homogeneous group of agents (all signallers are doing the "same
  kind" of job as each other), not across two roles doing opposite jobs
  (encode vs. decode). Because each network here is exclusively one role's,
  its raw output is already restricted to that role's own actions -- there's
  nothing to mask, because there's nothing else in the output *to* mask.
* :class:`IndependentIQLSignaller` / :class:`IndependentIQLGuesser` -- **one
  brain per agent**. No sharing at all: every agent gets its own private
  network, its own replay buffer, its own optimizer. This is the most
  literal reading of Section 4.4.1's stated algorithm -- "each agent
  maintains its own Q-function" -- and needs neither an identity flag nor
  output masking nor observation padding, for the same reason: a private
  network was never going to be asked to represent any agent but this one,
  so there's nothing else to distinguish it from or restrict it to.

In the first two variants, "one brain" means one Python object referenced
from every agent sharing it, not a separately-created copy per agent. This
distinction matters: if you *copied* a network's weights into separate
modules instead, each copy would start identical but immediately diverge
the moment any one of them was updated by its own optimizer step -- they'd
stop being "shared" after the very first gradient update. A shared
*reference* is what keeps them permanently tied to the same weights,
updated by one optimizer, for the lifetime of training. The third variant
has no such reference to share in the first place -- see
:class:`_IndependentIQLAgent` below.

Every agent in the first two variants still needs its own *identity* (which
specific node it is, not just which role) so agents sharing a brain can
still be told apart even when they observe the same thing -- see
``agents/networks.py``'s module docstring for why that's essential once a
graph has more than one agent per role. The third variant needs no identity
at all, for the same "nothing else to distinguish it from" reason above.

One correctness note that applies to *both* the two-brain and fully-
independent variants: whenever two agents' input/output shapes can genuinely
differ (a guesser's action count depends on ``num_item_values``; different
agents' neighbourhood sizes can differ across a graph), they also need
*separate* replay buffers -- mixing differently-shaped transitions into one
buffer would break ``ReplayBuffer.sample()``'s ``torch.stack()`` the moment
the input vectors aren't all the same length. See ``training/trainer.py``'s
``train_two_brain_iql()`` and ``train_independent_iql()`` for where that
separation happens.

Training itself (the episode loop, epsilon schedule, when to call
:func:`train_step`) lives in ``training/trainer.py`` -- this module only
defines what each kind of IQL agent *is* and how one gradient step against
a replay buffer works (shared by all three variants -- see :func:`train_step`).
"""

from __future__ import annotations

import random
from typing import Optional

import torch
import torch.nn.functional as F

from ..agents.base_agent import GuesserAgent, Observation, SignallerAgent
from ..agents.networks import (
    ActionLayout,
    GraphEncoding,
    RoleEncoding,
    SharedQNetwork,
    encode_observation,
    masked_q_values,
)
from ..agents.replay_buffer import ReplayBuffer, Transition


class _SharedIQLAgent:
    """Shared logic between :class:`IQLSignaller` and :class:`IQLGuesser`.

    Not a full :class:`Agent` subclass itself (the two role classes below
    provide that, so ``isinstance`` checks and ``role`` still work as
    expected elsewhere in the codebase) -- just the epsilon-greedy /
    buffer-recording behaviour both roles share.
    """

    #: Overridden by the subclasses: True for a signaller, False for a guesser.
    is_signaller: bool

    def __init__(
        self,
        node_id: int,
        encoding: GraphEncoding,
        layout: ActionLayout,
        network: SharedQNetwork,
        buffer: ReplayBuffer,
        epsilon: float = 0.0,
    ) -> None:
        # This agent's own identity -- which specific node it is, used to
        # pick out the right slot of the identity one-hot every act() call
        # (see GraphEncoding.encode_input). This is what lets two agents of
        # the same role, sharing the same weights, still learn to behave
        # differently from one another.
        self.node_id = node_id
        self.encoding = encoding
        self.layout = layout
        self.network = network
        self.buffer = buffer
        self.epsilon = epsilon
        # If False, observe_reward() still runs but skips the buffer.add()
        # call -- lets a caller play a "just show me what it's learned so
        # far" episode mid-training (e.g. a periodic checkpoint demo)
        # without that one extra episode quietly becoming training data.
        self.recording = True
        # Set by act(), read by observe_reward() -- the input/action this
        # agent chose this episode, so the eventual team reward can be
        # attributed to the right (input, action) pair once it's known.
        self._last_input: Optional[torch.Tensor] = None
        self._last_action_index: Optional[int] = None

    def act(self, observation: Observation) -> int:
        x = self.encoding.encode_input(self.is_signaller, self.node_id, observation)

        # Epsilon-greedy: explore with probability epsilon, otherwise exploit
        # the current best action *for this role* -- masking is applied
        # before the argmax, so exploitation can never select an action that
        # belongs to the other role. A signaller always has exactly 2
        # possible actions (the channel is always binary); a guesser has
        # num_item_values of them -- so the exploration choice below has to
        # ask the layout how many options this role actually has, rather
        # than assuming 2 for both (that assumption is what would silently
        # break the moment num_item_values > 2, e.g. Section 4.6.6's
        # ternary variant).
        if random.random() < self.epsilon:
            num_choices = 2 if self.is_signaller else self.layout.num_item_values
            env_action = random.randrange(num_choices)
        else:
            with torch.no_grad():
                raw_q = self.network(x.unsqueeze(0)).squeeze(0)  # (num_actions,)
                q = masked_q_values(raw_q, self.is_signaller, self.layout)
                action_index = int(torch.argmax(q).item())
                env_action = self.layout.env_action_from_index(action_index)

        self._last_input = x
        self._last_action_index = self.layout.combined_action_index(self.is_signaller, env_action)
        return env_action

    def observe_reward(self, reward: float) -> None:
        if not self.recording:
            return
        # Guard: if act() was never called this episode (shouldn't happen in
        # normal play, but cheap to check), there's nothing to record.
        if self._last_input is None or self._last_action_index is None:
            return
        self.buffer.add(Transition(
            input_vector=self._last_input,
            action_index=self._last_action_index,
            reward=reward,
        ))

    def reset_episode(self) -> None:
        self._last_input = None
        self._last_action_index = None

    def masked_q_preview(self, observation: Observation) -> torch.Tensor:
        """The network's current masked Q-values for a given observation.

        Not used during play -- this is purely a diagnostic/teaching hook,
        letting you print "here's what the shared brain currently believes,
        from this specific agent's point of view".
        """
        x = self.encoding.encode_input(self.is_signaller, self.node_id, observation)
        with torch.no_grad():
            raw_q = self.network(x.unsqueeze(0)).squeeze(0)
        return masked_q_values(raw_q, self.is_signaller, self.layout)


class IQLSignaller(_SharedIQLAgent, SignallerAgent):
    """A signaller whose policy is the shared brain, masked to signal actions.

    Base-class order matters here: ``_SharedIQLAgent`` must come *before*
    ``SignallerAgent`` so Python's method resolution order finds
    ``_SharedIQLAgent.act`` (the real implementation) before it reaches
    ``Agent.act`` (abstract). Written the other way around, Python would
    resolve ``act`` to the still-abstract one and refuse to let you
    instantiate the class at all.
    """

    is_signaller = True

    def __init__(
        self,
        node_id: int,
        encoding: GraphEncoding,
        layout: ActionLayout,
        network: SharedQNetwork,
        buffer: ReplayBuffer,
        epsilon: float = 0.0,
    ) -> None:
        _SharedIQLAgent.__init__(self, node_id, encoding, layout, network, buffer, epsilon)


class IQLGuesser(_SharedIQLAgent, GuesserAgent):
    """A guesser whose policy is the shared brain, masked to guess actions.

    See :class:`IQLSignaller` for why ``_SharedIQLAgent`` must be listed
    first in the base classes.
    """

    is_signaller = False

    def __init__(
        self,
        node_id: int,
        encoding: GraphEncoding,
        layout: ActionLayout,
        network: SharedQNetwork,
        buffer: ReplayBuffer,
        epsilon: float = 0.0,
    ) -> None:
        _SharedIQLAgent.__init__(self, node_id, encoding, layout, network, buffer, epsilon)


class _RoleIQLAgent:
    """Shared logic between :class:`TwoBrainIQLSignaller` and
    :class:`TwoBrainIQLGuesser` -- the two-brain counterpart to
    :class:`_SharedIQLAgent` above.

    Structurally almost identical to ``_SharedIQLAgent`` (same epsilon-greedy
    / buffer-recording shape), with one simplification worth noticing: no
    masking anywhere. ``_SharedIQLAgent`` needs ``masked_q_values()`` because
    its one shared network's output has slots for *both* roles' actions, and
    only half of them are legal for any given agent. Here, each agent's
    network is exclusively its own role's brain -- its output width *is*
    exactly that role's action count, so there is no "other role's slots" to
    mask out, and a combined-action-space index/offset (like
    ``ActionLayout.combined_action_index``) is unnecessary too: the
    network's own output index already *is* the environment action.
    """

    def __init__(
        self,
        node_id: int,
        encoding: RoleEncoding,
        num_actions: int,
        network: SharedQNetwork,
        buffer: ReplayBuffer,
        epsilon: float = 0.0,
    ) -> None:
        self.node_id = node_id
        self.encoding = encoding
        self.num_actions = num_actions
        self.network = network
        self.buffer = buffer
        self.epsilon = epsilon
        # Same purpose as in _SharedIQLAgent -- see that class for why.
        self.recording = True
        self._last_input: Optional[torch.Tensor] = None
        self._last_action_index: Optional[int] = None

    def act(self, observation: Observation) -> int:
        x = self.encoding.encode_input(self.node_id, observation)

        if random.random() < self.epsilon:
            action = random.randrange(self.num_actions)
        else:
            with torch.no_grad():
                q = self.network(x.unsqueeze(0)).squeeze(0)  # already exactly this role's actions
                action = int(torch.argmax(q).item())

        self._last_input = x
        self._last_action_index = action  # no offset needed -- see class docstring
        return action

    def observe_reward(self, reward: float) -> None:
        if not self.recording:
            return
        if self._last_input is None or self._last_action_index is None:
            return
        self.buffer.add(Transition(
            input_vector=self._last_input,
            action_index=self._last_action_index,
            reward=reward,
        ))

    def reset_episode(self) -> None:
        self._last_input = None
        self._last_action_index = None

    def q_preview(self, observation: Observation) -> torch.Tensor:
        """The network's current Q-values for a given observation.

        Named ``q_preview``, not ``masked_q_preview`` like the shared-brain
        agent's version -- nothing needs masking here (see class docstring),
        so calling it "masked" would claim a step that doesn't happen.
        """
        x = self.encoding.encode_input(self.node_id, observation)
        with torch.no_grad():
            return self.network(x.unsqueeze(0)).squeeze(0)


class TwoBrainIQLSignaller(_RoleIQLAgent, SignallerAgent):
    """A signaller whose policy comes from the signaller-only brain.

    Base-class order matters here for the same reason as
    :class:`IQLSignaller`: ``_RoleIQLAgent`` must come *before*
    ``SignallerAgent`` so Python's method resolution order finds
    ``_RoleIQLAgent.act`` before ``Agent``'s still-abstract one.
    """

    def __init__(
        self,
        node_id: int,
        encoding: RoleEncoding,
        network: SharedQNetwork,
        buffer: ReplayBuffer,
        epsilon: float = 0.0,
    ) -> None:
        # A signaller's action space is always binary (the channel is
        # always one bit, regardless of num_item_values) -- so num_actions
        # is fixed at 2 here, unlike the guesser below.
        _RoleIQLAgent.__init__(self, node_id, encoding, 2, network, buffer, epsilon)


class TwoBrainIQLGuesser(_RoleIQLAgent, GuesserAgent):
    """A guesser whose policy comes from the guesser-only brain.

    See :class:`TwoBrainIQLSignaller` for why ``_RoleIQLAgent`` must be
    listed first in the base classes.
    """

    def __init__(
        self,
        node_id: int,
        encoding: RoleEncoding,
        network: SharedQNetwork,
        buffer: ReplayBuffer,
        num_item_values: int,
        epsilon: float = 0.0,
    ) -> None:
        # A guesser names one of the possible items, so its action count is
        # the item space's size -- unlike the signaller, this isn't fixed at
        # 2, which is exactly why it's passed in here rather than hard-coded
        # (the same num_item_values generality the one-brain variant gets
        # from ActionLayout).
        _RoleIQLAgent.__init__(self, node_id, encoding, num_item_values, network, buffer, epsilon)


class _IndependentIQLAgent:
    """Shared logic between :class:`IndependentIQLSignaller` and
    :class:`IndependentIQLGuesser` -- the fully-independent counterpart to
    :class:`_SharedIQLAgent` and :class:`_RoleIQLAgent` above.

    This is the simplest of the three agent bases, and it's simpler for a
    real reason, not just less code: every piece of bookkeeping the other
    two need exists to let several agents safely share one network without
    stepping on each other. ``_SharedIQLAgent`` needs an identity flag
    *and* masking (many agents, many roles, one network). ``_RoleIQLAgent``
    still needs an identity flag (many agents, one network per role). Here,
    ``self.network`` is never handed to any other agent -- it was
    constructed for this one node id and nothing else will ever call it --
    so there's no one else to be told apart from, and no other role's
    actions ever occupy its output to mask out.
    """

    def __init__(
        self,
        num_actions: int,
        network: SharedQNetwork,
        buffer: ReplayBuffer,
        epsilon: float = 0.0,
    ) -> None:
        self.num_actions = num_actions
        self.network = network
        self.buffer = buffer
        self.epsilon = epsilon
        # Same purpose as in _SharedIQLAgent -- see that class for why.
        self.recording = True
        self._last_input: Optional[torch.Tensor] = None
        self._last_action_index: Optional[int] = None

    def act(self, observation: Observation) -> int:
        x = encode_observation(observation)

        if random.random() < self.epsilon:
            action = random.randrange(self.num_actions)
        else:
            with torch.no_grad():
                q = self.network(x.unsqueeze(0)).squeeze(0)  # already exactly this agent's own actions
                action = int(torch.argmax(q).item())

        self._last_input = x
        self._last_action_index = action
        return action

    def observe_reward(self, reward: float) -> None:
        if not self.recording:
            return
        if self._last_input is None or self._last_action_index is None:
            return
        self.buffer.add(Transition(
            input_vector=self._last_input,
            action_index=self._last_action_index,
            reward=reward,
        ))

    def reset_episode(self) -> None:
        self._last_input = None
        self._last_action_index = None

    def q_preview(self, observation: Observation) -> torch.Tensor:
        """This agent's own network's current Q-values for a given observation."""
        x = encode_observation(observation)
        with torch.no_grad():
            return self.network(x.unsqueeze(0)).squeeze(0)


class IndependentIQLSignaller(_IndependentIQLAgent, SignallerAgent):
    """A signaller with its own private network -- no parameters shared
    with any other agent, of either role.

    Base-class order matters here for the same reason as
    :class:`IQLSignaller`: ``_IndependentIQLAgent`` must come *before*
    ``SignallerAgent`` so Python's method resolution order finds
    ``_IndependentIQLAgent.act`` before ``Agent``'s still-abstract one.
    """

    is_signaller = True

    def __init__(self, network: SharedQNetwork, buffer: ReplayBuffer, epsilon: float = 0.0) -> None:
        # Always 2 -- the channel is always binary, regardless of num_item_values.
        _IndependentIQLAgent.__init__(self, 2, network, buffer, epsilon)


class IndependentIQLGuesser(_IndependentIQLAgent, GuesserAgent):
    """A guesser with its own private network -- no parameters shared with
    any other agent, of either role.

    See :class:`IndependentIQLSignaller` for why ``_IndependentIQLAgent``
    must be listed first in the base classes.
    """

    is_signaller = False

    def __init__(
        self,
        network: SharedQNetwork,
        buffer: ReplayBuffer,
        num_item_values: int,
        epsilon: float = 0.0,
    ) -> None:
        # The item space's size -- see TwoBrainIQLGuesser for the same reasoning.
        _IndependentIQLAgent.__init__(self, num_item_values, network, buffer, epsilon)


def train_step(
    network: SharedQNetwork,
    buffer: ReplayBuffer,
    optimizer: torch.optim.Optimizer,
    batch_size: int,
) -> Optional[float]:
    """One gradient update from a random minibatch. Returns the loss, or
    ``None`` if the buffer doesn't yet hold a full batch.

    The target is just ``reward`` (see the "why gamma=0 simplifies this"
    note in replay_buffer.py) -- no target network, no bootstrapping. The
    loss only ever touches the Q-value at the action that was actually
    taken (via ``gather``), which is always a valid index for whichever
    agent produced that transition, so no masking is needed here -- masking
    is purely a concern for *choosing* an action (in ``act()`` above), not
    for scoring one that was already chosen. This function is also
    completely unaware of graph size or item count -- it just samples
    whatever's in the buffer and fits toward the recorded reward, so it
    needed no changes to support arbitrary graphs/item counts.
    """
    if len(buffer) < batch_size:
        return None

    inputs, action_indices, rewards = buffer.sample(batch_size)

    predicted_q_all = network(inputs)                                   # (batch, num_actions)
    predicted_q = predicted_q_all.gather(1, action_indices.unsqueeze(1)).squeeze(1)  # (batch,)

    loss = F.mse_loss(predicted_q, rewards)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return float(loss.item())
