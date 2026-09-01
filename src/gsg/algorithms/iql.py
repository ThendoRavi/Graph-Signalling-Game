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
  instance, whichever role it plays, distinguished at the output by a mask
  (see :func:`masked_q_values`).
* :class:`TwoBrainIQLSignaller` / :class:`TwoBrainIQLGuesser` -- **one brain
  per role**. Signallers share one ``SharedQNetwork``; guessers share a
  *separate* one. The more conventional MARL choice -- parameter sharing
  within a homogeneous group of agents (all signallers are doing the "same
  kind" of job as each other), not across two roles doing opposite jobs
  (encode vs. decode).
* :class:`IndependentIQLSignaller` / :class:`IndependentIQLGuesser` -- **one
  brain per agent**. No sharing at all: every agent gets its own private
  network, its own replay buffer, its own optimizer. The most literal
  reading of Section 4.4.1's stated algorithm -- "each agent maintains its
  own Q-function."

All three use the *same* input scheme (:class:`~gsg.agents.networks.GraphEncoding`
-- a per-node identity one-hot spanning every agent in the graph, plus
observation padded to the graph's largest neighbourhood), and the same
underlying network architecture (``SharedQNetwork``). This is deliberate,
not incidental: the whole reason to compare these three variants is to
isolate the effect of *how much a network is shared*, and that comparison
would be confounded the moment one variant's agents also had access to more
or less input information than another's. See
``agents/networks.py``'s "controlled input" note for the fuller version of
this argument. The two role-/agent-exclusive variants (:class:`TwoBrainIQLSignaller`
et al.) don't strictly *need* the identity flag for themselves (a network
that's already private to one role or one agent has no one else to be
confused with) -- they carry it anyway, so that a convergence-rate
difference between variants can only be attributed to the sharing scheme,
never to a difference in what each agent was told.

Only the fully-shared variant needs output masking -- see
:func:`masked_q_values`'s docstring and :class:`_SharedIQLAgent`. The other
two variants' networks are already restricted to their own role's actions
by construction (nothing else was ever going to call them), so
:class:`_UnmaskedIQLAgent` -- the shared base for both -- needs none.

In the first two variants, "one brain" means one Python object referenced
from every agent sharing it, not a separately-created copy per agent. This
distinction matters: if you *copied* a network's weights into separate
modules instead, each copy would start identical but immediately diverge
the moment any one of them was updated by its own optimizer step -- they'd
stop being "shared" after the very first gradient update. A shared
*reference* is what keeps them permanently tied to the same weights,
updated by one optimizer, for the lifetime of training. The third variant
has no such reference to share in the first place.

One correctness note that applies to *both* the two-brain and fully-
independent variants: whenever two agents' output shapes can genuinely
differ (a guesser's action count depends on ``num_item_values``), they also
need *separate* replay buffers -- mixing differently-shaped transitions
into one buffer would break ``ReplayBuffer.sample()``'s ``torch.stack()``
the moment the input vectors aren't all the same length. See
``training/trainer.py``'s ``train_two_brain_iql()`` and
``train_independent_iql()`` for where that separation happens.

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
from ..agents.networks import ActionLayout, GraphEncoding, SharedQNetwork, masked_q_values
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


class _UnmaskedIQLAgent:
    """Shared logic between the two-brain variant (:class:`TwoBrainIQLSignaller`,
    :class:`TwoBrainIQLGuesser`) and the fully-independent variant
    (:class:`IndependentIQLSignaller`, :class:`IndependentIQLGuesser`).

    These two variants turn out to need *identical* per-agent decision logic
    -- same input encoding, same "no masking needed" reasoning, same
    epsilon-greedy/buffer-recording shape -- once both use the same
    :class:`~gsg.agents.networks.GraphEncoding` input as the shared-brain
    variant (see that module's "controlled input" note for why). The only
    thing that actually distinguishes "one brain per role" from "one brain
    per agent" is which agents share a reference to the same ``network``/
    ``buffer`` objects -- a question this class has no opinion on, because
    it's answered entirely by *how many times* ``training/trainer.py``
    constructs a network and hands the *same* one to multiple agents
    (``train_two_brain_iql()``) versus a fresh one per agent
    (``train_independent_iql()``). One shared base here is the honest
    reflection of that: there is nothing left for two separate classes to
    disagree about at the level of a single agent's own decision-making.

    No masking anywhere, unlike ``_SharedIQLAgent``: that class needs
    ``masked_q_values()`` because its one shared network's output has slots
    for *both* roles' actions, and only half are legal for any given agent.
    Here, whichever network this agent was handed -- role-exclusive or
    agent-exclusive -- was never going to be asked about any action outside
    this agent's own role, so its output width already *is* exactly this
    agent's own action count, and a combined-action-space offset (like
    ``ActionLayout.combined_action_index``) is unnecessary too: the
    network's own output index already *is* the environment action.
    """

    #: Overridden by the subclasses: True for a signaller, False for a guesser.
    is_signaller: bool

    def __init__(
        self,
        node_id: int,
        encoding: GraphEncoding,
        num_actions: int,
        network: SharedQNetwork,
        buffer: ReplayBuffer,
        epsilon: float = 0.0,
    ) -> None:
        # Still carried even for the fully-independent variant, where this
        # agent's own network only ever sees one identity -- see the module
        # docstring's "controlled input" note for why that's deliberate.
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
        x = self.encoding.encode_input(self.is_signaller, self.node_id, observation)

        if random.random() < self.epsilon:
            action = random.randrange(self.num_actions)
        else:
            with torch.no_grad():
                q = self.network(x.unsqueeze(0)).squeeze(0)  # already exactly this agent's own actions
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
        """This agent's network's current Q-values for a given observation.

        Named ``q_preview``, not ``masked_q_preview`` like the shared-brain
        agent's version -- nothing needs masking here (see class docstring),
        so calling it "masked" would claim a step that doesn't happen.
        """
        x = self.encoding.encode_input(self.is_signaller, self.node_id, observation)
        with torch.no_grad():
            return self.network(x.unsqueeze(0)).squeeze(0)


class TwoBrainIQLSignaller(_UnmaskedIQLAgent, SignallerAgent):
    """A signaller whose policy comes from the signaller-only brain.

    Base-class order matters here for the same reason as
    :class:`IQLSignaller`: ``_UnmaskedIQLAgent`` must come *before*
    ``SignallerAgent`` so Python's method resolution order finds
    ``_UnmaskedIQLAgent.act`` before ``Agent``'s still-abstract one.
    """

    is_signaller = True

    def __init__(
        self,
        node_id: int,
        encoding: GraphEncoding,
        network: SharedQNetwork,
        buffer: ReplayBuffer,
        epsilon: float = 0.0,
    ) -> None:
        # A signaller's action space is always binary (the channel is
        # always one bit, regardless of num_item_values) -- so num_actions
        # is fixed at 2 here, unlike the guesser below.
        _UnmaskedIQLAgent.__init__(self, node_id, encoding, 2, network, buffer, epsilon)


class TwoBrainIQLGuesser(_UnmaskedIQLAgent, GuesserAgent):
    """A guesser whose policy comes from the guesser-only brain.

    See :class:`TwoBrainIQLSignaller` for why ``_UnmaskedIQLAgent`` must be
    listed first in the base classes.
    """

    is_signaller = False

    def __init__(
        self,
        node_id: int,
        encoding: GraphEncoding,
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
        _UnmaskedIQLAgent.__init__(self, node_id, encoding, num_item_values, network, buffer, epsilon)


class IndependentIQLSignaller(_UnmaskedIQLAgent, SignallerAgent):
    """A signaller with its own private network -- no parameters shared
    with any other agent, of either role.

    Structurally identical to :class:`TwoBrainIQLSignaller` (same base
    class, same constructor) -- see :class:`_UnmaskedIQLAgent`'s docstring
    for why: the only thing distinguishing this variant from the two-brain
    one is whether ``training/trainer.py`` hands this same ``network``
    object to any other agent (``train_two_brain_iql()`` does; here,
    ``train_independent_iql()`` never does), not anything about how this
    class itself decides what to do.
    """

    is_signaller = True

    def __init__(
        self,
        node_id: int,
        encoding: GraphEncoding,
        network: SharedQNetwork,
        buffer: ReplayBuffer,
        epsilon: float = 0.0,
    ) -> None:
        _UnmaskedIQLAgent.__init__(self, node_id, encoding, 2, network, buffer, epsilon)


class IndependentIQLGuesser(_UnmaskedIQLAgent, GuesserAgent):
    """A guesser with its own private network -- no parameters shared with
    any other agent, of either role.

    See :class:`IndependentIQLSignaller` for why this is structurally
    identical to :class:`TwoBrainIQLGuesser`.
    """

    is_signaller = False

    def __init__(
        self,
        node_id: int,
        encoding: GraphEncoding,
        network: SharedQNetwork,
        buffer: ReplayBuffer,
        num_item_values: int,
        epsilon: float = 0.0,
    ) -> None:
        _UnmaskedIQLAgent.__init__(self, node_id, encoding, num_item_values, network, buffer, epsilon)


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
