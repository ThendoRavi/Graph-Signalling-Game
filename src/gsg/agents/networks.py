"""Shared Q-network architecture for parameter-shared *and* fully-independent
IQL (Section 4.4.2).

Three points on the parameter-sharing spectrum live in this file, all built
from the same ``SharedQNetwork`` (a plain feedforward net -- see below) and
-- this is deliberate, see "controlled input" below -- the same
:class:`GraphEncoding` input scheme:

* **One brain for everyone** (``IQLSignaller``/``IQLGuesser`` in
  ``algorithms/iql.py``): a single network plays *every* agent, of both
  roles, restricted per-call to that role's own actions via
  :func:`masked_q_values` and :class:`ActionLayout`. This is the least
  conventional choice -- see the section below for why it works at all.
* **One brain per role** (``TwoBrainIQLSignaller``/``TwoBrainIQLGuesser``):
  signallers share one network, guessers share a *separate* network. The
  more standard MARL choice -- parameter sharing *within* a homogeneous
  group of agents, not across fundamentally different roles.
* **One brain per agent** (``IndependentIQLSignaller``/
  ``IndependentIQLGuesser``): no sharing at all -- every single agent gets
  its own private network, its own replay buffer, its own optimizer. The
  most literal reading of Section 4.4.1's "each agent maintains its own
  Q-function Q_i(o_i, a_i; phi_i)".

Controlled input: all three variants see the same thing
-------------------------------------------------------------
Early versions of the last two variants used smaller, role-scoped or
identity-free inputs, on the reasoning that a role-exclusive or
agent-exclusive network doesn't strictly *need* an identity flag to
disambiguate itself (that's implicit in which network it is). That was a
mistake for the purpose of *comparing* the three variants: it meant a
convergence-rate difference between them could have been caused by "how
much information this agent's input contains," not just by "how much this
agent's network is shared with others" -- two different things tangled into
one number. All three variants now build their input the same way, via
:class:`GraphEncoding` -- the *same* per-node identity one-hot spanning
every agent in the graph, padded to the *same* graph-wide largest
neighbourhood size -- regardless of whether the network reading that input
is shared by everyone, shared within a role, or private to one agent. For a
role- or agent-exclusive network this makes part of the input constant
(e.g. an independent signaller's identity slice never changes, since that
network is only ever called for that one node), which costs nothing but a
few always-the-same input dimensions -- the point is to hold input
information fixed and vary *only* the thing actually being studied: how
many agents share one network.

What still legitimately differs between the three is *output width*, and
that is not a confound to control away: a role- or agent-exclusive
network's output only needs slots for the actions that specific network
will ever be asked about (2 for a signaller, ``num_item_values`` for a
guesser) -- giving it the shared-brain's full masked, combined output width
would just add permanently-unused, never-trained output neurons for no
reason. Only the fully-shared design needs :class:`ActionLayout`'s combined
layout and :func:`masked_q_values`'s masking at all, because it's the only
one where a single network's output must represent more than one role's
actions at once.

**None of the three designs hard-code a particular graph or item count** --
that's what makes "just change the graph" or "just change num_item_values"
a one-line change at the call site for any of them.

One brain for everyone: the input and output shapes
---------------------------------------------------------
What makes each agent's behaviour different, when one network plays every
role, is (a) a per-agent *identity* flag included in the input, and (b) a
*mask* applied to the output, restricting it to that role's own actions.
:class:`GraphEncoding` (input side) and :class:`ActionLayout` (output side)
are computed *once* from whatever graph and item-count you hand them, so the
network's shape, the identity encoding, and the action masking all follow
automatically from the graph.

Why one shared network can represent every agent at all
-----------------------------------------------------------
A signaller observes some items and outputs one binary signal; a guesser
observes some signals and outputs one guess. Every agent's *action* is a
single discrete choice (which is why one output layer, split into masked
regions, can represent all of them) even though different agents may
*observe* different numbers of things (out-degree/in-degree can vary across
a graph). :class:`GraphEncoding` handles that second part by padding every
observation to the graph's largest neighbourhood size.

The input: identity one-hot + padded observation
------------------------------------------------------
    [one-hot over every signaller+guesser id]  +  [padded observed values]

The identity one-hot is *per node*, not just per role. This matters as soon
as a graph has more than one agent of the same role who can see the same
thing: in a fully-connected graph like K_{2,2}, both signallers observe the
exact same neighbourhood every episode, so if the network were only told
"you're a signaller" (not "you're specifically signaller 0"), it would have
no way to ever tell the two of them apart -- they'd always output the same
action, capping team performance well below what two *complementary*
signallers can achieve. Full per-node identity is what lets the shared
network still learn different behaviour for different agents of the same
role.

The padded-observation slice uses ``-1`` for "no value here" -- the same
sentinel used elsewhere in this codebase (the PettingZoo adapter's
placeholder observations) and for the same reason: ``0`` is a real,
meaningful observed value, so it can't double as "nothing observed here."

The output: Q-values over the union of every role's actions, masked
------------------------------------------------------------------------
The signal channel is *always* binary (Section 2's channel-capacity
discussion: one bit, regardless of how large the item space is), so the
first two output slots are always "emit signal 0" / "emit signal 1". The
remaining ``num_item_values`` slots are "guess 0", "guess 1", ... -- as many
as the item space has, so changing item count from binary to ternary (or
any size) only changes how many *guess* slots exist, never the signaller's.
:class:`ActionLayout` computes this from ``num_item_values`` alone.

:func:`masked_q_values` overwrites whichever region doesn't belong to the
current role with ``-1`` before anything looks at the output. This is safe
by construction, not just in practice: episodes are single-step (gamma = 0,
Table 4.1), so any real action's true Q-value is an expected reward in
[0, 1] (Eq. 4.4), and the network's final ``Sigmoid`` layer only ever
outputs values in (0, 1) -- so ``-1`` is guaranteed strictly below every
genuine prediction, for any number of actions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import torch
import torch.nn as nn

from ..environment.graphs import SignallingGraph

#: A masked-out Q-value -- see the "why -1" note in the module docstring.
MASK_VALUE = -1.0
#: A "no value observed here" padding slot -- same convention, same reason.
PAD_VALUE = -1.0


@dataclass(frozen=True)
class ActionLayout:
    """Describes the shared network's *output*: the combined action space.

    Layout is ``[signal=0, signal=1, guess=0, guess=1, ..., guess=(n-1)]``:
    2 signaller slots (the channel is always binary) followed by
    ``num_item_values`` guesser slots (a guesser names one of the possible
    items, so its action space *is* the item space -- see
    ``GraphSignallingGame.guess_space`` in the environment).
    """

    num_item_values: int

    @property
    def num_actions(self) -> int:
        return 2 + self.num_item_values

    @property
    def signaller_action_indices(self) -> Tuple[int, ...]:
        return (0, 1)

    @property
    def guesser_action_indices(self) -> Tuple[int, ...]:
        return tuple(range(2, 2 + self.num_item_values))

    def combined_action_index(self, is_signaller: bool, env_action: int) -> int:
        """Map a real environment action to its slot in the combined output."""
        return env_action if is_signaller else 2 + env_action

    def env_action_from_index(self, action_index: int) -> int:
        """Map a combined-output slot back to a real environment action."""
        return action_index if action_index < 2 else action_index - 2


@dataclass(frozen=True)
class GraphEncoding:
    """Describes the *input* every IQL variant in this codebase uses:
    per-node identity + padded observation. Used identically by all three
    parameter-sharing variants (see the module docstring's "controlled
    input" section) -- what differs between them is which network(s) get
    called with this input and how many other agents share a reference to
    that network, never the shape or content of the input itself.

    Built once from a graph via :meth:`from_graph`; everything else in this
    module (and in ``algorithms/iql.py``) is driven by the resulting sizes,
    never by a hard-coded agent count or neighbourhood size.
    """

    signaller_ids: Tuple[int, ...]
    guesser_ids: Tuple[int, ...]
    #: Largest neighbourhood size in the graph (max out-degree over
    #: signallers, or in-degree over guessers, whichever is larger) -- every
    #: agent's observation is padded up to this length.
    max_neighbourhood: int

    @classmethod
    def from_graph(cls, graph: SignallingGraph) -> "GraphEncoding":
        signaller_ids = tuple(graph.signallers)
        guesser_ids = tuple(graph.guessers)
        max_out = max((graph.out_degree(s) for s in signaller_ids), default=0)
        max_in = max((graph.in_degree(g) for g in guesser_ids), default=0)
        return cls(signaller_ids, guesser_ids, max(max_out, max_in, 1))

    @property
    def num_agents(self) -> int:
        """Total identity slots: every signaller plus every guesser."""
        return len(self.signaller_ids) + len(self.guesser_ids)

    @property
    def input_dim(self) -> int:
        """Identity one-hot length + padded-observation length."""
        return self.num_agents + self.max_neighbourhood

    def identity_index(self, is_signaller: bool, node_id: int) -> int:
        """Which slot of the identity one-hot belongs to this specific agent."""
        if is_signaller:
            return self.signaller_ids.index(node_id)
        return len(self.signaller_ids) + self.guesser_ids.index(node_id)

    def encode_input(
        self, is_signaller: bool, node_id: int, observed_values: Sequence[int]
    ) -> torch.Tensor:
        """Build the ``[identity one-hot] + [padded observed values]`` input.

        ``observed_values`` is this agent's actual observation this episode
        (the items it sees, if a signaller; the signals it hears, if a
        guesser) -- padded on the right with ``PAD_VALUE`` up to
        ``max_neighbourhood`` so every agent in the graph produces the same
        fixed-length input, however many neighbours it individually has.
        """
        identity = [0.0] * self.num_agents
        identity[self.identity_index(is_signaller, node_id)] = 1.0

        if len(observed_values) > self.max_neighbourhood:
            raise ValueError(
                f"observation of length {len(observed_values)} exceeds this "
                f"encoding's max_neighbourhood={self.max_neighbourhood}"
            )
        padding = [PAD_VALUE] * (self.max_neighbourhood - len(observed_values))
        observed = [float(v) for v in observed_values] + padding

        return torch.tensor(identity + observed, dtype=torch.float32)


class SharedQNetwork(nn.Module):
    """The one shared "brain" -- a small feedforward net, Q-values in (0, 1).

    Architecture follows Section 4.4.2 (two hidden layers of 64 units each,
    ReLU) with one addition beyond the proposal's literal spec: a final
    ``Sigmoid``, which bounds every output to (0, 1). That bound is not
    cosmetic -- it's what makes the ``-1`` masking scheme in
    :func:`masked_q_values` provably safe (see the module docstring) rather
    than "safe unless the network's weights happen to do something unusual."

    ``input_dim``/``num_actions`` come from a :class:`GraphEncoding` and
    :class:`ActionLayout` built for whatever graph/item-count you're using --
    see ``training/trainer.py`` for where those get constructed.
    """

    def __init__(self, input_dim: int, num_actions: int, hidden_size: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_actions),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x`` has shape ``(batch, input_dim)``; returns ``(batch, num_actions)``."""
        return self.net(x)


def masked_q_values(raw_q: torch.Tensor, is_signaller: bool, layout: ActionLayout) -> torch.Tensor:
    """Return ``raw_q`` with the *other* role's entries set to ``-1``.

    ``raw_q`` is the network's raw output for one observation. The returned
    tensor is safe to both ``argmax`` over (for action selection) and
    print/inspect (for "what does the shared brain think, from this role's
    point of view") -- in both cases, masked slots read as an unambiguous,
    always-invalid ``-1``.
    """
    masked = raw_q.clone()
    invalid_indices = layout.guesser_action_indices if is_signaller else layout.signaller_action_indices
    for index in invalid_indices:
        masked[..., index] = MASK_VALUE
    return masked
