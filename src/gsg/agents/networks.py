"""Shared Q-network architecture for parameter-shared *and* fully-independent
IQL (Section 4.4.2).

Three points on the parameter-sharing spectrum live in this file, all built
from the same ``SharedQNetwork`` (a plain feedforward net -- see below),
just wired up with different amounts of sharing:

* **One brain for everyone** (:class:`GraphEncoding` + :class:`ActionLayout`,
  used by ``IQLSignaller``/``IQLGuesser`` in ``algorithms/iql.py``): a single
  network plays *every* agent, of both roles, restricted per-call to that
  role's own actions via masking. This is the least conventional choice --
  see the section below for why it works at all.
* **One brain per role** (:class:`RoleEncoding`, used by
  ``TwoBrainIQLSignaller``/``TwoBrainIQLGuesser``): signallers share one
  network, guessers share a *separate* network. This is the more standard
  MARL choice -- parameter sharing *within* a homogeneous group of agents,
  not across fundamentally different roles -- and needs no output masking
  at all, since a role-exclusive network's output already only has slots
  for that role's own actions.
* **One brain per agent** (:func:`encode_observation`, used by
  ``IndependentIQLSignaller``/``IndependentIQLGuesser``): no sharing at
  all -- every single agent gets its own private network, its own replay
  buffer, its own optimizer. This is the most literal reading of Section
  4.4.1's "each agent maintains its own Q-function Q_i(o_i, a_i; phi_i)".

Moving along that spectrum toward less sharing removes bookkeeping, not
adds it: the fully-shared design needs an identity one-hot (to tell agents
apart), observation padding (so one input layer fits every agent's
neighbourhood size), *and* output masking (so one output layer only lets
each agent choose its own actions). One-brain-per-role still needs identity
and padding, but not masking (the output width already is that role's
actions). One-brain-per-agent needs *none* of the three -- each network's
input is already sized to exactly this one agent's own observation, and its
output already is exactly this one agent's own actions, because nothing
else was ever going to be asked of it. What you gain going the other way
(more sharing) is faster learning through shared experience across agents;
what you gain going this way (less sharing) is that every agent's policy is
free to specialise without any cross-agent interference at all -- the
tradeoff the multi-seed K_{2,2} comparison between the first two variants
was built to start examining.

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
    """Describes the shared network's *input*: per-node identity + padding.

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


@dataclass(frozen=True)
class RoleEncoding:
    """Like :class:`GraphEncoding`, but scoped to *one role's own agents* --
    the input spec for the two-brain variant (``TwoBrainIQLSignaller`` /
    ``TwoBrainIQLGuesser`` in ``algorithms/iql.py``), where signallers and
    guessers each get their own separate network.

    The difference from ``GraphEncoding`` is exactly the difference between
    the two designs: ``GraphEncoding`` builds one identity space spanning
    *both* roles, because one shared brain needs to tell a signaller from a
    guesser as well as tell same-role agents apart. A role-exclusive brain
    never needs to distinguish role -- that's implicit in which network you
    ask -- so it only needs identity among its *own* role's agents, and its
    observation padding only needs to cover its *own* role's neighbourhood
    sizes (a signaller's out-degree, or a guesser's in-degree -- these can
    differ across a graph, so each brain is sized to just what it needs).
    """

    node_ids: Tuple[int, ...]
    max_neighbourhood: int

    @classmethod
    def for_signallers(cls, graph: SignallingGraph) -> "RoleEncoding":
        ids = tuple(graph.signallers)
        max_out = max((graph.out_degree(s) for s in ids), default=0)
        return cls(ids, max(max_out, 1))

    @classmethod
    def for_guessers(cls, graph: SignallingGraph) -> "RoleEncoding":
        ids = tuple(graph.guessers)
        max_in = max((graph.in_degree(g) for g in ids), default=0)
        return cls(ids, max(max_in, 1))

    @property
    def num_agents(self) -> int:
        return len(self.node_ids)

    @property
    def input_dim(self) -> int:
        return self.num_agents + self.max_neighbourhood

    def encode_input(self, node_id: int, observed_values: Sequence[int]) -> torch.Tensor:
        """Build ``[identity one-hot over this role's agents] + [padded observation]``."""
        identity = [0.0] * self.num_agents
        identity[self.node_ids.index(node_id)] = 1.0

        if len(observed_values) > self.max_neighbourhood:
            raise ValueError(
                f"observation of length {len(observed_values)} exceeds this "
                f"encoding's max_neighbourhood={self.max_neighbourhood}"
            )
        padding = [PAD_VALUE] * (self.max_neighbourhood - len(observed_values))
        observed = [float(v) for v in observed_values] + padding

        return torch.tensor(identity + observed, dtype=torch.float32)


def encode_observation(observed_values: Sequence[int]) -> torch.Tensor:
    """Build the input for a fully-independent (one-brain-per-agent) network:
    just the raw observed values, nothing else.

    No identity flag, because a private network never needs to be told
    apart from any other agent's -- there's nothing else feeding it. No
    padding, because a private network's input layer is sized to exactly
    this one agent's own neighbourhood (see how ``input_dim`` is computed
    per-agent in ``training/trainer.py``'s ``train_independent_iql()``),
    so there's no larger shape to pad up to. This is the payoff of having
    no sharing at all: the input is exactly what the agent actually sees,
    nothing more.
    """
    return torch.tensor([float(v) for v in observed_values], dtype=torch.float32)


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
