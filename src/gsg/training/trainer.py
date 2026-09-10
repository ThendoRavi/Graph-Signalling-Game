"""Single-run trainers for all three IQL parameter-sharing variants (Section
4.6.4, Table 4.1).

Ties together everything the other new modules define:

    environment/graphs.py       -- SignallingGraph (any shape, incl.
                                    from_adjacency_matrix())
    environment/*_game.py       -- GraphSignallingGame / GraphSignallingParallelEnv
    agents/networks.py          -- SharedQNetwork, GraphEncoding (the input
                                    scheme all three variants share), and
                                    ActionLayout (the shared-brain variant's
                                    combined, masked output layout)
    agents/replay_buffer.py     -- ReplayBuffer
    algorithms/iql.py           -- IQLSignaller/IQLGuesser (one brain),
                                    TwoBrainIQLSignaller/TwoBrainIQLGuesser
                                    (two brains), IndependentIQLSignaller/
                                    IndependentIQLGuesser (one brain per
                                    agent), train_step()

Three entry points, mirroring the three variants in algorithms/iql.py:
:func:`train_shared_iql` (one network for every agent), :func:`train_two_brain_iql`
(a separate network per role), and :func:`train_independent_iql` (a separate
network per *agent* -- the most literal reading of Section 4.4.1). All three
build their network(s)' *input* the same way, via ``GraphEncoding`` derived
from ``env.graph`` -- deliberately, so comparing the three isolates the
effect of how much a network is shared, not also how much input information
each variant's agents were given (see ``agents/networks.py``'s "controlled
input" note). Only network *output* width, and how many separate network
objects exist, differ between the three -- both of those are intrinsic to
what each variant actually is, not confounds to control away. Nothing in
any of the three hard-codes a graph shape or an item count: swapping the
graph (or ``num_item_values``) is entirely the caller's job -- pick a
different ``SignallingGraph`` factory, or a different number, when building
``env``; none of the three trainers needs to change.

The loop itself is plain and short once those pieces exist: play an episode
(which, since ``run_episode()`` calls ``observe_reward()``, automatically
records a transition for every agent into the relevant buffer), occasionally
sample a minibatch and take one gradient step per network that exists (one,
two, or as many as there are agents, depending on the variant), and decay
epsilon from "always explore" to "mostly exploit" over training.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import torch

from ..agents.networks import ActionLayout, GraphEncoding, SharedQNetwork
from ..agents.replay_buffer import ReplayBuffer
from ..algorithms.iql import (
    IndependentIQLGuesser,
    IndependentIQLSignaller,
    IQLGuesser,
    IQLSignaller,
    TwoBrainIQLGuesser,
    TwoBrainIQLSignaller,
    train_step,
)
from ..environment.graph_signalling_game import EpisodeRecord


@dataclass
class TrainingHistory:
    """Per-episode record of training progress, for plotting/inspection."""

    episode_rewards: List[float] = field(default_factory=list)
    losses: List[float] = field(default_factory=list)  # only recorded on episodes with a gradient step


def linear_epsilon(episode: int, decay_episodes: int, start: float = 1.0, end: float = 0.05) -> float:
    """Linearly interpolate epsilon from ``start`` to ``end`` over
    ``decay_episodes``, then hold at ``end``.

    Table 4.1 specifies epsilon decaying from 1.0 to 0.05 over 2.5e5 *steps*
    of the full-scale K_{2,2} experiments. Smaller graphs converge faster,
    so ``decay_episodes`` is left as a caller-supplied parameter rather than
    hard-coding Table 4.1's number -- the *shape* of the schedule (linear,
    1.0 -> 0.05) is kept faithful to the proposal; only the schedule's
    *length* is something you're expected to tune per problem size.
    """
    if episode >= decay_episodes:
        return end
    fraction = episode / decay_episodes
    return start + fraction * (end - start)


def _all_agents(signaller_agents: Dict[int, object], guesser_agents: Dict[int, object]) -> List:
    return list(signaller_agents.values()) + list(guesser_agents.values())


def play_greedy_example(
    env,
    signaller_agents: Dict[int, object],
    guesser_agents: Dict[int, object],
) -> EpisodeRecord:
    """Play one episode fully greedy (epsilon=0) across *every* agent,
    without it counting as training data, then restore every agent's
    previous epsilon/recording state exactly as it was.

    This is what a "checkpoint: here's an example of what it's learned so
    far" call should use *instead of* just calling ``env.run_episode()``
    directly mid-training -- calling it directly would (a) act with
    whatever epsilon training currently happens to be at, so the example
    could just be random exploration rather than every agent's actual best
    guess, and (b) push that example into the replay buffer as if it were
    ordinary training data, which it isn't meant to be.

    Works unchanged for all three IQL variants (:func:`train_shared_iql`,
    :func:`train_two_brain_iql`, or :func:`train_independent_iql`) -- it
    only ever touches ``.epsilon``/``.recording``, which every agent class
    in ``algorithms/iql.py`` exposes identically, so nothing here needs to
    know which variant's agents it was handed.
    """
    agents = _all_agents(signaller_agents, guesser_agents)
    saved = [(a.epsilon, a.recording) for a in agents]
    for a in agents:
        a.epsilon = 0.0
        a.recording = False
    try:
        return env.run_episode(signaller_agents, guesser_agents)
    finally:
        for a, (epsilon, recording) in zip(agents, saved):
            a.epsilon = epsilon
            a.recording = recording


#: Called every ``log_every`` episodes during training, if provided, with
#: (episodes_so_far, rewards_since_last_call, network, signaller_agents,
#: guesser_agents) -- e.g. to print a progress line and/or play a
#: play_greedy_example() episode (pass ``env`` in via a closure) to show
#: what's been learned so far.
LogCallback = Callable[
    [int, List[float], SharedQNetwork, Dict[int, IQLSignaller], Dict[int, IQLGuesser]], None
]

#: Called *every* episode, if provided, with (episode_index, EpisodeRecord)
#: -- the record from the episode that was just played, with exploration
#: still on and epsilon at whatever the schedule currently says. Lets a
#: caller instrument the actual *learning* behaviour (e.g. tally an
#: action-response matrix from real training play), rather than only the
#: settled greedy policy you'd get by evaluating afterward. Accepted by all
#: three trainers.
EpisodeCallback = Callable[[int, EpisodeRecord], None]


def train_shared_iql(
    env,
    num_episodes: int = 1500,
    epsilon_decay_episodes: int = 800,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    buffer_capacity: int = 50_000,
    train_every: int = 1,
    log_every: Optional[int] = None,
    on_log: Optional[LogCallback] = None,
    on_episode: Optional[EpisodeCallback] = None,
    seed: Optional[int] = None,
) -> tuple[SharedQNetwork, Dict[int, IQLSignaller], Dict[int, IQLGuesser], TrainingHistory]:
    """Train one shared-parameter IQL population on ``env``.

    ``env`` can be either a ``GraphSignallingGame`` or a
    ``GraphSignallingParallelEnv`` -- both expose the same ``.graph``,
    ``.item_space``, and ``run_episode(signaller_agents, guesser_agents) ->
    EpisodeRecord`` interface (see pettingzoo_env.py's "parity" section), so
    this function doesn't need to know or care which one it was given, or
    how many signallers/guessers the graph has, or how many values an item
    can take -- all of that is read from ``env`` itself.

    If ``log_every`` is set, ``on_log`` is called every ``log_every``
    episodes *during* the loop below (not collected and reported only at
    the very end) -- pass a function that prints a progress line, or plays
    a :func:`play_greedy_example` episode, or both.

    Returns ``(network, signaller_agents, guesser_agents, history)`` --
    dicts of ``{node_id: agent}``, one entry per signaller/guesser the graph
    actually has. All agents are already wired to the trained network -- hand
    the dicts straight to ``gsg.evaluation.metrics.evaluate()`` afterwards
    (with every agent's ``.epsilon`` set to 0 for a purely greedy evaluation).
    """
    if seed is not None:
        # Seed *both* sources of randomness the training loop draws on:
        # torch (network initialisation) and Python's `random` module
        # (epsilon-greedy exploration in agents/iql.py, and minibatch
        # sampling in replay_buffer.py). Seeding only torch -- as this code
        # used to -- left exploration and replay sampling uncontrolled, so
        # "the same seed" produced materially different training runs,
        # especially for runs that don't cleanly converge. The environment's
        # item sampling has its own seeded `random.Random` instance and is
        # handled separately, at env construction.
        torch.manual_seed(seed)
        random.seed(seed)

    # These two specs are the entire reason changing the graph or the item
    # count is a one-line change: they're derived from env.graph/env.item_space
    # right here, and everything below (network shape, per-agent identity,
    # exploration range, action masking) follows from them automatically.
    encoding = GraphEncoding.from_graph(env.graph)
    layout = ActionLayout(num_item_values=env.item_space.n)

    network = SharedQNetwork(encoding.input_dim, layout.num_actions)
    buffer = ReplayBuffer(capacity=buffer_capacity)
    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)

    # One agent instance *per node id* -- not one instance reused across
    # several ids. Reusing a single instance would mean two different
    # signallers overwrite each other's _last_input/_last_action_index
    # (set inside act()) before observe_reward() ever runs for either of
    # them, silently corrupting whichever one acted first. Every agent below
    # still shares the same `network` and `buffer` objects ("one brain");
    # only the per-episode bookkeeping is kept separate per agent.
    signaller_agents: Dict[int, IQLSignaller] = {
        s: IQLSignaller(s, encoding, layout, network, buffer, epsilon=1.0)
        for s in env.graph.signallers
    }
    guesser_agents: Dict[int, IQLGuesser] = {
        g: IQLGuesser(g, encoding, layout, network, buffer, epsilon=1.0)
        for g in env.graph.guessers
    }
    all_agents = _all_agents(signaller_agents, guesser_agents)

    history = TrainingHistory()
    rewards_since_last_log: List[float] = []

    for episode in range(num_episodes):
        epsilon = linear_epsilon(episode, epsilon_decay_episodes)
        for agent in all_agents:
            agent.epsilon = epsilon

        # Playing the episode is enough to record a transition for every
        # agent: run_episode() calls observe_reward() on all of them once
        # the team reward is known (see graph_signalling_game.py /
        # pettingzoo_env.py), which is what pushes (input, action, reward)
        # into the shared replay buffer.
        record = env.run_episode(signaller_agents, guesser_agents)
        history.episode_rewards.append(record.reward)
        rewards_since_last_log.append(record.reward)
        if on_episode is not None:
            on_episode(episode, record)

        if episode % train_every == 0:
            loss = train_step(network, buffer, optimizer, batch_size)
            if loss is not None:
                history.losses.append(loss)

        # Report progress *now*, mid-loop, rather than handing back the
        # whole history for the caller to print after training finishes --
        # that's the difference between a live progress readout and a wall
        # of numbers dumped at the end.
        episodes_so_far = episode + 1
        if log_every and on_log is not None and episodes_so_far % log_every == 0:
            on_log(episodes_so_far, rewards_since_last_log, network, signaller_agents, guesser_agents)
            rewards_since_last_log = []

    return network, signaller_agents, guesser_agents, history


#: Like LogCallback, but for train_two_brain_iql() -- called with *two*
#: networks (signaller_network, guesser_network) instead of one, since
#: there are two brains to report on.
TwoBrainLogCallback = Callable[
    [int, List[float], SharedQNetwork, SharedQNetwork, Dict[int, TwoBrainIQLSignaller], Dict[int, TwoBrainIQLGuesser]],
    None,
]


def train_two_brain_iql(
    env,
    num_episodes: int = 1500,
    epsilon_decay_episodes: int = 800,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    buffer_capacity: int = 50_000,
    train_every: int = 1,
    log_every: Optional[int] = None,
    on_log: Optional[TwoBrainLogCallback] = None,
    on_episode: Optional[EpisodeCallback] = None,
    seed: Optional[int] = None,
) -> tuple[SharedQNetwork, SharedQNetwork, Dict[int, TwoBrainIQLSignaller], Dict[int, TwoBrainIQLGuesser], TrainingHistory]:
    """Train a *two-brain* IQL population on ``env``: one network shared
    across all signallers, a separate network shared across all guessers.

    Contrast with :func:`train_shared_iql`, which uses one network for
    every agent regardless of role. The loop shape here is otherwise the
    same -- same epsilon schedule, same "play an episode, occasionally take
    a gradient step" structure -- the only real difference is that *two*
    independent networks/buffers/optimizers exist instead of one, and each
    episode now takes two calls to :func:`train_step` (one per brain)
    instead of one.

    Returns ``(signaller_network, guesser_network, signaller_agents,
    guesser_agents, history)``. As with :func:`train_shared_iql`, hand the
    agent dicts straight to ``gsg.evaluation.metrics.evaluate()`` afterwards
    (with every agent's ``.epsilon`` set to 0 for a purely greedy evaluation).
    """
    if seed is not None:
        # Seed *both* sources of randomness the training loop draws on:
        # torch (network initialisation) and Python's `random` module
        # (epsilon-greedy exploration in agents/iql.py, and minibatch
        # sampling in replay_buffer.py). Seeding only torch -- as this code
        # used to -- left exploration and replay sampling uncontrolled, so
        # "the same seed" produced materially different training runs,
        # especially for runs that don't cleanly converge. The environment's
        # item sampling has its own seeded `random.Random` instance and is
        # handled separately, at env construction.
        torch.manual_seed(seed)
        random.seed(seed)

    # Same GraphEncoding as train_shared_iql() -- both brains here see the
    # same identity-one-hot-plus-padded-observation input the shared-brain
    # variant does, deliberately, so the comparison between variants isolates
    # how much the *network* is shared rather than also comparing how much
    # *input information* each variant's agents were given. See
    # agents/networks.py's "controlled input" note.
    encoding = GraphEncoding.from_graph(env.graph)
    num_item_values = env.item_space.n

    # A signaller's action count is always 2 (the channel is always binary);
    # a guesser's is the item space's size -- see TwoBrainIQLGuesser. Output
    # width is the one thing that legitimately still differs from the
    # shared-brain network -- see the module docstring in agents/networks.py.
    signaller_network = SharedQNetwork(encoding.input_dim, 2)
    guesser_network = SharedQNetwork(encoding.input_dim, num_item_values)

    # Separate buffers, not one shared buffer -- see this module's/iql.py's
    # docstrings for why mixing the two roles' transitions into one buffer
    # would break ReplayBuffer.sample()'s torch.stack() the moment the two
    # roles' input vectors aren't the same length.
    signaller_buffer = ReplayBuffer(capacity=buffer_capacity)
    guesser_buffer = ReplayBuffer(capacity=buffer_capacity)

    signaller_optimizer = torch.optim.Adam(signaller_network.parameters(), lr=learning_rate)
    guesser_optimizer = torch.optim.Adam(guesser_network.parameters(), lr=learning_rate)

    # One agent instance per node id, same reasoning as train_shared_iql().
    signaller_agents: Dict[int, TwoBrainIQLSignaller] = {
        s: TwoBrainIQLSignaller(s, encoding, signaller_network, signaller_buffer, epsilon=1.0)
        for s in env.graph.signallers
    }
    guesser_agents: Dict[int, TwoBrainIQLGuesser] = {
        g: TwoBrainIQLGuesser(
            g, encoding, guesser_network, guesser_buffer, num_item_values, epsilon=1.0
        )
        for g in env.graph.guessers
    }
    all_agents = _all_agents(signaller_agents, guesser_agents)

    history = TrainingHistory()
    rewards_since_last_log: List[float] = []

    for episode in range(num_episodes):
        epsilon = linear_epsilon(episode, epsilon_decay_episodes)
        for agent in all_agents:
            agent.epsilon = epsilon

        record = env.run_episode(signaller_agents, guesser_agents)
        history.episode_rewards.append(record.reward)
        rewards_since_last_log.append(record.reward)
        if on_episode is not None:
            on_episode(episode, record)

        if episode % train_every == 0:
            signaller_loss = train_step(signaller_network, signaller_buffer, signaller_optimizer, batch_size)
            guesser_loss = train_step(guesser_network, guesser_buffer, guesser_optimizer, batch_size)
            # Combined into one number purely for a single progress metric --
            # the two losses come from two different networks/objectives, so
            # this sum isn't meaningful beyond "roughly how much is either
            # brain still adjusting."
            if signaller_loss is not None and guesser_loss is not None:
                history.losses.append(signaller_loss + guesser_loss)

        episodes_so_far = episode + 1
        if log_every and on_log is not None and episodes_so_far % log_every == 0:
            on_log(
                episodes_so_far, rewards_since_last_log,
                signaller_network, guesser_network,
                signaller_agents, guesser_agents,
            )
            rewards_since_last_log = []

    return signaller_network, guesser_network, signaller_agents, guesser_agents, history


#: Like LogCallback/TwoBrainLogCallback, but for train_independent_iql() --
#: called with *dicts* of networks ({node_id: SharedQNetwork}) for both
#: roles, since there's one brain per agent, not one or two brains total.
IndependentLogCallback = Callable[
    [
        int, List[float],
        Dict[int, SharedQNetwork], Dict[int, SharedQNetwork],
        Dict[int, IndependentIQLSignaller], Dict[int, IndependentIQLGuesser],
    ],
    None,
]


def train_independent_iql(
    env,
    num_episodes: int = 1500,
    epsilon_decay_episodes: int = 800,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    buffer_capacity: int = 50_000,
    train_every: int = 1,
    log_every: Optional[int] = None,
    on_log: Optional[IndependentLogCallback] = None,
    on_episode: Optional[EpisodeCallback] = None,
    seed: Optional[int] = None,
) -> tuple[
    Dict[int, SharedQNetwork], Dict[int, SharedQNetwork],
    Dict[int, IndependentIQLSignaller], Dict[int, IndependentIQLGuesser],
    TrainingHistory,
]:
    """Train a *fully independent* IQL population on ``env``: every single
    agent gets its own private network, its own replay buffer, and its own
    optimizer -- the most literal reading of Section 4.4.1's stated
    algorithm ("each agent maintains its own Q-function").

    Contrast with :func:`train_shared_iql` (one network, every agent) and
    :func:`train_two_brain_iql` (one network per role): here there are as
    many networks as there are agents in the graph, none of them sharing a
    single weight with any other. The loop shape is otherwise the same;
    the only structural difference is that a training step now means one
    :func:`train_step` call *per agent*, however many that is, instead of
    one or two.

    Returns ``(signaller_networks, guesser_networks, signaller_agents,
    guesser_agents, history)`` -- the first two are ``{node_id:
    SharedQNetwork}`` dicts, one entry per agent (not a single shared
    network like the other two trainers return). Hand the agent dicts
    straight to ``gsg.evaluation.metrics.evaluate()`` afterwards (with every
    agent's ``.epsilon`` set to 0 for a purely greedy evaluation).
    """
    if seed is not None:
        # Seed *both* sources of randomness the training loop draws on:
        # torch (network initialisation) and Python's `random` module
        # (epsilon-greedy exploration in agents/iql.py, and minibatch
        # sampling in replay_buffer.py). Seeding only torch -- as this code
        # used to -- left exploration and replay sampling uncontrolled, so
        # "the same seed" produced materially different training runs,
        # especially for runs that don't cleanly converge. The environment's
        # item sampling has its own seeded `random.Random` instance and is
        # handled separately, at env construction.
        torch.manual_seed(seed)
        random.seed(seed)

    num_item_values = env.item_space.n
    graph = env.graph

    # Same GraphEncoding as the other two trainers -- every one of these
    # private networks still sees the identity-one-hot-plus-padded-
    # observation input the shared-brain variant uses, even though a
    # private network's own identity slice never varies across calls (it's
    # only ever asked about one node). That constancy costs nothing but a
    # few always-the-same input dimensions, and it's what keeps this
    # variant's comparison against the other two isolated to "how much is
    # the network shared" rather than "how much input information did each
    # variant's agents get" -- see agents/networks.py's "controlled input"
    # note.
    encoding = GraphEncoding.from_graph(graph)

    signaller_networks: Dict[int, SharedQNetwork] = {}
    signaller_buffers: Dict[int, ReplayBuffer] = {}
    signaller_optimizers: Dict[int, torch.optim.Optimizer] = {}
    signaller_agents: Dict[int, IndependentIQLSignaller] = {}
    for s in graph.signallers:
        network = SharedQNetwork(encoding.input_dim, 2)
        buffer = ReplayBuffer(capacity=buffer_capacity)
        signaller_networks[s] = network
        signaller_buffers[s] = buffer
        signaller_optimizers[s] = torch.optim.Adam(network.parameters(), lr=learning_rate)
        signaller_agents[s] = IndependentIQLSignaller(s, encoding, network, buffer, epsilon=1.0)

    guesser_networks: Dict[int, SharedQNetwork] = {}
    guesser_buffers: Dict[int, ReplayBuffer] = {}
    guesser_optimizers: Dict[int, torch.optim.Optimizer] = {}
    guesser_agents: Dict[int, IndependentIQLGuesser] = {}
    for g in graph.guessers:
        network = SharedQNetwork(encoding.input_dim, num_item_values)
        buffer = ReplayBuffer(capacity=buffer_capacity)
        guesser_networks[g] = network
        guesser_buffers[g] = buffer
        guesser_optimizers[g] = torch.optim.Adam(network.parameters(), lr=learning_rate)
        guesser_agents[g] = IndependentIQLGuesser(g, encoding, network, buffer, num_item_values, epsilon=1.0)

    all_agents = _all_agents(signaller_agents, guesser_agents)

    history = TrainingHistory()
    rewards_since_last_log: List[float] = []

    for episode in range(num_episodes):
        epsilon = linear_epsilon(episode, epsilon_decay_episodes)
        for agent in all_agents:
            agent.epsilon = epsilon

        record = env.run_episode(signaller_agents, guesser_agents)
        history.episode_rewards.append(record.reward)
        rewards_since_last_log.append(record.reward)
        if on_episode is not None:
            on_episode(episode, record)

        if episode % train_every == 0:
            # One train_step() call per agent -- each agent's network,
            # buffer, and optimizer are entirely its own.
            step_losses = []
            for s in graph.signallers:
                loss = train_step(signaller_networks[s], signaller_buffers[s], signaller_optimizers[s], batch_size)
                if loss is not None:
                    step_losses.append(loss)
            for g in graph.guessers:
                loss = train_step(guesser_networks[g], guesser_buffers[g], guesser_optimizers[g], batch_size)
                if loss is not None:
                    step_losses.append(loss)
            # Summed across every agent's own gradient step -- same "rough
            # single progress number, not a meaningful combined loss" caveat
            # as train_two_brain_iql().
            if step_losses:
                history.losses.append(sum(step_losses))

        episodes_so_far = episode + 1
        if log_every and on_log is not None and episodes_so_far % log_every == 0:
            on_log(
                episodes_so_far, rewards_since_last_log,
                signaller_networks, guesser_networks,
                signaller_agents, guesser_agents,
            )
            rewards_since_last_log = []

    return signaller_networks, guesser_networks, signaller_agents, guesser_agents, history
