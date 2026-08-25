"""Single-run trainer for shared-parameter IQL (Section 4.6.4, Table 4.1).

Ties together everything the other new modules define:

    environment/graphs.py       -- SignallingGraph (any shape, incl.
                                    from_adjacency_matrix())
    environment/*_game.py       -- GraphSignallingGame / GraphSignallingParallelEnv
    agents/networks.py          -- SharedQNetwork + GraphEncoding + ActionLayout
    agents/replay_buffer.py     -- ReplayBuffer
    algorithms/iql.py           -- IQLSignaller / IQLGuesser / train_step()

Nothing here hard-codes a graph shape or an item count: :func:`train_shared_iql`
derives the network's input/output sizes from ``env.graph`` and
``env.item_space`` (via ``GraphEncoding``/``ActionLayout``) and builds one
agent per node id, however many the graph has. Swapping the graph (or
``num_item_values``) is entirely the caller's job -- pick a different
``SignallingGraph`` factory, or a different number, when building ``env``;
this module never needs to change.

The loop itself is plain and short once those pieces exist: play an episode
(which, since ``run_episode()`` calls ``observe_reward()``, automatically
records a transition for every agent into the shared buffer), occasionally
sample a minibatch and take one gradient step, and decay epsilon from
"always explore" to "mostly exploit" over training.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import torch

from ..agents.networks import ActionLayout, GraphEncoding, SharedQNetwork
from ..agents.replay_buffer import ReplayBuffer
from ..algorithms.iql import IQLGuesser, IQLSignaller, train_step
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


def _all_agents(signaller_agents: Dict[int, IQLSignaller], guesser_agents: Dict[int, IQLGuesser]) -> List:
    return list(signaller_agents.values()) + list(guesser_agents.values())


def play_greedy_example(
    env,
    signaller_agents: Dict[int, IQLSignaller],
    guesser_agents: Dict[int, IQLGuesser],
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
        torch.manual_seed(seed)

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
