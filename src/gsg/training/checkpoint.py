"""Save and load trained populations, so experiments can reuse models instead
of retraining.

A population is the set ``{signaller_agents, guesser_agents}`` returned by any
of the three trainers in :mod:`gsg.training.trainer`. This module persists just
what is needed to *reconstruct the agents for evaluation* (forward passes with
``epsilon = 0``): each agent's network weights plus the variant tag. The
replay buffers and optimisers are deliberately not saved -- they matter only
for further training, not for playing or cross-play.

Networks that are *shared* at training time (the single brain of the
shared-brain variant; the per-role brains of two-brain) are simply saved once
per agent and rebuilt as independent copies on load. For evaluation that is
indistinguishable from the original (the forward pass is identical); only
continued training would care that the copies are no longer tied together.

Typical use::

    # after training
    save_population(path, "independent", signaller_agents, guesser_agents)

    # later, instead of retraining
    sig, gue = load_population(path, graph, num_item_values)   # epsilon = 0
    result = evaluate(env, sig, gue, episodes=...)
"""

from __future__ import annotations

import os
from typing import Dict, Tuple

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
)
from ..environment.graphs import SignallingGraph

VALID_VARIANTS = ("shared", "two_brain", "independent")


def save_population(
    path: str,
    variant: str,
    signaller_agents: Dict[int, object],
    guesser_agents: Dict[int, object],
) -> None:
    """Write a population's network weights (plus its ``variant`` tag) to ``path``.

    ``variant`` must be one of ``"shared"``, ``"two_brain"``, ``"independent"``
    -- it determines how the agents are rebuilt by :func:`load_population`.
    """
    if variant not in VALID_VARIANTS:
        raise ValueError(f"variant must be one of {VALID_VARIANTS}, got {variant!r}")
    payload = {
        "variant": variant,
        "signallers": {s: agent.network.state_dict() for s, agent in signaller_agents.items()},
        "guessers": {g: agent.network.state_dict() for g, agent in guesser_agents.items()},
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(payload, path)


def load_population(
    path: str, graph: SignallingGraph, num_item_values: int = 2
) -> Tuple[Dict[int, object], Dict[int, object]]:
    """Reconstruct a saved population as greedy (``epsilon = 0``) eval agents.

    ``graph`` and ``num_item_values`` must match the ones the population was
    trained on (they fix the network input/output dimensions). Returns
    ``(signaller_agents, guesser_agents)`` ready to hand to
    :func:`gsg.evaluation.metrics.evaluate` or any ``run_episode`` caller.
    """
    payload = torch.load(path, map_location="cpu", weights_only=True)
    variant = payload["variant"]
    encoding = GraphEncoding.from_graph(graph)
    layout = ActionLayout(num_item_values=num_item_values)

    def _dummy_buffer() -> ReplayBuffer:
        return ReplayBuffer(capacity=1)

    signaller_agents: Dict[int, object] = {}
    guesser_agents: Dict[int, object] = {}

    if variant == "shared":
        for s in graph.signallers:
            net = SharedQNetwork(encoding.input_dim, layout.num_actions)
            net.load_state_dict(payload["signallers"][s])
            net.eval()
            signaller_agents[s] = IQLSignaller(s, encoding, layout, net, _dummy_buffer(), epsilon=0.0)
        for g in graph.guessers:
            net = SharedQNetwork(encoding.input_dim, layout.num_actions)
            net.load_state_dict(payload["guessers"][g])
            net.eval()
            guesser_agents[g] = IQLGuesser(g, encoding, layout, net, _dummy_buffer(), epsilon=0.0)

    elif variant == "two_brain":
        for s in graph.signallers:
            net = SharedQNetwork(encoding.input_dim, 2)
            net.load_state_dict(payload["signallers"][s])
            net.eval()
            signaller_agents[s] = TwoBrainIQLSignaller(s, encoding, net, _dummy_buffer(), epsilon=0.0)
        for g in graph.guessers:
            net = SharedQNetwork(encoding.input_dim, num_item_values)
            net.load_state_dict(payload["guessers"][g])
            net.eval()
            guesser_agents[g] = TwoBrainIQLGuesser(g, encoding, net, _dummy_buffer(), num_item_values, epsilon=0.0)

    elif variant == "independent":
        for s in graph.signallers:
            net = SharedQNetwork(encoding.input_dim, 2)
            net.load_state_dict(payload["signallers"][s])
            net.eval()
            signaller_agents[s] = IndependentIQLSignaller(s, encoding, net, _dummy_buffer(), epsilon=0.0)
        for g in graph.guessers:
            net = SharedQNetwork(encoding.input_dim, num_item_values)
            net.load_state_dict(payload["guessers"][g])
            net.eval()
            guesser_agents[g] = IndependentIQLGuesser(g, encoding, net, _dummy_buffer(), num_item_values, epsilon=0.0)

    else:
        raise ValueError(f"unknown variant {variant!r} in checkpoint {path}")

    return signaller_agents, guesser_agents
