"""Oracle baseline (Section 4.5.4).

The oracle is not a learned agent: it is a hand-coded *optimal* encoding that
defines the theoretical performance ceiling and confirms the environment can, in
fact, reach it. If the oracle does not score ``R = 1.00``, something in the
environment (reward, wiring, observation construction) is broken -- so the oracle
doubles as an implementation sanity check.

For the K_{2,2} game (Section 4.5.4) the oracle has S1 track G1 and S2 track G2,
so the two-bit joint signal perfectly encodes the neighbourhood. For the base
case here -- one signaller, one guesser -- the oracle degenerates to the obvious
strategy: the signaller broadcasts the guesser's item verbatim, and the guesser
copies the bit. That is the ``single_pair_oracle`` helper below.
"""

from __future__ import annotations

from typing import Dict, Tuple

from ..agents.base_agent import GuesserAgent, SignallerAgent
from ..agents.guesser import CopySignalGuesser
from ..agents.signaller import TrackingSignaller
from ..environment.graphs import SignallingGraph


def single_pair_oracle() -> Tuple[Dict[int, SignallerAgent], Dict[int, GuesserAgent]]:
    """Optimal agents for the one-signaller/one-guesser graph.

    Returns ``(signaller_agents, guesser_agents)`` ready to hand to
    :meth:`GraphSignallingGame.run_episode`. Signaller 0 transmits guesser 0's
    item directly (``TrackingSignaller``); guesser 0 copies that bit
    (``CopySignalGuesser``). Expected reward: 1.00 on every episode.
    """
    signallers: Dict[int, SignallerAgent] = {0: TrackingSignaller(index=0)}
    guessers: Dict[int, GuesserAgent] = {0: CopySignalGuesser(index=0)}
    return signallers, guessers


def k22_oracle() -> Tuple[Dict[int, SignallerAgent], Dict[int, GuesserAgent]]:
    """Optimal agents for the K_{2,2} game (Section 4.5.4).

    Unlike ``single_pair_oracle``, every signaller here sees *both* guessers'
    items (K_{2,2} is fully connected), so "tracking" a guesser means picking
    out the right element of that shared 2-item observation, and every guesser
    hears *both* signals, so "decoding" means picking out the right element of
    that shared 2-signal observation. Signaller 0 tracks guesser 0 (emits the
    item at index 0 of its observation); signaller 1 tracks guesser 1 (emits
    the item at index 1). Guesser 0 copies the signal at index 0 of what it
    hears (signaller 0's bit); guesser 1 copies index 1 (signaller 1's bit).

    Together the two-bit joint signal (m0, m1) perfectly encodes both hidden
    items, so every guesser is right on every episode: R = 1.00.
    """
    signallers: Dict[int, SignallerAgent] = {
        0: TrackingSignaller(index=0),  # sees (x_G0, x_G1); tracks G0
        1: TrackingSignaller(index=1),  # sees (x_G0, x_G1); tracks G1
    }
    guessers: Dict[int, GuesserAgent] = {
        0: CopySignalGuesser(index=0),  # hears (m_S0, m_S1); reads S0's bit
        1: CopySignalGuesser(index=1),  # hears (m_S0, m_S1); reads S1's bit
    }
    return signallers, guessers


def matching_oracle(
    graph: SignallingGraph,
) -> Tuple[Dict[int, SignallerAgent], Dict[int, GuesserAgent]]:
    """Optimal agents for any graph where each guesser has exactly one signaller.

    Generalises :func:`single_pair_oracle` to a perfect matching (the E1
    degenerate control, Section 4.7): every signaller with a single guesser
    neighbour transmits that guesser's item, and every guesser with a single
    signaller neighbour copies its bit. Raises if the graph is not a matching,
    since the "track one item" encoding is only optimal there.
    """
    signallers: Dict[int, SignallerAgent] = {}
    for s in graph.signallers:
        if graph.out_degree(s) != 1:
            raise ValueError(
                "matching_oracle requires every signaller to have out-degree 1; "
                f"signaller {s} has {graph.out_degree(s)}"
            )
        signallers[s] = TrackingSignaller(index=0)

    guessers: Dict[int, GuesserAgent] = {}
    for g in graph.guessers:
        if graph.in_degree(g) != 1:
            raise ValueError(
                "matching_oracle requires every guesser to have in-degree 1; "
                f"guesser {g} has {graph.in_degree(g)}"
            )
        guessers[g] = CopySignalGuesser(index=0)

    return signallers, guessers
