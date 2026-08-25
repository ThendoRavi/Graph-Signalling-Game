"""Core environment.

Cooperative one-step Dec-POMDP defined by the tuple
``G = <N, x, {O_i}, {A_i}, R, G>``. One *episode* is a single-step
interaction:

    reset()            1. sample a hidden item x_j ~ Uniform(item space) for
                          every guesser; hand each signaller the items of the
                          guessers it can see  ->  signaller observations O_i.
    submit_signals()   2. each signaller emits one binary signal m_i in {0,1};
                          each guesser is handed the signals it hears  ->  O_j.
    submit_guesses()   3. each guesser emits a guess x̂_j of its own item;
                          the shared team reward is computed.

The reward is the *fraction of guessers that guess correctly*:

        R = (1 / |N_G|) * sum_j  1[ x̂_j == x_j ].


"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

from .graphs import SignallingGraph
from .spaces import DiscreteSpace

# A local observation is a fixed-order tuple of ints (one per neighbour).
Observation = Tuple[int, ...]


@dataclass
class EpisodeRecord:
    """A full, immutable trace of one played episode.

    Everything a renderer, a test, or an information-theoretic metric needs is
    captured here, keyed by node id.
    """

    items: Dict[int, int]                       # guesser_id -> hidden item x_j
    signaller_observations: Dict[int, Observation]  # signaller_id -> items it saw
    signals: Dict[int, int]                     # signaller_id -> emitted bit m_i
    guesser_observations: Dict[int, Observation]    # guesser_id -> signals it saw
    guesses: Dict[int, int]                     # guesser_id -> guess x̂_j
    correct: Dict[int, bool]                    # guesser_id -> was the guess right?
    reward: float                               # team reward R in [0, 1]
    extras: Dict[str, object] = field(default_factory=dict)


class GraphSignallingGame:
    """The Graph Signalling Game environment.

    Parameters
    ----------
    graph:
        The signalling structure. Any :class:`SignallingGraph`; the base case
        is ``SignallingGraph.single_pair()``.
    num_item_values:
        Size of the hidden-item / guess space. ``2`` = binary (default),
        ``3`` = ternary extended variant.
    seed:
        Seed for the environment's item-sampling random.
    """

    #: Phase constants make the state machine self-documenting.
    _PHASE_AWAIT_RESET = "await_reset"
    _PHASE_AWAIT_SIGNALS = "await_signals"
    _PHASE_AWAIT_GUESSES = "await_guesses"
    _PHASE_DONE = "done"

    def __init__(
        self,
        graph: SignallingGraph,
        num_item_values: int = 2,
        seed: Optional[int] = None,
    ) -> None:
        graph.validate()
        self.graph = graph
        self.item_space = DiscreteSpace(num_item_values, name="item")
        self.signal_space = DiscreteSpace(2, name="signal")   # always one bit
        # A guesser names its own item, so the guess space *is* the item space.
        self.guess_space = self.item_space

        self.rng = random.Random(seed)

        # Per-episode state (all reset in reset()).
        self._phase = self._PHASE_AWAIT_RESET
        self._items: Dict[int, int] = {}
        self._signals: Dict[int, int] = {}
        self._guesses: Dict[int, int] = {}
        self._last_record: Optional[EpisodeRecord] = None

    # -- phase 1: reset ------------------------------------------------------

    def reset(self, seed: Optional[int] = None) -> Dict[int, Observation]:
        """Start a new episode; return each signaller's observation.

        Samples a fresh hidden item for every guesser and returns
        ``{signaller_id: observation}``, where a signaller's observation is the
        tuple of items of the guessers it is connected to, ordered by guesser id
        (Eq. 4.2).
        """
        if seed is not None:
            self.rng.seed(seed)

        # 1. Sample hidden items x_j for every guesser.
        self._items = {
            g: self.item_space.sample(self.rng) for g in self.graph.guessers
        }
        self._signals = {}
        self._guesses = {}
        self._last_record = None
        self._phase = self._PHASE_AWAIT_SIGNALS

        return self._signaller_observations()

    # -- phase 2: signals ----------------------------------------------------

    def submit_signals(self, signals: Dict[int, int]) -> Dict[int, Observation]:
        """Record signaller bits; return each guesser's observation.

        ``signals`` must give exactly one binary value per signaller. Returns
        ``{guesser_id: observation}`` where a guesser's observation is the tuple
        of signals from the signallers it hears, ordered by signaller id
        (Eq. 4.3).
        """
        self._require_phase(self._PHASE_AWAIT_SIGNALS, "submit_signals")
        self._validate_actions(
            signals, self.graph.signallers, self.signal_space, role="signaller"
        )
        self._signals = dict(signals)
        self._phase = self._PHASE_AWAIT_GUESSES
        return self._guesser_observations()

    # -- phase 3: guesses + reward ------------------------------------------

    def submit_guesses(self, guesses: Dict[int, int]) -> Tuple[float, EpisodeRecord]:
        """Record guesses, compute the team reward, and close the episode.

        Returns ``(reward, record)`` where ``reward`` is the fraction of correct
        guesses (Eq. 4.4) and ``record`` is the full :class:`EpisodeRecord`.
        """
        self._require_phase(self._PHASE_AWAIT_GUESSES, "submit_guesses")
        self._validate_actions(
            guesses, self.graph.guessers, self.guess_space, role="guesser"
        )
        self._guesses = dict(guesses)

        # Team reward = fraction of guessers who named their own item (Eq. 4.4).
        correct = {g: (self._guesses[g] == self._items[g]) for g in self.graph.guessers}
        reward = sum(correct.values()) / len(self.graph.guessers)

        record = EpisodeRecord(
            items=dict(self._items),
            signaller_observations=self._signaller_observations(),
            signals=dict(self._signals),
            guesser_observations=self._guesser_observations(),
            guesses=dict(self._guesses),
            correct=correct,
            reward=reward,
        )
        self._last_record = record
        self._phase = self._PHASE_DONE
        return reward, record

    # -- convenience: play a whole episode with agents -----------------------

    def run_episode(
        self,
        signaller_agents: Dict[int, "object"],
        guesser_agents: Dict[int, "object"],
        seed: Optional[int] = None,
    ) -> EpisodeRecord:
        """Play one full episode driven by agent objects.

        ``signaller_agents`` / ``guesser_agents`` map node id -> an object with
        an ``act(observation)`` method (see :mod:`gsg.agents.base_agent`). This
        wires the three phases together so callers that don't care about the
        step-by-step interface can just hand over policies.

        Also calls each agent's ``reset_episode()`` (at the start) and
        ``observe_reward()`` (once the team reward is known, at the end) --
        the two learning hooks every :class:`~gsg.agents.base_agent.Agent`
        exposes. These are no-ops for non-learning agents (random, oracle,
        deterministic), so calling them here is safe for every existing
        caller; it's what lets a *learning* agent (e.g. IQL) actually update
        from playing through this method, without ``run_episode`` needing to
        know or care which kind of agent it was handed.
        """
        all_agents = list(signaller_agents.values()) + list(guesser_agents.values())
        for agent in all_agents:
            agent.reset_episode()

        signaller_obs = self.reset(seed=seed)
        signals = {s: signaller_agents[s].act(signaller_obs[s]) for s in self.graph.signallers}

        guesser_obs = self.submit_signals(signals)
        guesses = {g: guesser_agents[g].act(guesser_obs[g]) for g in self.graph.guessers}

        reward, record = self.submit_guesses(guesses)
        for agent in all_agents:
            agent.observe_reward(reward)
        return record

    @property
    def last_record(self) -> Optional[EpisodeRecord]:
        """The most recently completed episode's record (or ``None``)."""
        return self._last_record

    # -- internals -----------------------------------------------------------

    def _signaller_observations(self) -> Dict[int, Observation]:
        """Build O_i for every signaller from the current hidden items (Eq. 4.2)."""
        return {
            s: tuple(self._items[g] for g in self.graph.out_neighbours(s))
            for s in self.graph.signallers
        }

    def _guesser_observations(self) -> Dict[int, Observation]:
        """Build O_j for every guesser from the current signals (Eq. 4.3)."""
        return {
            g: tuple(self._signals[s] for s in self.graph.in_neighbours(g))
            for g in self.graph.guessers
        }

    def _require_phase(self, expected: str, method: str) -> None:
        if self._phase != expected:
            raise RuntimeError(
                f"{method}() called during phase '{self._phase}', "
                f"but it is only valid during '{expected}'. "
                "Call reset() -> submit_signals() -> submit_guesses() in order."
            )

    @staticmethod
    def _validate_actions(
        actions: Dict[int, int],
        expected_ids,
        space: DiscreteSpace,
        role: str,
    ) -> None:
        """Enforce that exactly the right agents acted with in-range values."""
        expected = set(expected_ids)
        got = set(actions)
        if got != expected:
            missing = expected - got
            extra = got - expected
            raise ValueError(
                f"{role} actions must cover exactly {sorted(expected)}; "
                f"missing={sorted(missing)}, unexpected={sorted(extra)}"
            )
        for node_id, value in actions.items():
            if not space.contains(value):
                raise ValueError(
                    f"{role} {node_id} produced {value!r}, "
                    f"outside {space} (expected 0..{space.n - 1})"
                )

    def __repr__(self) -> str:
        return (
            f"GraphSignallingGame({self.graph!r}, "
            f"items={self.item_space.n}, phase={self._phase})"
        )
