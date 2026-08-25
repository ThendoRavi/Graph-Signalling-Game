"""PettingZoo ParallelEnv adapter for the Graph Signalling Game.

This module is a thin *wrapper* around :class:`GraphSignallingGame` — the game
engine itself is untouched, and this file is the only place in the codebase
that imports ``gymnasium`` / ``pettingzoo``. That keeps the core (environment,
agents, baselines, evaluation) dependency-free, and makes this adapter purely
optional: you only need ``pip install gymnasium pettingzoo`` if you actually
want to drive the game through this API (e.g. to hand it to an RL library that
expects PettingZoo's ``ParallelEnv`` contract).

Why "Parallel" and not the plain single-agent Gym API
-------------------------------------------------------
Gym's classic ``reset()``/``step(action)`` assumes one observation, one action,
one reward per call. Our game has *multiple* simultaneous decision-makers per
role (every signaller acts at once, then every guesser acts at once), so there
is no single "the observation" or "the action" to hand to a plain ``gym.Env``.
PettingZoo's ``ParallelEnv`` is the standard extension for exactly this case:
every call is keyed by a dict of ``{agent_name: value}`` instead of one value.

The two-phase mapping (and a real constraint this ran into)
-------------------------------------------------------------
A Graph Signalling Game episode already has two phases (Section 4.2.2):
signallers act, *then* guessers act once they can see the signals. The first
version of this adapter tried to mirror that directly -- shrink
``self.agents`` to just the signallers after ``reset()``, then to just the
guessers after the first ``step()``. That fails PettingZoo's own compliance
checker (``pettingzoo.test.parallel_api_test``): ``ParallelEnv`` has no notion
of an agent "waiting" mid-episode. Once an agent is live, it must submit an
action on *every* subsequent ``step()`` call until it actually terminates --
there's no third state between "must act" and "done."

So both roles stay live for the whole episode, and each ``step()`` call uses
only the half of the submitted actions that's meaningful for the current
phase, discarding the other half:

    reset()                          -> all 4 agents are in ``self.agents``;
                                         guessers get a zero placeholder
                                         observation (nothing to hear yet)
    step({signallers: signal,        -> signaller entries are the real
          guessers: <ignored>})         signals; guesser entries are
                                         discarded (no signal exists for
                                         them to react to yet). Everyone
                                         stays live; reward is 0.0 for all
                                         (nothing scored yet).
    step({signallers: <ignored>,     -> guesser entries are the real
          guessers: guess})            guesses; signaller entries are
                                         discarded (their decision was
                                         already locked in). Reward is the
                                         team reward, **broadcast to every
                                         agent including signallers** --
                                         Section 4.2's reward is one shared
                                         number, and independent learners
                                         (IQL, Section 4.4.1) each need it
                                         to update, even the signallers who
                                         finished acting a step earlier.
                                         Every agent terminates together.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from gymnasium import spaces
    from pettingzoo import ParallelEnv
except ImportError as exc:  # pragma: no cover - environment dependent
    raise ImportError(
        "GraphSignallingParallelEnv needs gymnasium and pettingzoo. "
        "Install them with `pip install gymnasium pettingzoo`."
    ) from exc

from .graph_signalling_game import EpisodeRecord, GraphSignallingGame
from .graphs import SignallingGraph

AgentName = str
SIGNALLER_PREFIX = "signaller_"
GUESSER_PREFIX = "guesser_"


class GraphSignallingParallelEnv(ParallelEnv):
    """PettingZoo ``ParallelEnv`` view of a :class:`GraphSignallingGame`.

    Parameters mirror :class:`GraphSignallingGame` directly; this class adds no
    new game logic, only the dict-of-agents reshaping PettingZoo expects.
    """

    metadata = {"name": "graph_signalling_game_v0", "render_modes": ["human"]}

    def __init__(
        self,
        graph: SignallingGraph,
        num_item_values: int = 2,
        seed: Optional[int] = None,
        render_mode: Optional[str] = None,
    ) -> None:
        self.game = GraphSignallingGame(graph, num_item_values=num_item_values, seed=seed)
        self.render_mode = render_mode

        # Stable agent-name <-> node-id maps, built once from the graph.
        # PettingZoo agents are identified by strings, our engine by ints, so
        # this is the one place that translation happens.
        self._signaller_name: Dict[int, AgentName] = {
            s: f"{SIGNALLER_PREFIX}{s}" for s in graph.signallers
        }
        self._guesser_name: Dict[int, AgentName] = {
            g: f"{GUESSER_PREFIX}{g}" for g in graph.guessers
        }
        self._signaller_id: Dict[AgentName, int] = {
            v: k for k, v in self._signaller_name.items()
        }
        self._guesser_id: Dict[AgentName, int] = {
            v: k for k, v in self._guesser_name.items()
        }

        self.possible_agents: List[AgentName] = (
            list(self._signaller_name.values()) + list(self._guesser_name.values())
        )
        self.agents: List[AgentName] = []

        # Declare each agent's Gym space straight from the graph's own
        # neighbourhood sizes and the game's own DiscreteSpace sizes -- a
        # signaller with out-degree k sees k items, each in {0..num_item_values-1};
        # a guesser with in-degree k hears k binary signals.
        self._observation_spaces: Dict[AgentName, "spaces.Space"] = {}
        self._action_spaces: Dict[AgentName, "spaces.Space"] = {}
        for s in graph.signallers:
            name = self._signaller_name[s]
            k = graph.out_degree(s)
            self._observation_spaces[name] = spaces.MultiDiscrete([self.game.item_space.n] * k)
            self._action_spaces[name] = spaces.Discrete(self.game.signal_space.n)  # always binary
        for g in graph.guessers:
            name = self._guesser_name[g]
            k = graph.in_degree(g)
            self._observation_spaces[name] = spaces.MultiDiscrete([self.game.signal_space.n] * k)
            self._action_spaces[name] = spaces.Discrete(self.game.guess_space.n)

        # "signals_pending" = waiting on signallers; "guesses_pending" = waiting
        # on guessers; "reset" = episode over, must call reset() again.
        self._phase = "reset"

    # -- PettingZoo required API ---------------------------------------------

    def observation_space(self, agent: AgentName) -> "spaces.Space":
        return self._observation_spaces[agent]

    def action_space(self, agent: AgentName) -> "spaces.Space":
        return self._action_spaces[agent]

    # -- public node-id <-> agent-name maps -----------------------------------
    # Exposed so callers driving this env with per-node Agent objects (e.g.
    # {0: RandomSignaller(), 1: RandomSignaller()}) can translate between the
    # engine's int ids and PettingZoo's string agent names without reaching
    # into private attributes.

    @property
    def signaller_agent_names(self) -> Dict[int, AgentName]:
        """``{signaller_id: agent_name}`` for every signaller in the graph."""
        return dict(self._signaller_name)

    @property
    def guesser_agent_names(self) -> Dict[int, AgentName]:
        """``{guesser_id: agent_name}`` for every guesser in the graph."""
        return dict(self._guesser_name)

    # -- parity with GraphSignallingGame's interface --------------------------
    # These three (graph, item_space, run_episode) match GraphSignallingGame's
    # own names, shapes, and return types exactly. That means code written
    # against the raw engine -- most notably gsg.evaluation.metrics.evaluate()
    # -- runs unchanged against this PettingZoo-backed env too, with no
    # PettingZoo-specific evaluation code needed at all.

    @property
    def graph(self) -> SignallingGraph:
        return self.game.graph

    @property
    def item_space(self):
        return self.game.item_space

    def run_episode(
        self, signaller_agents: Dict[int, "object"], guesser_agents: Dict[int, "object"]
    ) -> EpisodeRecord:
        """Play one episode via reset()/step(); same signature and return
        type as GraphSignallingGame.run_episode(), for callers that don't
        care about the phase-by-phase PettingZoo interface.

        Each phase, the role that isn't acting yet has its dict entries
        filled with ``0`` as harmless filler -- the env discards them (see
        the module docstring for why that's required and safe).

        Also calls ``reset_episode()``/``observe_reward()`` on every agent,
        same as ``GraphSignallingGame.run_episode()`` -- see that method's
        docstring for why. Non-learning agents (random, oracle,
        deterministic) no-op on both, so this is safe for every existing
        caller of this method.
        """
        all_agents = list(signaller_agents.values()) + list(guesser_agents.values())
        for agent in all_agents:
            agent.reset_episode()

        observations, _infos = self.reset()

        actions: Dict[AgentName, int] = {}
        for s, name in self._signaller_name.items():
            obs = tuple(int(x) for x in observations[name])
            actions[name] = signaller_agents[s].act(obs)
        for name in self._guesser_name.values():
            actions[name] = 0
        observations, _rewards, _term, _trunc, _infos = self.step(actions)

        actions = {}
        for g, name in self._guesser_name.items():
            obs = tuple(int(x) for x in observations[name])
            actions[name] = guesser_agents[g].act(obs)
        for name in self._signaller_name.values():
            actions[name] = 0
        _observations, rewards, _term, _trunc, infos = self.step(actions)

        any_agent = self.possible_agents[0]
        for agent in all_agents:
            agent.observe_reward(rewards[any_agent])
        return infos[any_agent]["record"]

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[Dict[AgentName, np.ndarray], Dict[AgentName, dict]]:
        """Start a new episode; every agent (both roles) becomes live."""
        signaller_obs = self.game.reset(seed=seed)
        # Cached so _step_signals can keep handing signallers *an*
        # observation on the next call too -- see module docstring.
        self._cached_signaller_obs = signaller_obs
        self.agents = list(self.possible_agents)
        self._phase = "signals_pending"

        observations: Dict[AgentName, np.ndarray] = {
            self._signaller_name[s]: self._to_array(obs) for s, obs in signaller_obs.items()
        }
        # Guessers haven't heard anything yet. ParallelEnv requires every
        # live agent to have an observation to act on every step, so this is
        # a placeholder -- and it's deliberately -1, not 0, because 0 is a
        # real, meaningful signal value. If this were 0, "not connected yet"
        # would be visually indistinguishable from "connected and heard a
        # 0 bit," which is exactly the ambiguity a placeholder must avoid.
        # (Note: -1 falls outside the declared observation_space's normal
        # {0, 1} range for that reason; whatever a guesser "decides" from it
        # this phase is discarded anyway, so this is safe.)
        for g, name in self._guesser_name.items():
            k = self.game.graph.in_degree(g)
            observations[name] = np.full(k, -1, dtype=np.int64)

        infos = {name: {} for name in self.agents}
        return observations, infos

    def step(self, actions: Dict[AgentName, int]):
        """Advance one phase: signal phase first, then guess phase."""
        if self._phase == "signals_pending":
            return self._step_signals(actions)
        if self._phase == "guesses_pending":
            return self._step_guesses(actions)
        raise RuntimeError(
            "step() called with no episode in progress -- call reset() first."
        )

    def render(self) -> str:
        """Text render of the graph and (once played) the last episode."""
        from ..visualization import render_text  # local import: keeps matplotlib optional
        text = render_text(self.game.graph, self.game.last_record)
        if self.render_mode == "human":
            print(text)
        return text

    def close(self) -> None:
        """No resources to release; present to satisfy the PettingZoo API."""

    # -- phase implementations ------------------------------------------------

    def _step_signals(self, actions: Dict[AgentName, int]):
        # Only the signaller entries are real this phase. Guesser entries
        # are present (ParallelEnv requires every live agent to act every
        # step) but discarded -- there's no signal for them to react to yet.
        signals = {
            s: int(actions[name]) for s, name in self._signaller_name.items()
        }
        guesser_obs = self.game.submit_signals(signals)
        self._phase = "guesses_pending"
        # self.agents is unchanged: nobody has terminated yet, so everybody
        # who was live stays live (see module docstring).

        observations: Dict[AgentName, np.ndarray] = {
            self._signaller_name[s]: self._to_array(obs)
            for s, obs in self._cached_signaller_obs.items()
        }
        observations.update(
            {self._guesser_name[g]: self._to_array(obs) for g, obs in guesser_obs.items()}
        )
        # Nothing has been scored yet -- the team reward only exists once
        # guessers act -- so every agent gets an honest 0.0 here rather than
        # a placeholder guess at what it might later earn.
        all_names = self.possible_agents
        rewards = {name: 0.0 for name in all_names}
        terminations = {name: False for name in all_names}
        truncations = {name: False for name in all_names}
        infos: Dict[AgentName, dict] = {name: {} for name in all_names}
        return observations, rewards, terminations, truncations, infos

    def _step_guesses(self, actions: Dict[AgentName, int]):
        # Only the guesser entries are real this phase. Signaller entries
        # are present but discarded -- their decision was already locked in
        # during the signal phase.
        guesses = {
            g: int(actions[name]) for g, name in self._guesser_name.items()
        }
        reward, record = self.game.submit_guesses(guesses)

        # Broadcast the one shared team reward to every agent (see module
        # docstring) and end the episode for everyone.
        all_names = self.possible_agents
        observations = {name: self._final_observation(name, record) for name in all_names}
        rewards = {name: reward for name in all_names}
        terminations = {name: True for name in all_names}
        truncations = {name: False for name in all_names}
        infos: Dict[AgentName, dict] = {name: {"record": record} for name in all_names}

        self.agents = []  # PettingZoo convention: episode over, nobody is live
        self._phase = "reset"
        return observations, rewards, terminations, truncations, infos

    # -- helpers ---------------------------------------------------------------

    @staticmethod
    def _to_array(observation) -> np.ndarray:
        """Cast an engine observation tuple to the array shape Gym spaces expect."""
        return np.array(observation, dtype=np.int64)

    def _final_observation(self, name: AgentName, record: EpisodeRecord) -> np.ndarray:
        """Each agent's last-seen observation, for the terminal step's return."""
        if name in self._signaller_id:
            s = self._signaller_id[name]
            return self._to_array(record.signaller_observations[s])
        g = self._guesser_id[name]
        return self._to_array(record.guesser_observations[g])
