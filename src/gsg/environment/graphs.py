"""Signalling-graph constructionn.

This module owns the *structure* of the game - who can talk to whom - and
nothing about learning or reward. 

Conventions used throughout the codebase
----------------------------------------
* Signallers and guessers each get their **own** integer ids starting at 0, so
  a graph always refers to "signaller 0", "guesser 0", etc. A *directed edge*
  is the pair ``(signaller_id, guesser_id)``: signaller -> guesser, reflecting
  the one-way flow of information (Section 4.2).
* ``out_neighbours(s)`` = the guessers a signaller observes and broadcasts to
  (its out-neighbourhood N(i)). ``in_neighbours(g)`` = the signallers a guesser
  hears (its in-neighbourhood N(j)). Both are returned **sorted by id**, so the
  observation vectors built from them have a stable, deterministic ordering.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Set, Tuple

# A directed edge is (signaller_id -> guesser_id).
Edge = Tuple[int, int]


class SignallingGraph:
    """Directed bipartite-style signalling graph ``G = (N, E)`` .

    Signallers and guessers are stored in separate id namespaces (both starting
    at 0). Only signaller -> guesser edges are allowed; the class rejects any
    attempt to create an edge that would violate that direction.
    """

    def __init__(self) -> None:
        self._num_signallers: int = 0
        self._num_guessers: int = 0
        # Adjacency stored both ways .
        self._out: Dict[int, Set[int]] = {}   # signaller_id -> {guesser_id, ...}
        self._in: Dict[int, Set[int]] = {}    # guesser_id   -> {signaller_id, ...}

    # -- construction --------------------------------------------------------

    def add_signaller(self) -> int:
        """Add a signaller node and return its new id."""
        node_id = self._num_signallers
        self._num_signallers += 1
        self._out[node_id] = set()
        return node_id

    def add_guesser(self) -> int:
        """Add a guesser node and return its new id."""
        node_id = self._num_guessers
        self._num_guessers += 1
        self._in[node_id] = set()
        return node_id

    def add_edge(self, signaller_id: int, guesser_id: int) -> None:
        """Connect ``signaller_id`` -> ``guesser_id`` (idempotent)."""
        if signaller_id not in self._out:
            raise ValueError(f"unknown signaller id {signaller_id}")
        if guesser_id not in self._in:
            raise ValueError(f"unknown guesser id {guesser_id}")
        self._out[signaller_id].add(guesser_id)
        self._in[guesser_id].add(signaller_id)

    # -- queries -------------------------------------------------------------

    @property
    def signallers(self) -> List[int]:
        """Ids of all signallers, ascending."""
        return list(range(self._num_signallers))

    @property
    def guessers(self) -> List[int]:
        """Ids of all guessers, ascending."""
        return list(range(self._num_guessers))

    @property
    def edges(self) -> List[Edge]:
        """All ``(signaller, guesser)`` edges, sorted for determinism."""
        return sorted((s, g) for s, gs in self._out.items() for g in gs)

    def out_neighbours(self, signaller_id: int) -> List[int]:
        """Guessers that ``signaller_id`` observes/broadcasts to (N(i)), sorted."""
        return sorted(self._out[signaller_id])

    def in_neighbours(self, guesser_id: int) -> List[int]:
        """Signallers that ``guesser_id`` hears (N(j)), sorted."""
        return sorted(self._in[guesser_id])

    def out_degree(self, signaller_id: int) -> int:
        """Number of guessers a signaller must serve with one bit (delta_S)."""
        return len(self._out[signaller_id])

    def in_degree(self, guesser_id: int) -> int:
        """Number of signals a guesser receives (delta_G)."""
        return len(self._in[guesser_id])

    def has_full_guesser_coverage(self) -> bool:
        """Q2 hard constraint: every guesser has >= 1 incoming edge (Section 4.7.2)."""
        return all(len(self._in[g]) >= 1 for g in self.guessers)

    def validate(self) -> None:
        """Raise if the graph is ill-formed for play.

        Enforces the game's structural rules: there must be at least one
        signaller and one guesser, and every guesser must be able to hear at
        least one signal (otherwise its guess is pure chance and it cannot take
        part in convention formation).
        """
        if self._num_signallers == 0:
            raise ValueError("graph has no signallers")
        if self._num_guessers == 0:
            raise ValueError("graph has no guessers")
        if not self.has_full_guesser_coverage():
            uncovered = [g for g in self.guessers if self.in_degree(g) == 0]
            raise ValueError(f"guessers with no incoming signal: {uncovered}")

    def __repr__(self) -> str:
        return (
            f"SignallingGraph(|N_S|={self._num_signallers}, "
            f"|N_G|={self._num_guessers}, |E|={len(self.edges)})"
        )

    # -- factories -----------------------------------------------------------

    @classmethod
    def single_pair(cls) -> "SignallingGraph":
        """The minimal game: one signaller -> one guesser (the base case).

        This is the simplest non-degenerate signalling problem (the two-agent
        "on for cat, off for dog" case from the room analogy in Section 4.1),
        and the starting point for verifying the environment's rules.
        """
        g = cls()
        s0 = g.add_signaller()
        g0 = g.add_guesser()
        g.add_edge(s0, g0)
        return g

    @classmethod
    def complete_bipartite(cls, num_signallers: int, num_guessers: int) -> "SignallingGraph":
        """Complete bipartite graph K_{m,n}: every signaller -> every guesser.

        ``complete_bipartite(2, 2)`` is the Question 1 experimental graph
        K_{2,2} (Section 4.6.1). Provided now so the same code path scales up
        later; only ``single_pair`` is exercised at this stage.
        """
        g = cls()
        signallers = [g.add_signaller() for _ in range(num_signallers)]
        guessers = [g.add_guesser() for _ in range(num_guessers)]
        for s in signallers:
            for gu in guessers:
                g.add_edge(s, gu)
        return g

    @classmethod
    def from_adjacency_matrix(cls, matrix: Sequence[Sequence[int]]) -> "SignallingGraph":
        """Build *any* signalling graph from a plain adjacency matrix.

        ``matrix[s][g]`` is truthy iff signaller ``s`` is connected to
        guesser ``g``. Row count = number of signallers, column count =
        number of guessers -- so this is the one-call way to try a graph
        shape that doesn't already have a named factory above:

            SignallingGraph.from_adjacency_matrix([
                [1, 1],   # S0 -> G0, S0 -> G1
                [0, 1],   # S1 -> G1 only
            ])

        This is deliberately the *only* thing you need to touch to change
        the game's structure everywhere else in the codebase -- the
        environment, the PettingZoo adapter, and (once built generically)
        the shared IQL network all derive their shapes from the graph
        object itself, not from any hard-coded agent count.
        """
        if len(matrix) == 0:
            raise ValueError("adjacency matrix must have at least one row (signaller)")
        num_guessers = len(matrix[0])
        if any(len(row) != num_guessers for row in matrix):
            raise ValueError("adjacency matrix rows must all have the same length")

        g = cls()
        signallers = [g.add_signaller() for _ in range(len(matrix))]
        guessers = [g.add_guesser() for _ in range(num_guessers)]
        for s, row in zip(signallers, matrix):
            for gu, connected in zip(guessers, row):
                if connected:
                    g.add_edge(s, gu)
        return g
