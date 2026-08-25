"""Tests for graph construction (Section 4.6.1): K_{2,2} adjacency,
neighbourhoods, and full guesser coverage.

Runnable two ways:
    pytest tests/                # if pytest is installed
    python tests/test_graphs.py   # falls back to a built-in runner
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from gsg.environment.graphs import SignallingGraph


def _k22() -> SignallingGraph:
    return SignallingGraph.complete_bipartite(2, 2)


def test_k22_node_counts():
    """K_{2,2} has exactly two signallers and two guessers."""
    g = _k22()
    assert g.signallers == [0, 1]
    assert g.guessers == [0, 1]


def test_k22_is_fully_connected():
    """Every signaller connects to every guesser: 2 x 2 = 4 edges."""
    g = _k22()
    assert set(g.edges) == {(0, 0), (0, 1), (1, 0), (1, 1)}
    assert len(g.edges) == 4


def test_k22_neighbourhoods():
    """Each signaller sees both guessers; each guesser hears both signallers."""
    g = _k22()
    for s in g.signallers:
        assert g.out_neighbours(s) == [0, 1]
        assert g.out_degree(s) == 2
    for guesser in g.guessers:
        assert g.in_neighbours(guesser) == [0, 1]
        assert g.in_degree(guesser) == 2


def test_k22_full_guesser_coverage():
    """K_{2,2} satisfies the hard coverage constraint (Section 4.7.2)."""
    g = _k22()
    assert g.has_full_guesser_coverage()
    g.validate()  # should not raise


def test_k22_scales_from_single_pair():
    """complete_bipartite(1, 1) is exactly the base-case single_pair() graph."""
    base = SignallingGraph.single_pair()
    scaled = SignallingGraph.complete_bipartite(1, 1)
    assert base.edges == scaled.edges == [(0, 0)]


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for t in tests:
        t()
        print(f"  PASS {t.__name__}")
        passed += 1
    print(f"\n{passed}/{len(tests)} graph tests passed.")


if __name__ == "__main__":
    _run_all()
