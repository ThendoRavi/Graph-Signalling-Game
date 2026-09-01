"""Visualisation of the Graph Signalling Game.

Three renderers, so you can "see" the environment and its learned
conventions at different levels of detail:

* :func:`render_text` -- a zero-dependency ASCII view (the room-analogy light
  switches). Always available; great for quick terminal sanity checks.
* :func:`render_matplotlib` -- a proper node-link diagram of the signalling
  graph, with signallers on the left and guessers on the right, coloured by the
  current episode's items / signals / guesses. Requires matplotlib (optional);
  the function raises a clear message if it is missing.
* :func:`render_policy_heatmap` -- a small grid heatmap per agent showing its
  *entire* greedy policy (every possible observation -> its action), not just
  one played episode -- this is what actually shows "the convention," as
  opposed to one sample from it. Pairs with
  :func:`gsg.evaluation.conventions.extract_all_policies`.

The first two take a :class:`~gsg.environment.graphs.SignallingGraph` and,
optionally, an :class:`~gsg.environment.graph_signalling_game.EpisodeRecord`.
With a record they show a concrete played episode; without one they just show
the topology.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

from .environment.graph_signalling_game import EpisodeRecord
from .environment.graphs import SignallingGraph

# Room-analogy names for raw bit values (Section 4.1's "on for cat, off for
# dog" framing), used only by render_text() to make convention formation
# readable at a glance: which convention a run settled on -- OFF-means-CAT or
# OFF-means-DOG, and vice versa -- is exactly the thing Question 1 asks
# about, so it needs to be visible directly in the printed trace, not just
# inferable from raw 0/1s. Falls back to the raw number for any value
# outside this table (e.g. the ternary item space, Section 4.6.6, which adds
# a third value this table doesn't name).
ITEM_NAMES = {0: "CAT", 1: "DOG"}
SIGNAL_NAMES = {0: "OFF", 1: "ON"}


def _item_name(value: int) -> str:
    return ITEM_NAMES.get(value, str(value))


def _signal_name(value: int) -> str:
    return SIGNAL_NAMES.get(value, str(value))


# ---------------------------------------------------------------------------
# Text renderer (stdlib only)
# ---------------------------------------------------------------------------

def render_text(graph: SignallingGraph, record: Optional[EpisodeRecord] = None) -> str:
    """Return a human-readable ASCII summary of the graph (and episode).

    Without a record: lists nodes and edges. With a record: shows each
    signaller's observation and emitted signal, and each guesser's received
    signal(s), hidden item, guess, and whether it was correct -- using the
    room-analogy names (CAT/DOG, Light Switch ON/OFF) rather than raw 0/1s,
    so the specific convention a run converged on is readable directly from
    the trace instead of having to be decoded by hand.
    """
    lines = [f"Graph: {graph!r}"]

    lines.append("Edges (signaller -> guesser):")
    for s, g in graph.edges:
        lines.append(f"  S{s} -> G{g}")

    if record is None:
        return "\n".join(lines)

    lines.append("")
    lines.append(f"Episode  team reward R = {record.reward:.3f}")
    lines.append("Signallers:")
    for s in graph.signallers:
        obs = record.signaller_observations[s]
        bit = record.signals[s]
        noun = "item" if len(obs) == 1 else "items"
        seen = ", ".join(_item_name(v) for v in obs)
        lines.append(
            f"  S{s}: sees {noun} ({seen}) -> signal: Light Switch {_signal_name(bit)}"
        )

    lines.append("Guessers:")
    for g in graph.guessers:
        obs = record.guesser_observations[g]
        item = record.items[g]
        guess = record.guesses[g]
        mark = "OK " if record.correct[g] else "XX "
        noun = "signal" if len(obs) == 1 else "signals"
        heard = ", ".join(f"Light Switch {_signal_name(v)}" for v in obs)
        lines.append(
            f"  G{g}: hears {noun} ({heard}) | "
            f"item={_item_name(item)} guess={_item_name(guess)} [{mark}]"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Matplotlib renderer (optional dependency)
# ---------------------------------------------------------------------------

def render_matplotlib(
    graph: SignallingGraph,
    record: Optional[EpisodeRecord] = None,
    ax: "object" = None,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
):
    """Draw the signalling graph as a node-link diagram.

    Signallers are placed on the left column, guessers on the right, with
    directed edges between them. When a ``record`` is supplied:

    * signaller nodes are gold if their emitted signal is 1 ("light on"),
      grey if 0, and annotated with the items they observed;
    * guesser nodes are green if the guess was correct and red otherwise, and
      annotated with ``item`` vs ``guess``.

    Returns the matplotlib ``Axes``. Pass ``save_path`` to also write a PNG.
    Requires matplotlib; raises ``ImportError`` with guidance if unavailable.
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle, FancyArrowPatch
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "render_matplotlib needs matplotlib. Install it with "
            "`pip install matplotlib`, or use render_text() instead."
        ) from exc

    signallers = graph.signallers
    guessers = graph.guessers

    # --- node positions: two vertical columns, centred ---------------------
    def _column_positions(ids, x):
        n = len(ids)
        # Spread nodes over a fixed height, centred on 0.
        return {
            node: (x, (n - 1) / 2.0 - i)  # top-to-bottom in id order
            for i, node in enumerate(ids)
        }

    pos = {}
    pos.update({("S", s): p for s, p in _column_positions(signallers, x=0.0).items()})
    pos.update({("G", g): p for g, p in _column_positions(guessers, x=3.0).items()})

    created_ax = ax is None
    if created_ax:
        _, ax = plt.subplots(figsize=(6, max(3, 1.4 * max(len(signallers), len(guessers)))))

    # --- edges -------------------------------------------------------------
    for s, g in graph.edges:
        x0, y0 = pos[("S", s)]
        x1, y1 = pos[("G", g)]
        arrow = FancyArrowPatch(
            (x0 + 0.35, y0), (x1 - 0.35, y1),
            arrowstyle="-|>", mutation_scale=14,
            color="#888888", linewidth=1.2, zorder=1,
        )
        ax.add_patch(arrow)

    # --- nodes -------------------------------------------------------------
    def _draw_node(center, label, facecolor, sublabel=None):
        x, y = center
        ax.add_patch(Circle((x, y), 0.32, facecolor=facecolor,
                            edgecolor="black", linewidth=1.3, zorder=2))
        ax.text(x, y, label, ha="center", va="center",
                fontsize=11, fontweight="bold", zorder=3)
        if sublabel:
            ax.text(x, y - 0.5, sublabel, ha="center", va="center",
                    fontsize=8, color="#333333", zorder=3)

    for s in signallers:
        if record is not None:
            signal = record.signals[s]
            face = "#ffd23f" if signal == 1 else "#cccccc"  # light on/off
            sub = f"sees {record.signaller_observations[s]} -> {signal}"
        else:
            face, sub = "#9ecae1", None
        _draw_node(pos[("S", s)], f"S{s}", face, sub)

    for g in guessers:
        if record is not None:
            face = "#7bd389" if record.correct[g] else "#f08a8a"  # right/wrong
            sub = f"item {record.items[g]} / guess {record.guesses[g]}"
        else:
            face, sub = "#c7c7f0", None
        _draw_node(pos[("G", g)], f"G{g}", face, sub)

    # --- framing -----------------------------------------------------------
    if title is None:
        title = "Graph Signalling Game"
        if record is not None:
            title += f"  (R = {record.reward:.2f})"
    ax.set_title(title)
    ax.text(0.0, _top(pos) + 0.9, "signallers", ha="center", fontsize=9, color="#555")
    ax.text(3.0, _top(pos) + 0.9, "guessers", ha="center", fontsize=9, color="#555")
    ax.set_xlim(-1.0, 4.0)
    ax.set_ylim(_bottom(pos) - 1.0, _top(pos) + 1.4)
    ax.set_aspect("equal")
    ax.axis("off")

    if save_path is not None:
        ax.figure.savefig(save_path, bbox_inches="tight", dpi=130)

    return ax


def _top(pos) -> float:
    return max((y for _, y in pos.values()), default=0.0)


def _bottom(pos) -> float:
    return min((y for _, y in pos.values()), default=0.0)


# ---------------------------------------------------------------------------
# Policy heatmap (optional dependency) -- convergence analysis
# ---------------------------------------------------------------------------

def render_policy_heatmap(
    policies: Dict[str, Dict[Tuple[int, int], int]],
    title: Optional[str] = None,
    save_path: Optional[str] = None,
):
    """Render every agent's *entire* greedy policy as a small 2x2 grid --
    one heatmap per agent, arranged in a row.

    ``policies`` is the output of
    :func:`gsg.evaluation.conventions.extract_all_policies`: ``{"S0": {(0,
    0): 1, (0, 1): 0, ...}, "G0": {...}, ...}``. Each cell shows what that
    agent does for one possible pair of observed values -- for a signaller,
    whether it emits Light Switch ON/OFF; for a guesser, whether it guesses
    CAT/DOG. Reading a whole row of these side by side is the fastest way to
    see whether two signallers (or two guessers) converged to the *same*
    rule or to *different, complementary* ones -- exactly the question
    Section 4.6.3's "convention" definition is asking, made visible instead
    of inferred from a reward number.

    Only supports 2-neighbour, binary-observation agents (K_{2,2}'s uniform
    structure) -- a 2x2 grid doesn't generalise to other neighbourhood
    sizes without a different layout. Requires matplotlib; raises
    ``ImportError`` with guidance if unavailable.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "render_policy_heatmap needs matplotlib. Install it with "
            "`pip install matplotlib`."
        ) from exc

    # Signallers first (numeric order), then guessers -- reads the same
    # left-to-right order as render_text()'s "Signallers: / Guessers:" split.
    labels = sorted(policies.keys(), key=lambda label: (label[0] != "S", int(label[1:])))

    fig, axes = plt.subplots(1, len(labels), figsize=(2.6 * len(labels), 3.2))
    if len(labels) == 1:
        axes = [axes]

    for ax, label in zip(axes, labels):
        policy = policies[label]
        obs_keys = sorted(policy.keys())
        values = sorted({v for obs in obs_keys for v in obs})
        if len(values) != 2 or any(len(obs) != 2 for obs in obs_keys):
            raise ValueError(
                f"render_policy_heatmap only supports 2-neighbour, binary-valued "
                f"policies; {label}'s policy has observations {obs_keys}"
            )

        is_signaller = label.startswith("S")
        if is_signaller:
            action_colors = {0: "#cccccc", 1: "#ffd23f"}   # off / on (light bulb)
            action_text = {0: "OFF", 1: "ON"}
            axis_name = _item_name    # a signaller observes items
            role_word = "sees items"
        else:
            action_colors = {0: "#9ecae1", 1: "#fdae61"}   # cat / dog (guess)
            action_text = {0: "CAT", 1: "DOG"}
            axis_name = _signal_name  # a guesser observes signals
            role_word = "hears signals"

        for r_idx, r in enumerate(values):
            for c_idx, c in enumerate(values):
                action = policy[(r, c)]
                ax.add_patch(plt.Rectangle(
                    (c_idx, 1 - r_idx), 1, 1,
                    facecolor=action_colors.get(action, "#dddddd"),
                    edgecolor="black", linewidth=1.0,
                ))
                ax.text(c_idx + 0.5, 1 - r_idx + 0.5, action_text.get(action, str(action)),
                        ha="center", va="center", fontsize=9, fontweight="bold")

        ax.set_xlim(0, 2)
        ax.set_ylim(0, 2)
        ax.set_xticks([0.5, 1.5])
        ax.set_yticks([0.5, 1.5])
        ax.set_xticklabels([axis_name(v) for v in values], fontsize=8)
        ax.set_yticklabels([axis_name(v) for v in reversed(values)], fontsize=8)
        ax.set_xlabel("neighbour 1", fontsize=7)
        ax.set_ylabel("neighbour 0", fontsize=7)
        ax.set_title(f"{label}\n({role_word})", fontsize=9)
        ax.set_aspect("equal")

    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight", dpi=130)

    return fig
