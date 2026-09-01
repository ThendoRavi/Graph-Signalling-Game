"""Convention heatmaps for the random and oracle baselines on K_{2,2}.

Companion to convention_heatmaps_k22.py (which does this for the three
trained IQL variants), applied to the two reference baselines instead:

* The **oracle** has an actual, fixed rulebook (Section 4.5.4: S0 tracks
  G0, S1 tracks G1, verbatim) -- its heatmap is deterministic and will look
  identical every time this script runs.
* **Random** has no rulebook at all -- RandomSignaller/RandomGuesser ignore
  their observation and answer by coin flip every single call, with no
  epsilon to force "greedy" (there's no greedy policy to extract). Its
  heatmap is therefore just *one arbitrary sample* of what a patternless
  agent happened to do this run, not a stable convention -- rerunning this
  script will (and should) produce a different-looking random heatmap. It's
  included specifically so you can see, side by side, what "no convention"
  looks like next to what an actual rule looks like.

Run:  python experiments/convention_heatmaps_baselines_k22.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from gsg.baselines.oracle_baseline import k22_oracle
from gsg.baselines.random_baseline import RandomGuesser, RandomSignaller
from gsg.environment.graphs import SignallingGraph
from gsg.evaluation.conventions import convention_summary, exhaustive_joint_table, extract_all_policies
from gsg.visualization import ITEM_NAMES, SIGNAL_NAMES, render_policy_heatmap

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "question1", "figures")


def _item_name(value: int) -> str:
    return ITEM_NAMES.get(value, str(value))


def _signal_name(value: int) -> str:
    return SIGNAL_NAMES.get(value, str(value))


def print_joint_table(outcomes) -> None:
    for o in outcomes:
        items_str = ", ".join(f"G{g}={_item_name(v)}" for g, v in sorted(o.items.items()))
        signals_str = ", ".join(f"S{s}={_signal_name(v)}" for s, v in sorted(o.signals.items()))
        guesses_str = ", ".join(
            f"G{g}={_item_name(v)}{'(OK)' if o.correct[g] else '(XX)'}"
            for g, v in sorted(o.guesses.items())
        )
        print(f"    items[{items_str}] -> signals[{signals_str}] -> guesses[{guesses_str}]  R={o.reward:.2f}")


def make_graph():
    return SignallingGraph.complete_bipartite(2, 2)


def run_oracle() -> None:
    print("\n--- oracle (Section 4.5.4: S0 tracks G0, S1 tracks G1) ---")
    graph = make_graph()
    signaller_agents, guesser_agents = k22_oracle()

    outcomes = exhaustive_joint_table(graph, 2, signaller_agents, guesser_agents)
    print(f"    {convention_summary(outcomes)}")
    print_joint_table(outcomes)

    policies = extract_all_policies(graph, 2, signaller_agents, guesser_agents)
    save_path = os.path.join(FIGURES_DIR, "conventions_oracle.png")
    render_policy_heatmap(policies, title="oracle (deterministic, fixed rule)", save_path=save_path)
    print(f"    saved heatmap -> {os.path.normpath(save_path)}")


def run_random(seed: int) -> None:
    print(f"\n--- random baseline, seed {seed} (no rule -- one arbitrary sample) ---")
    graph = make_graph()
    signaller_agents = {0: RandomSignaller(seed=seed), 1: RandomSignaller(seed=seed + 1)}
    guesser_agents = {0: RandomGuesser(seed=seed + 2), 1: RandomGuesser(seed=seed + 3)}

    outcomes = exhaustive_joint_table(graph, 2, signaller_agents, guesser_agents)
    print(f"    {convention_summary(outcomes)}")
    print_joint_table(outcomes)

    policies = extract_all_policies(graph, 2, signaller_agents, guesser_agents)
    save_path = os.path.join(FIGURES_DIR, f"conventions_random_seed{seed}.png")
    render_policy_heatmap(
        policies, title=f"random baseline, seed {seed} (no rule -- one arbitrary sample)",
        save_path=save_path,
    )
    print(f"    saved heatmap -> {os.path.normpath(save_path)}")


def main() -> None:
    os.makedirs(FIGURES_DIR, exist_ok=True)
    print("=" * 72)
    print("Baseline convention heatmaps, K_{2,2}")
    print("=" * 72)

    run_oracle()
    run_random(seed=0)
    run_random(seed=1)


if __name__ == "__main__":
    main()
