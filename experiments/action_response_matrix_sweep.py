"""Action-response matrix sweep: every graph size x every IQL variant,
measured LIVE during training.

This is the batch job meant to run on the SSH / SLURM server (see
``train.sh``). For each complete bipartite graph K_{m,m} with
``m in {3, 4, 5, 6}`` it:

  1. builds a fresh environment on that graph;
  2. trains each of the three IQL parameter-sharing variants on it
     (shared-brain, two-brain, independent) with exploration left ON the
     whole time -- epsilon decays naturally on its normal schedule, exactly
     as it does during real training -- and tallies the action-response
     matrix from *every episode as it is played*
     (:class:`gsg.evaluation.conventions.ActionResponseTally`, fed via each
     trainer's ``on_episode`` hook). The random baseline is played (never
     trained) for the *same number of episodes* as that graph's training
     schedule, and tallied the same way, so every variant -- learning or
     not -- is measured over an identical episode budget. This is
     deliberately *not* a frozen, fully-greedy post-training snapshot: it
     shows what a still-learning, still-exploring population actually did,
     which is the only way the random baseline (which never stops exploring)
     is a fair, apples-to-apples comparison point.
  3. separately reports the *settled* greedy team reward (epsilon forced to
     0, a fresh short evaluation batch) purely as a summary statistic --
     this number is independent of, and does not feed into, the
     action-response matrix itself.
  4. saves one action-response figure per (graph, variant) into a
     per-graph-size subdirectory, plus prints a full text summary of every
     edge's peak cell.

Outputs
-------
* Figures  -> ``results/question1/figures/k{m}x{m}/action_response_{variant}.png``
              ``results/question1/figures/k{m}x{m}/graph_topology.png``
* Text log -> whatever captures stdout. On SLURM that's the job's
  ``--output`` file; ``train.sh`` additionally tees stdout to
  ``results/question1/logs/action_response_sweep_<timestamp>.log``.

Everything is driven off ``TRAIN_SEED`` and the per-size ``SCHEDULE`` below,
so a rerun reproduces the same numbers (trainer.py seeds torch + Python's
``random``; the env seeds its own item sampler).

Run:  python experiments/action_response_matrix_sweep.py
"""

from __future__ import annotations

import os
import sys
import time

import matplotlib
matplotlib.use("Agg")  # headless server: no display, write PNGs straight to disk
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from gsg.baselines.random_baseline import RandomGuesser, RandomSignaller
from gsg.environment.graphs import SignallingGraph
from gsg.environment.pettingzoo_env import GraphSignallingParallelEnv
from gsg.evaluation.conventions import ActionResponseTally
from gsg.evaluation.metrics import evaluate
from gsg.training.trainer import (
    train_independent_iql,
    train_shared_iql,
    train_two_brain_iql,
)
from gsg.visualization import render_action_response_matrix, render_matplotlib

# --- configuration ---------------------------------------------------------

# K_{m,m}: m signallers, m guessers, every signaller wired to every guesser.
GRAPH_SIZES = [3, 4, 5, 6]

# Bigger graphs put more agents through a harder joint-coordination problem
# at once, so they need a longer run to settle. (num_training_episodes,
# epsilon_decay_episodes) per m. The schedule *shape* stays the proposal's
# (linear epsilon 1.0 -> 0.05); only its length scales with the graph. This
# is also, now, the exact number of episodes the live action-response tally
# is built from for every variant at that size, including random.
SCHEDULE = {
    3: (5_000, 3_200),
    4: (7_000, 4_500),
    5: (9_000, 6_000),
    6: (12_000, 8_000),
}

# Variants to sweep, in report order. "random" is the no-convention
# reference point; the other three are the parameter-sharing designs.
VARIANTS = ["random", "shared", "two_brain", "independent"]

# Separate from the live tally above -- this is just a short, fully-greedy
# batch used to report a single "how good did this end up being" summary
# number alongside the matrix, on a held-out item stream.
EVAL_EPISODES = 2_000
TRAIN_SEED = 0
EVAL_SEED_OFFSET = 10_000  # keeps the eval item stream off the training one

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "question1", "figures")


# --- helpers -------------------------------------------------------------

def make_env(m: int, seed=None) -> GraphSignallingParallelEnv:
    return GraphSignallingParallelEnv(SignallingGraph.complete_bipartite(m, m), seed=seed)


def graph_figures_dir(m: int) -> str:
    """Per-graph-size output directory, e.g. .../figures/k3x3/."""
    path = os.path.join(FIGURES_DIR, f"k{m}x{m}")
    os.makedirs(path, exist_ok=True)
    return path


def build_population_with_live_tally(variant: str, env, num_episodes: int, decay: int):
    """Return ``(signaller_agents, guesser_agents, tally)`` for ``variant``.

    The action-response tally is built from the *live* episode stream --
    exploration on, epsilon following its normal decay schedule for the
    three IQL variants -- rather than from a separate frozen-greedy
    measurement pass. "random" has no training loop of its own, so it is
    instead simply played for ``num_episodes`` (matching the trained
    variants' training length at this graph size) and tallied the same way,
    putting every variant on the same episode axis.
    """
    tally = ActionResponseTally(env.graph)

    if variant == "random":
        signaller_agents = {
            s: RandomSignaller(seed=1_000 + s) for s in env.graph.signallers
        }
        guesser_agents = {
            g: RandomGuesser(num_item_values=env.item_space.n, seed=2_000 + g)
            for g in env.graph.guessers
        }
        for _ in range(num_episodes):
            record = env.run_episode(signaller_agents, guesser_agents)
            tally.add(record)
        return signaller_agents, guesser_agents, tally

    def on_episode(episode, record) -> None:
        tally.add(record)

    if variant == "shared":
        _net, signaller_agents, guesser_agents, _hist = train_shared_iql(
            env, num_episodes=num_episodes, epsilon_decay_episodes=decay,
            on_episode=on_episode, seed=TRAIN_SEED,
        )
        return signaller_agents, guesser_agents, tally

    if variant == "two_brain":
        _s, _g, signaller_agents, guesser_agents, _hist = train_two_brain_iql(
            env, num_episodes=num_episodes, epsilon_decay_episodes=decay,
            on_episode=on_episode, seed=TRAIN_SEED,
        )
        return signaller_agents, guesser_agents, tally

    if variant == "independent":
        _s, _g, signaller_agents, guesser_agents, _hist = train_independent_iql(
            env, num_episodes=num_episodes, epsilon_decay_episodes=decay,
            on_episode=on_episode, seed=TRAIN_SEED,
        )
        return signaller_agents, guesser_agents, tally

    raise ValueError(f"unknown variant {variant!r}")


def print_edge_peaks(freqs) -> None:
    """One line per edge: the biggest (signal, guess) cell and its share."""
    for (s, g), table in sorted(freqs.items()):
        (signal, guess), frac = max(table.items(), key=lambda kv: kv[1])
        light = "OFF" if signal == 0 else "ON"
        item = "CAT" if guess == 0 else "DOG"
        print(f"      S{s}->G{g}: peak {light:3s}/{item:3s} = {frac * 100:5.1f}%")


# --- main ---------------------------------------------------------------

def main() -> None:
    os.makedirs(FIGURES_DIR, exist_ok=True)
    overall_start = time.time()

    print("=" * 72)
    print("Action-response matrix sweep: graph sizes x IQL variants (LIVE)")
    print(f"graphs      : {', '.join(f'K{m}x{m}' for m in GRAPH_SIZES)}")
    print(f"variants    : {', '.join(VARIANTS)}")
    print("measurement : tallied from every episode of the training run itself")
    print("              (exploration ON, epsilon decaying naturally); random")
    print("              is played for the same number of episodes for a fair")
    print("              apples-to-apples comparison")
    print(f"train seed  : {TRAIN_SEED}")
    print("=" * 72, flush=True)

    for m in GRAPH_SIZES:
        num_episodes, decay = SCHEDULE[m]
        tag = f"K{m}x{m}"
        num_edges = m * m
        out_dir = graph_figures_dir(m)
        print(f"\n{'#' * 72}")
        print(f"# {tag}  ({m} signallers, {m} guessers, {num_edges} edges)")
        print(f"#   training schedule: {num_episodes} episodes, epsilon decay over {decay}")
        print(f"#   output directory: {os.path.normpath(out_dir)}")
        print(f"{'#' * 72}", flush=True)

        # Topology diagram for the graph itself (no episode -> just the
        # signaller/guesser node-link structure).
        topo_path = os.path.join(out_dir, "graph_topology.png")
        topo_ax = render_matplotlib(
            SignallingGraph.complete_bipartite(m, m),
            title=f"{tag}: complete bipartite signalling graph",
            save_path=topo_path,
        )
        plt.close(topo_ax.figure)
        print(f"  topology saved -> {os.path.normpath(topo_path)}", flush=True)

        for variant in VARIANTS:
            t0 = time.time()
            train_env = make_env(m, seed=TRAIN_SEED)
            signaller_agents, guesser_agents, tally = build_population_with_live_tally(
                variant, train_env, num_episodes, decay
            )
            run_secs = time.time() - t0
            freqs = tally.frequencies()

            # Separate, short, fully-greedy pass -- purely a summary
            # statistic of where the population ended up, independent of the
            # live action-response tally above.
            for agent in list(signaller_agents.values()) + list(guesser_agents.values()):
                if hasattr(agent, "epsilon"):
                    agent.epsilon = 0.0
            eval_env = make_env(m, seed=TRAIN_SEED + EVAL_SEED_OFFSET)
            eval_result = evaluate(
                eval_env, signaller_agents, guesser_agents,
                episodes=EVAL_EPISODES, seed=TRAIN_SEED + EVAL_SEED_OFFSET,
            )
            reward_label = "settled greedy team reward" if variant != "random" else "team reward"

            print(f"\n  --- {tag} / {variant} ---")
            print(f"      {tally.num_episodes} live episodes tallied "
                  f"(exploration on) in {run_secs:6.1f}s")
            print(f"      {reward_label}: {eval_result.mean_reward:.4f}  "
                  f"[{eval_result.band}]")
            print_edge_peaks(freqs)

            save_path = os.path.join(out_dir, f"action_response_{variant}.png")
            fig = render_action_response_matrix(
                freqs,
                title=f"{tag}  --  {variant}  (LIVE, exploration on, "
                      f"{tally.num_episodes} episodes; "
                      f"settled greedy R={eval_result.mean_reward:.3f})",
                save_path=save_path,
            )
            plt.close(fig)
            print(f"      saved -> {os.path.normpath(save_path)}", flush=True)

    total = time.time() - overall_start
    print(f"\n{'=' * 72}")
    print(f"sweep complete: {len(GRAPH_SIZES)} graphs x {len(VARIANTS)} variants "
          f"in {total / 60:.1f} min")
    print(f"figures in: {os.path.normpath(FIGURES_DIR)}/k{{m}}x{{m}}/")
    print(f"{'=' * 72}", flush=True)


if __name__ == "__main__":
    main()
