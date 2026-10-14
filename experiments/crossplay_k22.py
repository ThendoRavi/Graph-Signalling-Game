"""Cross-play evaluation on K_{2,2} (Section~sec:baseline-crossplay).

Cross-play measures convention *shareability*: an agent population that learned
a genuine, public convention should still coordinate when paired with an
*independently trained* population; one that learned a private co-adapted code
should fail. This is the K_{2,2} analogue of the cross-play matrices in the
zero-shot-coordination / Hanabi literature.

For each variant (shared-brain, two-brain, independent), NUM_RUNS independent
populations are trained from distinct seeds, each a full set {S0, S1, G0, G1}.
Two families of figure are produced.

  1. **Population cross-play matrix** (the direct Hanabi-style analogue). Cell
     (i, j) is the mean team reward when the *signallers* come from run j and
     the *guessers* come from run i. The x-axis is the signaller run, the
     y-axis the guesser run; the diagonal is self-play. A high diagonal with a
     low off-diagonal is the signature of private, non-shareable conventions;
     a uniformly high matrix means the conventions are mutually intelligible.

  2. **Per-edge cross-play matrices** (one per signalling edge S_a -> G_b). The
     guesser population is taken whole from run i, but a *single* signaller
     S_a is swapped in from run j (the other signaller stays native); the cell
     is then guesser G_b's own-item accuracy. This isolates how tolerant each
     edge is to a foreign partner. Rendered as a 2x2 grid of 15x15 matrices.
     The diagonal (i = j, no actual swap) is again self-play.

Runs are labelled 1..NUM_RUNS. Every pairing is evaluated on the *same* fixed
item stream (EVAL_SEED), so all cells are directly comparable. The cross-play
gap Delta = (mean self-play) - (mean off-diagonal) is printed per variant.

Run:  python experiments/crossplay_k22.py
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from gsg.environment.graphs import SignallingGraph
from gsg.environment.pettingzoo_env import GraphSignallingParallelEnv
from gsg.evaluation.metrics import evaluate
from gsg.training.trainer import train_independent_iql, train_shared_iql, train_two_brain_iql

# --- configuration -----------------------------------------------------

VARIANTS = ["shared", "two_brain", "independent"]
NUM_RUNS = 15              # independent populations per variant (axis = run 1..15)
TRAIN_EPISODES = 4_000
EPSILON_DECAY = 2_500
EVAL_EPISODES = 400        # episodes per pairing (cross-play is forward-only, cheap)
EVAL_SEED = 999_983        # fixed item stream shared by every pairing, for fairness

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "question1", "figures", "crossplay")


def make_env(seed=None) -> GraphSignallingParallelEnv:
    return GraphSignallingParallelEnv(SignallingGraph.complete_bipartite(2, 2), seed=seed)


def _set_greedy(signaller_agents, guesser_agents):
    for a in list(signaller_agents.values()) + list(guesser_agents.values()):
        if hasattr(a, "epsilon"):
            a.epsilon = 0.0
        if hasattr(a, "recording"):
            a.recording = False


def train_run(variant, env, seed):
    if variant == "shared":
        _n, sig, gue, _h = train_shared_iql(
            env, num_episodes=TRAIN_EPISODES, epsilon_decay_episodes=EPSILON_DECAY, seed=seed)
    elif variant == "two_brain":
        _s, _g, sig, gue, _h = train_two_brain_iql(
            env, num_episodes=TRAIN_EPISODES, epsilon_decay_episodes=EPSILON_DECAY, seed=seed)
    elif variant == "independent":
        _s, _g, sig, gue, _h = train_independent_iql(
            env, num_episodes=TRAIN_EPISODES, epsilon_decay_episodes=EPSILON_DECAY, seed=seed)
    else:
        raise ValueError(variant)
    _set_greedy(sig, gue)
    return sig, gue


def render_matrix(ax, matrix, title):
    """One 15x15 cross-play heatmap: x = signaller run, y = guesser run."""
    n = matrix.shape[0]
    im = ax.imshow(matrix, cmap="viridis", vmin=0.0, vmax=1.0, aspect="equal")
    for i in range(n):
        for j in range(n):
            v = matrix[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=5.5, color="white" if v < 0.62 else "black")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([f"run {k + 1}" for k in range(n)], fontsize=6, rotation=90)
    ax.set_yticklabels([f"run {k + 1}" for k in range(n)], fontsize=6)
    ax.set_xlabel("Signaller run", fontsize=9)
    ax.set_ylabel("Guesser run", fontsize=9)
    ax.set_title(title, fontsize=10)
    return im


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    t0 = time.time()
    print("=" * 72)
    print("Cross-play evaluation, K_2,2")
    print(f"{NUM_RUNS} runs/variant, {TRAIN_EPISODES} train episodes, "
          f"{EVAL_EPISODES} eval episodes/pairing")
    print("=" * 72, flush=True)

    for variant in VARIANTS:
        print(f"\n{'#' * 72}")
        print(f"# {variant}")
        print(f"{'#' * 72}", flush=True)

        # --- train NUM_RUNS populations ---
        runs = []
        for seed in range(NUM_RUNS):
            env = make_env(seed=seed)
            runs.append(train_run(variant, env, seed))
        print(f"  trained {NUM_RUNS} populations ({time.time() - t0:.0f}s)", flush=True)

        n = NUM_RUNS
        pop = np.zeros((n, n))                       # [guesser run i][signaller run j]
        edge = {(s, g): np.zeros((n, n)) for s in (0, 1) for g in (0, 1)}
        eval_env = make_env()

        for i in range(n):                            # guesser run (y)
            sig_i, gue_i = runs[i]
            for j in range(n):                        # signaller run (x)
                sig_j, _gue_j = runs[j]

                # full cross: both signallers from j, both guessers from i
                res = evaluate(eval_env, sig_j, gue_i, episodes=EVAL_EPISODES, seed=EVAL_SEED)
                pop[i, j] = res.mean_reward

                # swap only S0 from j (S1 native to i); gives edges S0->G0, S0->G1
                sig_swapA = {0: sig_j[0], 1: sig_i[1]}
                resA = evaluate(eval_env, sig_swapA, gue_i, episodes=EVAL_EPISODES, seed=EVAL_SEED)
                edge[(0, 0)][i, j] = resA.per_guesser_accuracy[0]
                edge[(0, 1)][i, j] = resA.per_guesser_accuracy[1]

                # swap only S1 from j (S0 native to i); gives edges S1->G0, S1->G1
                sig_swapB = {0: sig_i[0], 1: sig_j[1]}
                resB = evaluate(eval_env, sig_swapB, gue_i, episodes=EVAL_EPISODES, seed=EVAL_SEED)
                edge[(1, 0)][i, j] = resB.per_guesser_accuracy[0]
                edge[(1, 1)][i, j] = resB.per_guesser_accuracy[1]
            print(f"    guesser run {i + 1}/{n} paired against all signaller runs "
                  f"({time.time() - t0:.0f}s)", flush=True)

        # --- cross-play gap ---
        diag = np.diag(pop)
        off = pop[~np.eye(n, dtype=bool)]
        self_mean, off_mean = diag.mean(), off.mean()
        print(f"\n  --- {variant}: cross-play summary ---")
        print(f"      mean self-play (diagonal)   : {self_mean:.3f}")
        print(f"      mean cross-play (off-diag)  : {off_mean:.3f}")
        print(f"      cross-play gap Delta        : {self_mean - off_mean:.3f}")
        print(f"      (gap near 0 = shareable convention; near self-0.5 = private code)")

        # --- save raw matrices ---
        np.save(os.path.join(FIG_DIR, f"crossplay_population_{variant}.npy"), pop)
        for (s, g), m in edge.items():
            np.save(os.path.join(FIG_DIR, f"crossplay_edge_S{s}G{g}_{variant}.npy"), m)

        # --- population cross-play figure ---
        fig, ax = plt.subplots(figsize=(8.5, 7.2))
        im = render_matrix(ax, pop,
                           f"{variant}: population cross-play (mean team reward)\n"
                           f"self-play {self_mean:.3f}, cross {off_mean:.3f}, "
                           f"gap {self_mean - off_mean:.3f}")
        fig.colorbar(im, ax=ax, label="mean team reward", shrink=0.8)
        fig.tight_layout()
        p = os.path.join(FIG_DIR, f"crossplay_population_{variant}.png")
        fig.savefig(p, dpi=140)
        plt.close(fig)
        print(f"      saved -> {os.path.normpath(p)}", flush=True)

        # --- per-edge figure (2x2 grid) ---
        fig, axes = plt.subplots(2, 2, figsize=(15, 13))
        im = None
        for (s, g), ax in zip(sorted(edge.keys()), axes.flat):
            im = render_matrix(ax, edge[(s, g)],
                               f"edge S{s} -> G{g}  (G{g} accuracy, foreign S{s})")
        fig.suptitle(f"{variant}: per-edge cross-play "
                     f"(single foreign signaller swapped in, guesser accuracy)", fontsize=12)
        fig.subplots_adjust(right=0.9, hspace=0.35, wspace=0.3)
        cbar_ax = fig.add_axes((0.92, 0.25, 0.015, 0.5))
        fig.colorbar(im, cax=cbar_ax, label="guesser accuracy")
        p = os.path.join(FIG_DIR, f"crossplay_edges_{variant}.png")
        fig.savefig(p, dpi=130)
        plt.close(fig)
        print(f"      saved -> {os.path.normpath(p)}", flush=True)

    print(f"\n{'=' * 72}")
    print(f"crossplay_k22.py complete in {(time.time() - t0) / 60:.1f} min")
    print(f"figures in: {os.path.normpath(FIG_DIR)}")
    print("=" * 72, flush=True)


if __name__ == "__main__":
    main()
