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
from gsg.evaluation.boolean_analysis import analyse_signaller_policy
from gsg.evaluation.conventions import (
    exhaustive_joint_table,
    extract_all_policies,
    joint_signaller_convention_id,
)
from gsg.evaluation.metrics import evaluate
from gsg.training.checkpoint import load_population, save_population
from gsg.training.trainer import train_independent_iql, train_shared_iql, train_two_brain_iql

ITEM_NAME = {0: "CAT", 1: "DOG"}
SIGNAL_NAME = {0: "OFF", 1: "ON"}

# --- configuration -----------------------------------------------------

VARIANTS = ["shared", "two_brain", "independent"]
NUM_RUNS = 15              # independent populations per variant (axis = run 1..15)
TRAIN_EPISODES = 4_000
EPSILON_DECAY = 2_500
EVAL_EPISODES = 400        # episodes per pairing (cross-play is forward-only, cheap)
EVAL_SEED = 999_983        # fixed item stream shared by every pairing, for fairness

# Trained populations are cached here: the first run trains and saves them,
# every later run loads them instead of retraining. Delete this directory (or
# set USE_CACHE = False) to force a fresh retrain.
USE_CACHE = True
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "question1", "models", "crossplay")

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "question1", "figures", "crossplay")


def make_env(seed=None) -> GraphSignallingParallelEnv:
    return GraphSignallingParallelEnv(SignallingGraph.complete_bipartite(2, 2), seed=seed)


def _set_greedy(signaller_agents, guesser_agents):
    for a in list(signaller_agents.values()) + list(guesser_agents.values()):
        if hasattr(a, "epsilon"):
            a.epsilon = 0.0
        if hasattr(a, "recording"):
            a.recording = False


def _train_fresh(variant, env, seed):
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
    return sig, gue


def train_run(variant, env, seed):
    """Load the cached population for (variant, seed) if present; otherwise
    train it, save it, and return it. All agents returned greedy (epsilon 0)."""
    ckpt = os.path.join(MODEL_DIR, variant, f"run{seed}.pt")
    if USE_CACHE and os.path.isfile(ckpt):
        sig, gue = load_population(ckpt, env.graph, env.item_space.n)
        return sig, gue  # load_population already returns epsilon-0 agents
    sig, gue = _train_fresh(variant, env, seed)
    if USE_CACHE:
        save_population(ckpt, variant, sig, gue)
    _set_greedy(sig, gue)
    return sig, gue


def describe_run(graph, n_items, sig, gue):
    """Return (self_reward, conv_id, {signaller_id: rule_name}, outcomes) for one
    trained population, computed exactly by exhaustive enumeration."""
    outcomes = exhaustive_joint_table(graph, n_items, sig, gue)
    self_reward = sum(o.reward for o in outcomes) / len(outcomes)
    conv_id = joint_signaller_convention_id(graph, n_items, sig)
    policies = extract_all_policies(graph, n_items, sig, gue)
    rules = {
        s: analyse_signaller_policy(policies[f"S{s}"], graph.out_neighbours(s)).name
        for s in graph.signallers
    }
    return self_reward, conv_id, rules, outcomes


def format_world(graph, o):
    """One line for one enumerated world outcome (items -> lights -> guesses)."""
    items = ", ".join(f"G{g}={ITEM_NAME[o.items[g]]}" for g in graph.guessers)
    lights = ", ".join(f"S{s}={SIGNAL_NAME[o.signals[s]]}" for s in graph.signallers)
    guesses = ", ".join(
        f"G{g}={ITEM_NAME[o.guesses[g]]}{'ok' if o.correct[g] else 'XX'}"
        for g in graph.guessers
    )
    return f"items({items}) -> lights({lights}) -> guesses({guesses})  R={o.reward:.2f}"


def band(reward):
    if reward >= 0.999:
        return "PERFECT"
    if abs(reward - 0.875) < 0.02:
        return "exploiting joint channel (~0.875)"
    if abs(reward - 0.75) < 0.02:
        return "single-bit ceiling (~0.75)"
    if abs(reward - 0.5) < 0.02:
        return "at floor (~0.50)"
    return f"partial ({reward:.3f})"


def log_run_details(variant, runs, graph, n_items):
    """Print, per run: self-play reward, its convention (id + signaller rules),
    and the full enumerated rule (all four worlds). Returns the list of
    conv_ids and self-rewards, one per run, for the pairwise analysis."""
    print(f"\n  --- {variant}: per-run conventions ({len(runs)} runs) ---")
    conv_ids, self_rewards = [], []
    for k, (sig, gue) in enumerate(runs):
        r, cid, rules, outcomes = describe_run(graph, n_items, sig, gue)
        conv_ids.append(cid)
        self_rewards.append(r)
        rule_str = ", ".join(f"S{s}: {rules[s]}" for s in graph.signallers)
        print(f"\n  run {k + 1:2d} (seed {k}): self-play R={r:.3f}  [{band(r)}]")
        print(f"      convention id {cid}  |  {rule_str}")
        for o in outcomes:  # all 4 worlds = the complete rule
            print(f"        {format_world(graph, o)}")
    return conv_ids, self_rewards


def classify_pair(reward):
    if reward >= 0.95:
        return "COMPATIBLE"
    if reward <= 0.05:
        return "ANTI-COMPATIBLE"
    if abs(reward - 0.5) < 0.06:
        return "ORTHOGONAL"
    return "PARTIAL"


def log_pairwise(variant, pop, conv_ids, self_rewards):
    """Print the convention equivalence classes and a convention-vs-convention
    cross-play summary, so which conventions coordinate (and which clash) is
    explicit."""
    n = len(conv_ids)

    # group runs that found the identical joint signaller convention
    classes = {}
    for k, cid in enumerate(conv_ids):
        classes.setdefault(cid, []).append(k + 1)
    print(f"\n  --- {variant}: convention equivalence classes ---")
    for cid, members in sorted(classes.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        rep = members[0] - 1
        print(f"      conv {cid}: runs {members}  (self-play R={self_rewards[rep]:.3f})")

    # convention-vs-convention cross reward (mean over member pairings),
    # row = guesser-run convention, col = signaller-run convention
    cids = sorted(classes.keys())
    print(f"\n  --- {variant}: cross-play by convention "
          f"(guesser conv x signaller conv -> mean reward) ---")
    for ci in cids:
        for cj in cids:
            gi = [m - 1 for m in classes[ci]]
            sj = [m - 1 for m in classes[cj]]
            vals = [pop[i, j] for i in gi for j in sj if not (ci == cj and i == j)]
            if not vals:
                vals = [pop[gi[0], sj[0]]]  # singleton self-class: use the self cell
            mean = sum(vals) / len(vals)
            tag = "self " if ci == cj else classify_pair(mean)
            print(f"      guesser {ci}  x  signaller {cj}  ->  R={mean:.3f}  [{tag}]")

    # a few representative concrete pairings per outcome category
    print(f"\n  --- {variant}: example pairings by outcome ---")
    seen = {"COMPATIBLE": 0, "ANTI-COMPATIBLE": 0, "ORTHOGONAL": 0, "PARTIAL": 0}
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            cat = classify_pair(pop[i, j])
            if seen[cat] < 4:
                seen[cat] += 1
                print(f"      guesser run {i + 1:2d} (conv {conv_ids[i]})  x  "
                      f"signaller run {j + 1:2d} (conv {conv_ids[j]})  ->  "
                      f"R={pop[i, j]:.2f}  [{cat}]")


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

        # --- obtain NUM_RUNS populations (load from cache or train+save) ---
        cached = sum(
            1 for seed in range(NUM_RUNS)
            if USE_CACHE and os.path.isfile(os.path.join(MODEL_DIR, variant, f"run{seed}.pt"))
        )
        runs = []
        for seed in range(NUM_RUNS):
            env = make_env(seed=seed)
            runs.append(train_run(variant, env, seed))
        print(f"  {NUM_RUNS} populations ready ({cached} loaded from cache, "
              f"{NUM_RUNS - cached} trained) ({time.time() - t0:.0f}s)", flush=True)

        # --- log each run's convention (reward, id, signaller rules, full rule) ---
        _probe = make_env()
        graph, n_items = _probe.graph, _probe.item_space.n
        conv_ids, self_rewards = log_run_details(variant, runs, graph, n_items)

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

        # --- which conventions coordinate with which (and why) ---
        log_pairwise(variant, pop, conv_ids, self_rewards)

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
