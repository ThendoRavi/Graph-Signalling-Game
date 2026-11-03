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
     (i, j) is the team reward when the *signallers* come from run j and the
     *guessers* come from run i. The x-axis is the signaller run, the y-axis
     the guesser run.

     **Self-play is the diagonal, and it is played, not copied.** Cell (i, i)
     pairs run i's own signallers with run i's own guessers and plays them
     against each other through exactly the same evaluation as every
     off-diagonal cell -- the only difference between a diagonal and an
     off-diagonal cell is whose signallers are plugged in. Nothing is read
     back from training. (Because the agents are greedy and deterministic, a
     run played against itself necessarily reproduces the convention it
     settled on; that is the *result* of the self-play evaluation, not a
     shortcut around it.) A high diagonal with a low off-diagonal is the
     signature of private, non-shareable conventions; a uniformly high matrix
     means the conventions are mutually intelligible.

  2. **Per-edge cross-play matrices** (one per signalling edge S_a -> G_b). The
     guesser population is taken whole from run i, but a *single* signaller
     S_a is swapped in from run j (the other signaller stays native); the cell
     is then guesser G_b's own-item accuracy. This isolates how tolerant each
     edge is to a foreign partner. Rendered as a 2x2 grid of 15x15 matrices.
     The diagonal (i = j, no actual swap) is again self-play.

Every pairing is evaluated *exactly*: the greedy agents are played through all
four equally likely K_{2,2} worlds (every possible episode, once each), so a
cell is the pairing's true expected reward with no sampling noise and every
cell is directly comparable. As a cross-check, each run's self-play is also
measured by sampling EVAL_EPISODES real episodes on a fixed item stream.

Two modes (``--mode``):

  all      runs = seeds 0 .. NUM_RUNS-1, whatever they converged to. Imperfect
           runs (R = 0.875 / 0.75) are included, so self-play is below 1.0 on
           some diagonal cells. This is the "typical training outcome" view.
  perfect  seeds are scanned from 0 upward and only populations whose
           self-play is exactly R = 1.0 are kept, until NUM_RUNS are
           collected. Every diagonal cell is then 1.0, so any off-diagonal
           drop is purely a convention mismatch between two *perfect* codes,
           not a run that never learned. The log also compares the observed
           off-diagonal values with the prediction for runs drawn uniformly
           from the 8 tracking conventions (1.0 w.p. 1/8, 0.5 w.p. 3/4,
           0.0 w.p. 1/8; mean exactly 0.5).

The cross-play gap Delta = (mean self-play) - (mean off-diagonal) is printed per
variant in both modes.

Run:  python experiments/crossplay_k22.py --mode all
      python experiments/crossplay_k22.py --mode perfect
"""

from __future__ import annotations

import argparse
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
NUM_RUNS = 15              # populations per variant (axis = run 1..15)
TRAIN_EPISODES = 4_000
EPSILON_DECAY = 2_500
EVAL_EPISODES = 400        # sampled self-play cross-check, episodes per run
EVAL_SEED = 999_983        # fixed item stream for the sampled cross-check
# perfect mode: scan seeds 0 .. MAX_SEED_SCAN-1 until NUM_RUNS perfect runs found
MAX_SEED_SCAN = int(os.environ.get("GSG_CROSSPLAY_MAX_SEEDS", 100))

# Trained populations are cached here: the first run trains and saves them,
# every later run loads them instead of retraining. Delete this directory (or
# set USE_CACHE = False) to force a fresh retrain. Both modes share the cache,
# so perfect mode reuses every seed the all mode already trained.
USE_CACHE = True
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "question1", "models", "crossplay")

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "question1", "figures", "crossplay")

# Off-diagonal cross-play prediction when both runs are drawn uniformly from the
# 8 tracking conventions: same pairing (prob 1/2) and both polarities match ->
# 1.0, one matches -> 0.5, neither -> 0.0; different pairing -> 0.5.
TRACKING_PREDICTION = {1.0: 1 / 8, 0.5: 3 / 4, 0.0: 1 / 8}


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
    train it, save it, and return it. All agents returned greedy (epsilon 0).
    Returns (signallers, guessers, loaded_from_cache)."""
    ckpt = os.path.join(MODEL_DIR, variant, f"run{seed}.pt")
    if USE_CACHE and os.path.isfile(ckpt):
        sig, gue = load_population(ckpt, env.graph, env.item_space.n)
        return sig, gue, True  # load_population already returns epsilon-0 agents
    sig, gue = _train_fresh(variant, env, seed)
    if USE_CACHE:
        save_population(ckpt, variant, sig, gue)
    _set_greedy(sig, gue)
    return sig, gue, False


def play_pairing(graph, n_items, sig, gue):
    """Play signallers ``sig`` against guessers ``gue`` through every possible
    world (all item combinations, each once) and return (team reward,
    {guesser: accuracy}, outcomes). With sig and gue from the same run this
    *is* self-play; from different runs it is cross-play."""
    outcomes = exhaustive_joint_table(graph, n_items, sig, gue)
    reward = sum(o.reward for o in outcomes) / len(outcomes)
    acc = {g: sum(o.correct[g] for o in outcomes) / len(outcomes) for g in graph.guessers}
    return reward, acc, outcomes


def describe_run(graph, n_items, sig, gue):
    """Return (self_play_reward, conv_id, {signaller_id: rule_name}, outcomes)
    for one population, its signallers played against its own guessers."""
    self_reward, _acc, outcomes = play_pairing(graph, n_items, sig, gue)
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


def run_label(k, seed):
    return f"run {k + 1}" if seed == k else f"run {k + 1} (s{seed})"


def collect_runs(variant, mode, graph, n_items, t0):
    """Return (runs, seeds, skipped). In 'all' mode: seeds 0..NUM_RUNS-1. In
    'perfect' mode: scan seeds upward, keeping only populations whose self-play
    (own signallers x own guessers, all worlds) is exactly 1.0; ``skipped``
    lists (seed, self-play reward) of the rejected ones."""
    runs, seeds, skipped = [], [], []
    loaded = trained = 0
    seed_range = range(NUM_RUNS) if mode == "all" else range(MAX_SEED_SCAN)
    for seed in seed_range:
        sig, gue, from_cache = train_run(variant, make_env(seed=seed), seed)
        loaded += from_cache
        trained += not from_cache
        if mode == "perfect":
            r, _acc, _o = play_pairing(graph, n_items, sig, gue)
            if r < 1.0 - 1e-9:
                skipped.append((seed, r))
                continue
        runs.append((sig, gue))
        seeds.append(seed)
        if len(runs) == NUM_RUNS:
            break
    print(f"  {len(runs)} populations selected from seeds {seed_range.start}..{seed} "
          f"({loaded} loaded from cache, {trained} trained) ({time.time() - t0:.0f}s)", flush=True)
    if mode == "perfect":
        scanned = len(runs) + len(skipped)
        print(f"  perfect-only selection: {len(runs)}/{scanned} scanned seeds reached R = 1.0; "
              f"skipped {len(skipped)}:")
        for s, r in skipped:
            print(f"      seed {s:3d}: self-play R={r:.3f}  [{band(r)}]  -> excluded")
        if len(runs) < NUM_RUNS:
            print(f"  WARNING: only {len(runs)} perfect runs found in {MAX_SEED_SCAN} seeds "
                  f"(raise GSG_CROSSPLAY_MAX_SEEDS)")
    return runs, seeds, skipped


def log_run_details(variant, runs, seeds, graph, n_items):
    """Print, per run: its self-play reward (own signallers played against own
    guessers, exactly over all worlds, plus a sampled-episode cross-check), its
    convention (id + signaller rules), and the full enumerated play (all four
    worlds). Returns the list of conv_ids and self-play rewards."""
    print(f"\n  --- {variant}: per-run conventions and self-play ({len(runs)} runs) ---")
    print("      self-play = this run's signallers played against this run's guessers")
    conv_ids, self_rewards = [], []
    eval_env = make_env()
    for k, (sig, gue) in enumerate(runs):
        r, cid, rules, outcomes = describe_run(graph, n_items, sig, gue)
        sampled = evaluate(eval_env, sig, gue, episodes=EVAL_EPISODES, seed=EVAL_SEED).mean_reward
        conv_ids.append(cid)
        self_rewards.append(r)
        rule_str = ", ".join(f"S{s}: {rules[s]}" for s in graph.signallers)
        print(f"\n  {run_label(k, seeds[k])} (seed {seeds[k]}): self-play R={r:.3f} exact (all worlds), "
              f"{sampled:.3f} over {EVAL_EPISODES} sampled episodes  [{band(r)}]")
        print(f"      convention id {cid}  |  {rule_str}")
        for o in outcomes:  # all 4 worlds = the complete self-play
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


def log_off_diagonal_distribution(variant, pop, mode):
    """Tally the off-diagonal cell values. In perfect mode, compare them with
    the prediction for runs drawn uniformly from the 8 tracking conventions."""
    n = pop.shape[0]
    off = pop[~np.eye(n, dtype=bool)]
    print(f"\n  --- {variant}: distribution of the {off.size} off-diagonal (cross-play) cells ---")
    values = {float(v) for v in np.round(off, 4)}
    if mode == "perfect":
        values |= set(TRACKING_PREDICTION)  # show predicted values even when unobserved
    values = sorted(values, reverse=True)
    for v in values:
        c = int(np.sum(np.isclose(off, v)))
        pred = TRACKING_PREDICTION.get(float(v)) if mode == "perfect" else None
        pred_str = f"   (uniform-tracking prediction {pred:.3f})" if pred is not None else ""
        print(f"      R = {v:.3f}: {c:3d} cells = {c / off.size:.3f}  [{classify_pair(v)}]{pred_str}")
    if mode == "perfect":
        print("      prediction assumes both runs drawn uniformly from the 8 tracking conventions:")
        print("      P(1.0) = 1/8 (same convention), P(0.0) = 1/8 (same pairing, both polarities flipped),")
        print("      P(0.5) = 3/4 (one polarity flipped, or different pairing); expected mean exactly 0.500.")
        print("      Values other than 1.0 / 0.5 / 0.0 involve an XOR-type run.")


def render_matrix(ax, matrix, title, labels):
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
    ax.set_xticklabels(labels, fontsize=6, rotation=90)
    ax.set_yticklabels(labels, fontsize=6)
    ax.set_xlabel("Signaller run", fontsize=9)
    ax.set_ylabel("Guesser run", fontsize=9)
    ax.set_title(title, fontsize=10)
    return im


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--mode", choices=["all", "perfect"], default="all",
                        help="all = seeds 0..14 as trained; perfect = first 15 seeds with self-play R = 1.0")
    mode = parser.parse_args().mode
    suffix = "" if mode == "all" else "_perfect"
    mode_desc = ("all runs (seeds 0..14, imperfect included)" if mode == "all"
                 else "perfect runs only (self-play R = 1.0)")

    os.makedirs(FIG_DIR, exist_ok=True)
    t0 = time.time()
    print("=" * 72)
    print(f"Cross-play evaluation, K_2,2 -- mode: {mode} -- {mode_desc}")
    print(f"{NUM_RUNS} runs/variant, {TRAIN_EPISODES} train episodes; every pairing played "
          f"exactly over all 4 worlds")
    print("self-play = diagonal = a run's own signallers played against its own guessers")
    print("=" * 72, flush=True)

    _probe = make_env()
    graph, n_items = _probe.graph, _probe.item_space.n
    summary = []

    for variant in VARIANTS:
        print(f"\n{'#' * 72}")
        print(f"# {variant}  [{mode}]")
        print(f"{'#' * 72}", flush=True)

        runs, seeds, skipped = collect_runs(variant, mode, graph, n_items, t0)
        labels = [run_label(k, s) for k, s in enumerate(seeds)]
        conv_ids, self_rewards = log_run_details(variant, runs, seeds, graph, n_items)

        n = len(runs)
        pop = np.zeros((n, n))                       # [guesser run i][signaller run j]
        edge = {(s, g): np.zeros((n, n)) for s in (0, 1) for g in (0, 1)}

        for i in range(n):                            # guesser run (y)
            sig_i, gue_i = runs[i]
            for j in range(n):                        # signaller run (x)
                sig_j, _gue_j = runs[j]

                # both signallers from j, both guessers from i (i == j: self-play)
                pop[i, j], _acc, _o = play_pairing(graph, n_items, sig_j, gue_i)

                # swap only S0 from j (S1 native to i); gives edges S0->G0, S0->G1
                _r, accA, _o = play_pairing(graph, n_items, {0: sig_j[0], 1: sig_i[1]}, gue_i)
                edge[(0, 0)][i, j] = accA[0]
                edge[(0, 1)][i, j] = accA[1]

                # swap only S1 from j (S0 native to i); gives edges S1->G0, S1->G1
                _r, accB, _o = play_pairing(graph, n_items, {0: sig_i[0], 1: sig_j[1]}, gue_i)
                edge[(1, 0)][i, j] = accB[0]
                edge[(1, 1)][i, j] = accB[1]
        print(f"\n  {n}x{n} pairings played ({time.time() - t0:.0f}s)", flush=True)

        # --- cross-play gap ---
        diag = np.diag(pop)
        off = pop[~np.eye(n, dtype=bool)]
        self_mean, off_mean = diag.mean(), off.mean()
        n_perfect_self = int(np.sum(diag >= 1.0 - 1e-9))
        print(f"\n  --- {variant}: cross-play summary [{mode}] ---")
        print(f"      self-play (diagonal: own signallers x own guessers), per run: "
              + " ".join(f"{v:.3f}" for v in diag))
        print(f"      self-play runs at R = 1.0       : {n_perfect_self}/{n}")
        print(f"      mean self-play (diagonal)       : {self_mean:.3f}")
        print(f"      mean cross-play (off-diagonal)  : {off_mean:.3f}")
        print(f"      cross-play gap Delta            : {self_mean - off_mean:.3f}")
        print(f"      (gap near 0 = shareable convention; near self-0.5 = private code)")
        summary.append((variant, n, n_perfect_self, self_mean, off_mean, len(skipped)))

        log_off_diagonal_distribution(variant, pop, mode)

        # --- which conventions coordinate with which (and why) ---
        log_pairwise(variant, pop, conv_ids, self_rewards)

        # --- save raw matrices (+ the seeds behind each row/column) ---
        np.save(os.path.join(FIG_DIR, f"crossplay_population_{variant}{suffix}.npy"), pop)
        np.save(os.path.join(FIG_DIR, f"crossplay_seeds_{variant}{suffix}.npy"), np.array(seeds))
        for (s, g), m in edge.items():
            np.save(os.path.join(FIG_DIR, f"crossplay_edge_S{s}G{g}_{variant}{suffix}.npy"), m)

        # --- population cross-play figure ---
        fig, ax = plt.subplots(figsize=(8.5, 7.2))
        im = render_matrix(ax, pop,
                           f"{variant}: population cross-play, {mode_desc}\n"
                           f"self-play {self_mean:.3f}, cross {off_mean:.3f}, "
                           f"gap {self_mean - off_mean:.3f}", labels)
        fig.colorbar(im, ax=ax, label="team reward (exact, all worlds)", shrink=0.8)
        fig.tight_layout()
        p = os.path.join(FIG_DIR, f"crossplay_population_{variant}{suffix}.png")
        fig.savefig(p, dpi=140)
        plt.close(fig)
        print(f"      saved -> {os.path.normpath(p)}", flush=True)

        # --- per-edge figure (2x2 grid) ---
        fig, axes = plt.subplots(2, 2, figsize=(15, 13))
        im = None
        for (s, g), ax in zip(sorted(edge.keys()), axes.flat):
            im = render_matrix(ax, edge[(s, g)],
                               f"edge S{s} -> G{g}  (G{g} accuracy, foreign S{s})", labels)
        fig.suptitle(f"{variant}: per-edge cross-play, {mode_desc}\n"
                     f"(single foreign signaller swapped in, guesser accuracy)", fontsize=12)
        fig.subplots_adjust(right=0.9, hspace=0.35, wspace=0.3)
        cbar_ax = fig.add_axes((0.92, 0.25, 0.015, 0.5))
        fig.colorbar(im, cax=cbar_ax, label="guesser accuracy")
        p = os.path.join(FIG_DIR, f"crossplay_edges_{variant}{suffix}.png")
        fig.savefig(p, dpi=130)
        plt.close(fig)
        print(f"      saved -> {os.path.normpath(p)}", flush=True)

    print(f"\n{'=' * 72}")
    print(f"SUMMARY [{mode}: {mode_desc}]")
    print(f"  {'variant':<12} {'runs':>4} {'self=1.0':>8} {'self-play':>9} {'cross':>7} {'gap':>7}"
          + (f" {'seeds skipped':>13}" if mode == "perfect" else ""))
    for variant, n, n_perf, s_mean, o_mean, n_skip in summary:
        print(f"  {variant:<12} {n:>4} {n_perf:>5}/{n:<2} {s_mean:>9.3f} {o_mean:>7.3f} "
              f"{s_mean - o_mean:>7.3f}" + (f" {n_skip:>13}" if mode == "perfect" else ""))
    print(f"\ncrossplay_k22.py --mode {mode} complete in {(time.time() - t0) / 60:.1f} min")
    print(f"figures in: {os.path.normpath(FIG_DIR)}  (suffix '{suffix or 'none'}')")
    print("=" * 72, flush=True)


if __name__ == "__main__":
    main()
