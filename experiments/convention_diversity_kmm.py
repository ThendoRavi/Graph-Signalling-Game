"""Convention determination on K_{m,m}, m in {3,4,5,6}: longer training, with
the same mathematical-framework analysis used for K_{2,2}.

This is the batch job meant to run on the SSH / SLURM server (see
``conventions_kmm.sh``). It extends the K_{2,2} convention analysis
(convention_diversity_k22.py) to the larger complete-bipartite graphs, with
two changes appropriate to scale:

  1. **Longer training.** Each graph size gets a substantially longer episode
     budget than the earlier scaling probe (Table in the results chapter), so
     the larger populations have a real chance to settle before being read.

  2. **Structural (Boolean-function) convention analysis instead of an
     exhaustive catalogue.** For K_{2,2} we could enumerate all 24 perfect
     conventions. For K_{m,m} the perfect-convention space is (2^m)! -- 40320
     already at m=3, ~10^63 at m=6 -- so it cannot be catalogued. Instead each
     learned signaller policy is a Boolean function f: {0,1}^m -> {0,1}, and
     is classified structurally (:mod:`gsg.evaluation.boolean_analysis`):
     which guessers' items it depends on (its support), and whether it is a
     constant (pooling), a dictator (a clean one-item relay = "tracking", the
     compositional case), a parity (XOR of several items = holistic), or a
     general nonlinear rule. This is the exact m-item generalisation of the
     "track x0 / XOR / pooling" labels used for K_{2,2}.

For every seed and variant the log reports, exactly (via exhaustive
enumeration over the 2^m equally-likely worlds -- 64 worlds at m=6, cheap):

  * the greedy team reward and its convergence band;
  * a per-signaller Boolean classification (what each light encodes);
  * a whole-convention summary (perfect pure-tracking / perfect holistic /
    partial), and the exact joint convention id;
  * the information-theoretic profile (channel utilisation, per-edge mutual
    information, per-guesser entropy reduction);
  * 5 example greedy episodes, so the reader can verify the convention by eye.

Across seeds it reports the distinct-perfect-convention count and the
distribution of convention *types*.

Run:  python experiments/convention_diversity_kmm.py
"""

from __future__ import annotations

import os
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from gsg.environment.graphs import SignallingGraph
from gsg.environment.pettingzoo_env import GraphSignallingParallelEnv
from gsg.evaluation.boolean_analysis import (
    analyse_signaller_policy,
    classify_convention,
    theoretical_convention_counts,
)
from gsg.evaluation.conventions import (
    convention_summary,
    exhaustive_joint_table,
    extract_all_policies,
    joint_signaller_convention_id,
)
from gsg.evaluation.information_theory import (
    entropy_bits,
    guesser_info_metrics,
    signaller_info_metrics,
)
from gsg.training.trainer import (
    play_greedy_example,
    train_independent_iql,
    train_shared_iql,
    train_two_brain_iql,
)

# --- configuration -----------------------------------------------------

GRAPH_SIZES = [3, 4, 5, 6]

# Longer than the earlier single-seed scaling probe (which used
# 5k/7k/9k/12k). Roughly doubled, to give the larger populations a real
# chance to settle. (num_training_episodes, epsilon_decay_episodes) per m.
SCHEDULE = {
    3: (12_000, 8_000),
    4: (16_000, 11_000),
    5: (20_000, 14_000),
    6: (24_000, 16_000),
}

VARIANTS = ["shared", "two_brain", "independent"]

# P=30, matching the K_2,2 protocol, for a full convention-type distribution
# per size. This is the slow part at scale (K_6,6 independent at 24k episodes
# is a few minutes per seed), but well inside the 1-day SLURM budget.
NUM_SEEDS = 30

NUM_EXAMPLE_EPISODES = 5


def make_env(m: int, seed=None) -> GraphSignallingParallelEnv:
    return GraphSignallingParallelEnv(SignallingGraph.complete_bipartite(m, m), seed=seed)


def _train(variant: str, env, num_episodes: int, decay: int, seed):
    if variant == "shared":
        _net, sig, gue, _h = train_shared_iql(
            env, num_episodes=num_episodes, epsilon_decay_episodes=decay, seed=seed)
    elif variant == "two_brain":
        _s, _g, sig, gue, _h = train_two_brain_iql(
            env, num_episodes=num_episodes, epsilon_decay_episodes=decay, seed=seed)
    elif variant == "independent":
        _s, _g, sig, gue, _h = train_independent_iql(
            env, num_episodes=num_episodes, epsilon_decay_episodes=decay, seed=seed)
    else:
        raise ValueError(f"unknown variant {variant!r}")
    return sig, gue


ITEM_NAME = {0: "CAT", 1: "DOG"}
SIGNAL_NAME = {0: "OFF", 1: "ON"}


def format_episode(graph, record) -> str:
    items = ", ".join(f"G{g}={ITEM_NAME[record.items[g]]}" for g in graph.guessers)
    signals = ", ".join(f"S{s}={SIGNAL_NAME[record.signals[s]]}" for s in graph.signallers)
    guesses = ", ".join(
        f"G{g}={ITEM_NAME[record.guesses[g]]}{'ok' if record.correct[g] else 'XX'}"
        for g in graph.guessers)
    return f"items({items}) -> lights({signals}) -> guesses({guesses})  R={record.reward:.2f}"


def analyse_seed(env, sig, gue) -> dict:
    """Full mathematical-framework analysis of one trained population."""
    graph = env.graph
    n_items = env.item_space.n
    outcomes = exhaustive_joint_table(graph, n_items, sig, gue)
    mean_reward = sum(o.reward for o in outcomes) / len(outcomes)

    policies = extract_all_policies(graph, n_items, sig, gue)
    functions = {
        s: analyse_signaller_policy(policies[f"S{s}"], graph.out_neighbours(s))
        for s in graph.signallers
    }
    return {
        "mean_reward": mean_reward,
        "is_perfect": mean_reward == 1.0,
        "band": convention_summary(outcomes),
        "conv_id": joint_signaller_convention_id(graph, n_items, sig),
        "functions": functions,
        "sig_info": signaller_info_metrics(graph, outcomes),
        "gue_info": guesser_info_metrics(graph, outcomes),
    }


def main() -> None:
    overall = time.time()
    print("=" * 72)
    print("Convention determination on K_{m,m} (longer training + Boolean framework)")
    print(f"graphs   : {', '.join(f'K{m}x{m}' for m in GRAPH_SIZES)}")
    print(f"variants : {', '.join(VARIANTS)}")
    print(f"seeds    : {NUM_SEEDS} per (graph, variant)")
    print("=" * 72, flush=True)

    for m in GRAPH_SIZES:
        num_episodes, decay = SCHEDULE[m]
        tag = f"K{m}x{m}"
        counts = theoretical_convention_counts(m)
        print(f"\n{'#' * 72}")
        print(f"# {tag}  ({m} signallers, {m} guessers, {m * m} edges)")
        print(f"#   schedule: {num_episodes} episodes, epsilon decay over {decay}")
        print(f"#   convention space: {counts['perfect_conventions']:,} perfect "
              f"((2^{m})! bijections), of which {counts['tracking_conventions']} "
              f"are pure-tracking (m!*2^m)")
        print(f"{'#' * 72}", flush=True)

        for variant in VARIANTS:
            print(f"\n{'=' * 60}")
            print(f"  {tag} / {variant}")
            print(f"{'=' * 60}", flush=True)

            results = []
            for seed in range(NUM_SEEDS):
                t0 = time.time()
                env = make_env(m, seed=seed)
                sig, gue = _train(variant, env, num_episodes, decay, seed=seed)
                r = analyse_seed(env, sig, gue)
                results.append(r)

                conv_type = classify_convention(r["functions"], m, r["is_perfect"])
                print(f"\n  seed {seed:2d}  R={r['mean_reward']:.3f}  [{r['band']}]  "
                      f"({time.time() - t0:.1f}s)", flush=True)
                print(f"           {conv_type}")
                for s in env.graph.signallers:
                    f = r["functions"][s]
                    print(f"             S{s}: {f.name:32s} (kind={f.kind}, "
                          f"depends on {list(f.support) if f.support else 'nothing'})")
                # per-guesser: is it served? (entropy reduction near 1 = yes)
                served = ", ".join(
                    f"G{g}={r['gue_info'][g].entropy_reduction:.2f}"
                    for g in env.graph.guessers)
                print(f"             entropy reduction per guesser (out of 1.0): {served}")
                for i, rec in enumerate(
                        [play_greedy_example(env, sig, gue) for _ in range(NUM_EXAMPLE_EPISODES)],
                        start=1):
                    print(f"             ex{i}: {format_episode(env.graph, rec)}")

            # --- aggregate over seeds ---
            perfect = [r for r in results if r["is_perfect"]]
            print(f"\n  --- {tag} / {variant}: summary over {NUM_SEEDS} seeds ---")
            print(f"      mean team reward           : "
                  f"{sum(r['mean_reward'] for r in results) / len(results):.3f}")
            print(f"      complementarity rate       : "
                  f"{100 * len(perfect) / len(results):.1f}% of seeds perfect")

            if perfect:
                pcounter = Counter(r["conv_id"] for r in perfect)
                H = entropy_bits(pcounter, len(perfect))
                print(f"      distinct PERFECT conventions: {len(pcounter)} "
                      f"(from {len(perfect)} perfect seeds; entropy {H:.3f} bits)")

            # convention-type distribution (structural)
            type_counter = Counter()
            for r in results:
                kinds = [f.kind for f in r["functions"].values()]
                if r["is_perfect"] and all(k == "dictator" for k in kinds):
                    type_counter["perfect pure-tracking"] += 1
                elif r["is_perfect"]:
                    type_counter["perfect holistic"] += 1
                else:
                    type_counter["partial / non-perfect"] += 1
            print("      convention types:")
            for t, c in type_counter.most_common():
                print(f"        {t}: {c}/{len(results)}")

            # signaller-kind census (how compositional are the lights overall?)
            kind_census = Counter(
                f.kind for r in results for f in r["functions"].values())
            total_sigs = sum(kind_census.values())
            print("      signaller-light census (all seeds pooled): " + ", ".join(
                f"{k}={v} ({100 * v / total_sigs:.0f}%)" for k, v in kind_census.most_common()))

            # mean information-theoretic metrics
            n = len(results)
            util = sum(r["sig_info"][s].channel_utilisation
                       for r in results for s in env.graph.signallers) / (n * m)
            er = sum(r["gue_info"][g].entropy_reduction
                     for r in results for g in env.graph.guessers) / (n * m)
            print(f"      mean channel utilisation H(M) : {util:.3f} bits (out of 1.0)")
            print(f"      mean entropy reduction / guesser: {er:.3f} bits (out of 1.0)", flush=True)

    print(f"\n{'=' * 72}")
    print(f"convention_diversity_kmm.py complete in {(time.time() - overall) / 60:.1f} min")
    print("=" * 72, flush=True)


if __name__ == "__main__":
    main()
