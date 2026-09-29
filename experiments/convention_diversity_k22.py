"""Convention diversity on K_{2,2}: what conventions emerge, how consistently,
why, and what they cost/gain information-theoretically.

This is the batch job meant to run on the SSH / SLURM server (see
``conventions.sh``). It has two phases.

Phase 1 -- Convention-quality and information-theoretic metrics
-----------------------------------------------------------------
The outstanding O1.4 / information-theoretic work items from the results
chapter, run for real: for each of the three parameter-sharing variants
(shared-brain, two-brain, independent), P = 30 independently-seeded
populations are trained on K_{2,2} (matching compare_iql_variants_k22.py's
protocol exactly, so these numbers are directly comparable to
Table~tab:q1-variant-comparison). For each converged population:

  * its *joint signaller convention* is identified exactly
    (:func:`gsg.evaluation.conventions.joint_signaller_convention_id`) --
    not a reward number, but which of the literal possible signaller
    mappings the run landed on, for every signaller;
  * its information-theoretic profile is computed exactly, via exhaustive
    enumeration (:mod:`gsg.evaluation.information_theory`): channel
    utilisation and per-edge mutual information for every signaller, entropy
    reduction for every guesser.

Across the 30 seeds this gives the empirical *joint convention distribution*
-- literally "run it a bunch of times and see what different conventions it
lands on" -- plus convention entropy, dominant-convention frequency, and
complementarity rate (Section~sec:eval-criteria's convention-quality group).

Phase 2 -- What actually decides which convention a run finds?
-----------------------------------------------------------------
K_{2,2} is symmetric under swapping S0<->S1 and/or G0<->G1: every
convention has a "mirror" of exactly equal expected reward, so nothing in
the *topology itself* singles one out (see the module docstring further
down, and the chapter's Section~sec:results-scaling-crosscutting-style
discussion). What actually breaks that tie, run to run? This phase isolates
the three independent sources of randomness a training run draws on --
network initialisation (torch), exploration/replay-buffer sampling
(Python's `random` module), and the environment's hidden-item sampling order
(the env's own seeded RNG) -- and varies *each one alone*, holding the other
two fixed, to see which one actually moves the outcome. This is possible
without touching trainer.py at all: passing ``seed=None`` to a trainer skips
its internal seeding entirely, so the three sources can be seeded by hand,
independently, right here.

Run:  python experiments/convention_diversity_k22.py
"""

from __future__ import annotations

import os
import random
import sys
import time
from collections import Counter

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from gsg.environment.graphs import SignallingGraph
from gsg.environment.pettingzoo_env import GraphSignallingParallelEnv
from gsg.evaluation.conventions import (
    convention_summary,
    exhaustive_joint_table,
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

VARIANTS = ["shared", "two_brain", "independent"]

# Phase 1: matches compare_iql_variants_k22.py's protocol exactly.
NUM_SEEDS = 30
NUM_EPISODES = 4_000
EPSILON_DECAY_EPISODES = 2_500

# Phase 2: same schedule, fewer repeats per condition (this is a diagnostic
# probe of *what* drives the outcome, not a convergence-rate estimate, so it
# doesn't need P=30 per condition).
ABLATION_REPEATS = 12

GRAPH = lambda: SignallingGraph.complete_bipartite(2, 2)


def make_env(seed=None) -> GraphSignallingParallelEnv:
    return GraphSignallingParallelEnv(GRAPH(), seed=seed)


# --- K_2,2 convention catalogue (matches the 24 perfect conventions listed
#     in the Results chapter) ------------------------------------------------
# A single signaller's canonical id (0..15) is exactly
# conventions.canonical_id(policy, action_range=2, value_range=2,
# neighbourhood_size=2) -- i.e. digit 2^k over observations in the order
# (0,0),(0,1),(1,0),(1,1). This block turns that id, and a joint (S0,S1) pair,
# into a plain-English description, and numbers the 24 perfect conventions
# 1..24 in the same order as the thesis table.
ITEM_NAME = {0: "CAT", 1: "DOG"}
SIGNAL_NAME = {0: "OFF", 1: "ON"}
_INPUTS = [(0, 0), (0, 1), (1, 0), (1, 1)]  # (x0, x1)


def _policy_from_id(cid: int):
    return tuple((cid >> k) & 1 for k in range(4))


def _emit(policy, x0, x1) -> int:
    return policy[_INPUTS.index((x0, x1))]


def signaller_rule_name(cid: int) -> str:
    """Plain-English name for one signaller's canonical id, K_2,2."""
    outputs = tuple(_emit(_policy_from_id(cid), x0, x1) for (x0, x1) in _INPUTS)
    names = {
        (0, 0, 1, 1): "track x0 (ON iff G0=DOG)",
        (1, 1, 0, 0): "track NOT x0 (ON iff G0=CAT)",
        (0, 1, 0, 1): "track x1 (ON iff G1=DOG)",
        (1, 0, 1, 0): "track NOT x1 (ON iff G1=CAT)",
        (0, 1, 1, 0): "XOR (ON iff items DIFFER)",
        (1, 0, 0, 1): "XNOR (ON iff items SAME)",
        (0, 0, 0, 0): "constant OFF (pooling)",
        (1, 1, 1, 1): "constant ON (pooling)",
    }
    return names.get(outputs, f"other[{''.join(map(str, outputs))}]")


def _best_reward(id_a: int, id_b: int) -> float:
    """Team reward of signaller pair (id_a, id_b) with Bayes-optimal guessers."""
    from collections import defaultdict
    pa, pb = _policy_from_id(id_a), _policy_from_id(id_b)
    buckets = defaultdict(list)
    for (x0, x1) in _INPUTS:
        buckets[(_emit(pa, x0, x1), _emit(pb, x0, x1))].append((x0, x1))
    total = 0.0
    for (x0, x1) in _INPUTS:
        worlds = buckets[(_emit(pa, x0, x1), _emit(pb, x0, x1))]
        g0 = round(sum(w[0] for w in worlds) / len(worlds))
        g1 = round(sum(w[1] for w in worlds) / len(worlds))
        total += ((g0 == x0) + (g1 == x1)) / 2
    return total / len(_INPUTS)


# {(id0, id1): (number 1..24, "tracking"|"XOR-type")} for the 24 perfect
# conventions, numbered in sorted-pair order to match the thesis table.
_PERFECT_CATALOGUE = {}
for _num, (_a, _b) in enumerate(
    sorted((a, b) for a in range(16) for b in range(16) if _best_reward(a, b) == 1.0), start=1
):
    _la, _lb = signaller_rule_name(_a), signaller_rule_name(_b)
    _kind = "tracking" if ("track" in _la and "track" in _lb) else "XOR-type"
    _PERFECT_CATALOGUE[(_a, _b)] = (_num, _kind)


def describe_convention(conv_id, is_perfect: bool) -> str:
    """One-line description of a joint (S0, S1) convention id for the log."""
    id0, id1 = conv_id
    desc = f"S0: {signaller_rule_name(id0)} | S1: {signaller_rule_name(id1)}"
    if is_perfect and conv_id in _PERFECT_CATALOGUE:
        num, kind = _PERFECT_CATALOGUE[conv_id]
        return f"{desc}  ->  PERFECT #{num}/24 ({kind})"
    if is_perfect:
        return f"{desc}  ->  perfect (not in catalogue?!)"
    return f"{desc}  ->  not perfect"


def format_episode(graph, record) -> str:
    """Compact one-line render of one played episode, CAT/DOG + light ON/OFF."""
    items = ", ".join(f"G{g}={ITEM_NAME[record.items[g]]}" for g in graph.guessers)
    signals = ", ".join(f"S{s}={SIGNAL_NAME[record.signals[s]]}" for s in graph.signallers)
    guesses = ", ".join(
        f"G{g}={ITEM_NAME[record.guesses[g]]}{'ok' if record.correct[g] else 'XX'}"
        for g in graph.guessers
    )
    return (f"items({items}) -> lights({signals}) -> guesses({guesses})  "
            f"R={record.reward:.2f}")


def example_episodes(env, signaller_agents, guesser_agents, n=5):
    """Play ``n`` fully-greedy example episodes (no exploration, not recorded)."""
    return [play_greedy_example(env, signaller_agents, guesser_agents) for _ in range(n)]


def _train(variant: str, env, num_episodes: int, decay: int, seed):
    """Train ``variant`` on ``env``; ``seed=None`` skips the trainer's own
    seeding entirely (used by Phase 2 to seed torch/random by hand first)."""
    if variant == "shared":
        _net, sig, gue, _h = train_shared_iql(
            env, num_episodes=num_episodes, epsilon_decay_episodes=decay, seed=seed,
        )
    elif variant == "two_brain":
        _s, _g, sig, gue, _h = train_two_brain_iql(
            env, num_episodes=num_episodes, epsilon_decay_episodes=decay, seed=seed,
        )
    elif variant == "independent":
        _s, _g, sig, gue, _h = train_independent_iql(
            env, num_episodes=num_episodes, epsilon_decay_episodes=decay, seed=seed,
        )
    else:
        raise ValueError(f"unknown variant {variant!r}")
    return sig, gue


def _analyse(env, signaller_agents, guesser_agents) -> dict:
    outcomes = exhaustive_joint_table(env.graph, env.item_space.n, signaller_agents, guesser_agents)
    mean_reward = sum(o.reward for o in outcomes) / len(outcomes)
    return {
        "summary": convention_summary(outcomes),
        "conv_id": joint_signaller_convention_id(env.graph, env.item_space.n, signaller_agents),
        "sig_info": signaller_info_metrics(env.graph, outcomes),
        "gue_info": guesser_info_metrics(env.graph, outcomes),
        "mean_reward": mean_reward,
        "is_perfect": mean_reward == 1.0,
    }


# --- Phase 1: convention-quality + information-theoretic metrics --------

def run_phase1() -> None:
    print("=" * 72)
    print("PHASE 1: convention-quality + information-theoretic metrics")
    print(f"{NUM_SEEDS} seeds x {len(VARIANTS)} variants, K_2,2, "
          f"{NUM_EPISODES} training episodes each")
    print("=" * 72, flush=True)

    for variant in VARIANTS:
        print(f"\n{'#' * 72}")
        print(f"# {variant}")
        print(f"{'#' * 72}", flush=True)

        results = []
        for seed in range(NUM_SEEDS):
            t0 = time.time()
            env = make_env(seed=seed)
            sig, gue = _train(variant, env, NUM_EPISODES, EPSILON_DECAY_EPISODES, seed=seed)
            r = _analyse(env, sig, gue)
            r["seed"] = seed
            results.append(r)
            print(f"  seed {seed:2d}  R={r['mean_reward']:.3f}  {r['summary']:35s}  "
                  f"conv_id={r['conv_id']}  ({time.time() - t0:.1f}s)", flush=True)
            print(f"           convention: {describe_convention(r['conv_id'], r['is_perfect'])}")
            # 5 example greedy episodes, so the reader can verify by eye which
            # of the 24 conventions this seed actually plays out.
            for i, rec in enumerate(example_episodes(env, sig, gue, n=5), start=1):
                print(f"             ex{i}: {format_episode(env.graph, rec)}")

        # --- PERFECT convention distribution (only R=1.0 seeds counted) ---
        # Non-perfect seeds have not settled on a working coordination
        # convention, so they are excluded from the convention count/entropy;
        # they are summarised separately below.
        total = len(results)
        perfect = [r for r in results if r["is_perfect"]]
        n_perfect = len(perfect)
        pcounter = Counter(r["conv_id"] for r in perfect)

        print(f"\n  --- {variant}: PERFECT convention distribution "
              f"({n_perfect}/{total} seeds reached a perfect convention) ---")
        if n_perfect:
            for conv_id, count in pcounter.most_common():
                print(f"      {conv_id}  {describe_convention(conv_id, True)}"
                      f"  : {count}/{n_perfect}")
            H = entropy_bits(pcounter, n_perfect)
            dominant_id, dominant_count = pcounter.most_common(1)[0]
            kinds = Counter(_PERFECT_CATALOGUE[cid][1] for cid in pcounter)
            print(f"      distinct PERFECT conventions found : {len(pcounter)} / 24 possible")
            print(f"        (of which {kinds.get('tracking', 0)} tracking, "
                  f"{kinds.get('XOR-type', 0)} XOR-type)")
            print(f"      dominant perfect convention        : {dominant_id} "
                  f"({dominant_count}/{n_perfect} = {100 * dominant_count / n_perfect:.1f}%)")
            print(f"      convention entropy H (perfect only): {H:.3f} bits "
                  f"(0 = always the same convention, higher = more spread)")
        else:
            print("      (no seed reached a perfect convention)")
        print(f"      complementarity rate               : "
              f"{100 * n_perfect / total:.1f}% of seeds perfect")

        nonperfect = [r for r in results if not r["is_perfect"]]
        if nonperfect:
            bands = Counter(round(r["mean_reward"], 3) for r in nonperfect)
            band_str = ", ".join(f"R={b}: {c}" for b, c in sorted(bands.items(), reverse=True))
            print(f"      non-perfect seeds ({len(nonperfect)}), by reward : {band_str}")

        # --- information-theoretic metrics, averaged across the 30 seeds ---
        graph = GRAPH()
        n = len(results)
        print(f"\n  --- {variant}: information-theoretic metrics (mean over {n} seeds) ---")
        for s in graph.signallers:
            util = sum(r["sig_info"][s].channel_utilisation for r in results) / n
            mi_str = ", ".join(
                f"I(M{s};x{j})={sum(r['sig_info'][s].mutual_information[j] for r in results) / n:.3f}"
                for j in graph.out_neighbours(s)
            )
            print(f"      S{s}: channel utilisation H(M{s})={util:.3f} bits | {mi_str}")
        for g in graph.guessers:
            er = sum(r["gue_info"][g].entropy_reduction for r in results) / n
            print(f"      G{g}: entropy reduction I(o{g};x{g})={er:.3f} bits "
                  f"(out of 1.000 max)")


# --- Phase 2: what decides which convention a run finds? ----------------

# (torch_seed, random_seed, env_seed) as a function of repeat index r.
# "coupled" reproduces the standard protocol (all three tied to one seed);
# the other three vary exactly one source while holding the other two fixed
# at 0, isolating that source's individual effect.
ABLATION_CONDITIONS = {
    "coupled (torch=random=env=r, the standard protocol)": lambda r: (r, r, r),
    "vary init only (torch=r, random=0, env=0)":           lambda r: (r, 0, 0),
    "vary explore/replay only (torch=0, random=r, env=0)": lambda r: (0, r, 0),
    "vary env item-order only (torch=0, random=0, env=r)": lambda r: (0, 0, r),
}


def run_phase2() -> None:
    print(f"\n\n{'=' * 72}")
    print("PHASE 2: what decides which convention a run finds?")
    print(f"{ABLATION_REPEATS} repeats x {len(ABLATION_CONDITIONS)} conditions x "
          f"{len(VARIANTS)} variants, K_2,2")
    print("Note: the graph itself is held fixed at K_2,2 throughout -- this phase")
    print("cannot attribute variance to *topology* (that needs cross-graph")
    print("comparison, Question 2). What it tests is which of the three RNG")
    print("streams a *single fixed topology's* training run draws on is actually")
    print("responsible for which of K_2,2's several equally-rewarding, mirror-")
    print("symmetric conventions gets found.")
    print("=" * 72, flush=True)

    for variant in VARIANTS:
        print(f"\n{'#' * 72}")
        print(f"# {variant}")
        print(f"{'#' * 72}", flush=True)

        condition_counters = {}
        for cond_name, seed_fn in ABLATION_CONDITIONS.items():
            perfect_ids = []
            n_perfect = 0
            for r in range(ABLATION_REPEATS):
                torch_seed, random_seed, env_seed = seed_fn(r)
                torch.manual_seed(torch_seed)
                random.seed(random_seed)
                env = make_env(seed=env_seed)
                # seed=None: skip the trainer's own seeding entirely -- the
                # manual seeding above is what actually controls this run.
                sig, gue = _train(variant, env, NUM_EPISODES, EPSILON_DECAY_EPISODES, seed=None)
                res = _analyse(env, sig, gue)
                # Only perfect (R=1.0) runs land on a genuine coordination
                # convention; count distinct conventions among those only.
                if res["is_perfect"]:
                    n_perfect += 1
                    perfect_ids.append(res["conv_id"])

            counter = Counter(perfect_ids)
            condition_counters[cond_name] = (counter, n_perfect)
            print(f"\n  {cond_name}  ({n_perfect}/{ABLATION_REPEATS} repeats perfect)")
            for conv_id, count in counter.most_common():
                print(f"      {conv_id}  {describe_convention(conv_id, True)}"
                      f"  : {count}/{n_perfect}")
            print(f"      distinct PERFECT conventions: {len(counter)} "
                  f"(from {n_perfect} perfect repeats)")

        print(f"\n  --- {variant}: summary (distinct PERFECT conventions per condition) ---")
        for cond_name, (counter, n_perfect) in condition_counters.items():
            print(f"      {cond_name}: {len(counter)} distinct "
                  f"({n_perfect}/{ABLATION_REPEATS} perfect)")


def main() -> None:
    overall_start = time.time()
    run_phase1()
    run_phase2()
    print(f"\n\n{'=' * 72}")
    print(f"convention_diversity_k22.py complete in {(time.time() - overall_start) / 60:.1f} min")
    print("=" * 72, flush=True)


if __name__ == "__main__":
    main()
