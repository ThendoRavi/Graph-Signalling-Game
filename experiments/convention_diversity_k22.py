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
from gsg.training.trainer import train_independent_iql, train_shared_iql, train_two_brain_iql

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

        # --- joint convention distribution / entropy / complementarity ---
        ids = [r["conv_id"] for r in results]
        counter = Counter(ids)
        total = len(ids)
        H = entropy_bits(counter, total)
        dominant_id, dominant_count = counter.most_common(1)[0]
        complementarity = sum(1 for r in results if r["summary"].startswith("perfect")) / total

        print(f"\n  --- {variant}: joint convention distribution (n={total}) ---")
        for conv_id, count in counter.most_common():
            print(f"      {conv_id}: {count}/{total} ({100 * count / total:.1f}%)")
        print(f"      distinct conventions found : {len(counter)}")
        print(f"      dominant convention        : {dominant_id} "
              f"({dominant_count}/{total} = {100 * dominant_count / total:.1f}%)")
        print(f"      convention entropy H       : {H:.3f} bits "
              f"(0 = always the same convention, higher = more spread)")
        print(f"      complementarity rate       : {complementarity * 100:.1f}% of seeds "
              f"reached a perfect (complementary) convention")

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
            ids = []
            for r in range(ABLATION_REPEATS):
                torch_seed, random_seed, env_seed = seed_fn(r)
                torch.manual_seed(torch_seed)
                random.seed(random_seed)
                env = make_env(seed=env_seed)
                # seed=None: skip the trainer's own seeding entirely -- the
                # manual seeding above is what actually controls this run.
                sig, gue = _train(variant, env, NUM_EPISODES, EPSILON_DECAY_EPISODES, seed=None)
                conv_id = joint_signaller_convention_id(env.graph, env.item_space.n, sig)
                ids.append(conv_id)

            counter = Counter(ids)
            condition_counters[cond_name] = counter
            print(f"\n  {cond_name}")
            for conv_id, count in counter.most_common():
                print(f"      {conv_id}: {count}/{ABLATION_REPEATS}")
            print(f"      distinct conventions: {len(counter)}/{ABLATION_REPEATS} repeats")

        print(f"\n  --- {variant}: summary (distinct conventions per condition) ---")
        for cond_name, counter in condition_counters.items():
            print(f"      {cond_name}: {len(counter)} distinct")


def main() -> None:
    overall_start = time.time()
    run_phase1()
    run_phase2()
    print(f"\n\n{'=' * 72}")
    print(f"convention_diversity_k22.py complete in {(time.time() - overall_start) / 60:.1f} min")
    print("=" * 72, flush=True)


if __name__ == "__main__":
    main()
