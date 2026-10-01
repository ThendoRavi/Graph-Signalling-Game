"""What makes K_{2,2} agents land on an XOR (holistic) convention?

Background. In the convention-diversity run (convention_diversity_k22.py), the
24 perfect K_{2,2} conventions split into 8 "tracking" (each light relays one
guesser's item -- compositional) and 16 "XOR-type" (at least one light encodes
the SAME/DIFFER relation between the two items -- holistic). Learning
overwhelmingly finds tracking conventions. XOR-type perfect conventions
appeared only three times in that whole experiment, all in the *independent*
variant, and only in the ablation condition that pinned network-init and
item-order while sweeping the exploration/replay RNG. This script is the
focused follow-up that tests *what actually drives* those rare XOR outcomes.

Three sub-experiments, all on K_{2,2}:

  Experiment 1 -- Base rate under the standard (coupled) protocol.
  Independent variant, EXP1_N fully-coupled seeds (torch=random=env=r), to
  measure how often XOR appears under normal training at larger N than the
  P=30 main run (which saw zero).

  Experiment 2 -- Exploration sweep at several fixed launch points.
  Independent variant. For each fixed (init_seed, env_seed) launch point, the
  exploration/replay seed is swept over EXP2_REPEATS values. This disentangles
  three hypotheses for the XOR cases:
    * if XOR recurs at a similar rate across *all* launch points, it is the
      exploration/replay stream itself that can reach XOR basins (launch point
      irrelevant);
    * if XOR appears only when env is pinned at a particular value, the
      specific item-presentation order matters;
    * if XOR mostly vanishes at larger EXP2_REPEATS, it was rare chance.
  The launch grid deliberately includes points that share an init but differ
  in env (and vice versa) to separate the two.

  Experiment 3 -- Architecture control.
  The same fixed-launch exploration sweep for the shared-brain and two-brain
  variants, to confirm that the coupling of signaller weights in those designs
  suppresses XOR entirely (i.e. that the independent architecture is a
  necessary condition).

Each run is classified with the Boolean-function framework
(:mod:`gsg.evaluation.boolean_analysis`): a perfect convention is "XOR-type"
if at least one signaller's learned bit is a parity (XOR/XNOR) function, and
"tracking" if both are dictators (single-item relays).

Run:  python experiments/xor_conditions_k22.py
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
from gsg.evaluation.boolean_analysis import analyse_signaller_policy
from gsg.evaluation.conventions import (
    exhaustive_joint_table,
    extract_all_policies,
    joint_signaller_convention_id,
)
from gsg.training.trainer import train_independent_iql, train_shared_iql, train_two_brain_iql

# --- configuration -----------------------------------------------------

TRAIN_EPISODES = 4_000
EPSILON_DECAY = 2_500

EXP1_N = 60  # fully-coupled seeds, independent, to measure base XOR rate

EXP2_REPEATS = 40  # exploration seeds swept per launch point
# (init_seed, env_seed) launch points, held fixed while exploration varies.
# (0,0) is the original condition; the diagonal points test launch-point
# dependence; (0,7)/(7,0) separate "env pinned" from "init pinned".
EXP2_LAUNCH_POINTS = [(0, 0), (1, 1), (2, 2), (0, 7), (7, 0)]

EXP3_REPEATS = 40  # exploration seeds swept, architecture control
EXP3_VARIANTS = ["shared", "two_brain"]


def make_env(seed=None) -> GraphSignallingParallelEnv:
    return GraphSignallingParallelEnv(SignallingGraph.complete_bipartite(2, 2), seed=seed)


def _train(variant, env, seed):
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


def run_one(variant, init_seed, random_seed, env_seed):
    """Train one population with the three RNG sources set independently by
    hand (seed=None skips the trainer's own coupled seeding). Returns
    (kind, conv_id) with kind in {"tracking", "XOR-type", "non-perfect"}."""
    torch.manual_seed(init_seed)
    random.seed(random_seed)
    env = make_env(seed=env_seed)
    sig, gue = _train(variant, env, seed=None)

    outcomes = exhaustive_joint_table(env.graph, env.item_space.n, sig, gue)
    mean_reward = sum(o.reward for o in outcomes) / len(outcomes)
    conv_id = joint_signaller_convention_id(env.graph, env.item_space.n, sig)
    if mean_reward < 1.0:
        return "non-perfect", conv_id
    policies = extract_all_policies(env.graph, env.item_space.n, sig, gue)
    funcs = [analyse_signaller_policy(policies[f"S{s}"], env.graph.out_neighbours(s))
             for s in env.graph.signallers]
    kind = "XOR-type" if any(f.kind == "parity" for f in funcs) else "tracking"
    return kind, conv_id


def summarise(label, outcomes):
    """outcomes: list of (kind, conv_id). Print counts + XOR rate + distinct XOR."""
    n = len(outcomes)
    kinds = Counter(k for k, _ in outcomes)
    n_perfect = kinds["tracking"] + kinds["XOR-type"]
    n_xor = kinds["XOR-type"]
    xor_of_all = 100 * n_xor / n if n else 0.0
    xor_of_perfect = 100 * n_xor / n_perfect if n_perfect else 0.0
    print(f"  {label}")
    print(f"      runs={n}  perfect={n_perfect}  tracking={kinds['tracking']}  "
          f"XOR-type={n_xor}  non-perfect={kinds['non-perfect']}")
    print(f"      XOR rate: {xor_of_all:.1f}% of all runs, {xor_of_perfect:.1f}% of perfect runs")
    xor_ids = Counter(cid for k, cid in outcomes if k == "XOR-type")
    if xor_ids:
        print(f"      distinct XOR conventions: {dict(xor_ids)}")
    return {"n": n, "perfect": n_perfect, "xor": n_xor,
            "xor_of_perfect": xor_of_perfect}


def main():
    t0 = time.time()
    print("=" * 72)
    print("What drives XOR (holistic) conventions on K_2,2?")
    print(f"{TRAIN_EPISODES} training episodes per run")
    print("=" * 72, flush=True)

    # --- Experiment 1: base rate under the standard coupled protocol ---
    print(f"\n{'#' * 72}")
    print(f"# EXPERIMENT 1: base XOR rate, independent, coupled protocol, N={EXP1_N}")
    print(f"{'#' * 72}", flush=True)
    exp1 = []
    for r in range(EXP1_N):
        exp1.append(run_one("independent", r, r, r))
        if r % 10 == 9:
            print(f"    ...{r + 1}/{EXP1_N} done ({time.time() - t0:.0f}s)", flush=True)
    exp1_summ = summarise("independent / coupled", exp1)

    # --- Experiment 2: exploration sweep at fixed launch points ---
    print(f"\n{'#' * 72}")
    print("# EXPERIMENT 2: independent, sweep exploration/replay seed at fixed")
    print(f"#   launch points; {EXP2_REPEATS} exploration seeds per launch point")
    print(f"{'#' * 72}", flush=True)
    exp2_summ = {}
    for (init_seed, env_seed) in EXP2_LAUNCH_POINTS:
        outcomes = []
        xor_seeds = []
        for e in range(EXP2_REPEATS):
            kind, cid = run_one("independent", init_seed, e, env_seed)
            outcomes.append((kind, cid))
            if kind == "XOR-type":
                xor_seeds.append(e)
        print()
        s = summarise(f"launch (init={init_seed}, env={env_seed}); exploration swept 0..{EXP2_REPEATS - 1}",
                      outcomes)
        if xor_seeds:
            print(f"      exploration seeds that produced XOR: {xor_seeds}")
        exp2_summ[(init_seed, env_seed)] = s
        print(f"    (cumulative {time.time() - t0:.0f}s)", flush=True)

    # --- Experiment 3: architecture control ---
    print(f"\n{'#' * 72}")
    print("# EXPERIMENT 3: architecture control -- can shared / two_brain reach XOR?")
    print(f"#   exploration swept 0..{EXP3_REPEATS - 1} at fixed launch (init=0, env=0)")
    print(f"{'#' * 72}", flush=True)
    for variant in EXP3_VARIANTS:
        outcomes = [run_one(variant, 0, e, 0) for e in range(EXP3_REPEATS)]
        print()
        summarise(f"{variant} / launch (0,0), exploration swept", outcomes)
        print(f"    (cumulative {time.time() - t0:.0f}s)", flush=True)

    # --- overall verdict ---
    print(f"\n{'=' * 72}")
    print("SUMMARY: XOR rate (% of perfect runs) by condition")
    print(f"{'=' * 72}")
    print(f"  Exp1 independent/coupled (N={EXP1_N}): {exp1_summ['xor_of_perfect']:.1f}% "
          f"({exp1_summ['xor']} XOR / {exp1_summ['perfect']} perfect)")
    for lp, s in exp2_summ.items():
        print(f"  Exp2 independent, launch {lp}: {s['xor_of_perfect']:.1f}% "
              f"({s['xor']} XOR / {s['perfect']} perfect)")
    print(f"\n(total wall time: {(time.time() - t0) / 60:.1f} min)")
    print("=" * 72, flush=True)


if __name__ == "__main__":
    main()
