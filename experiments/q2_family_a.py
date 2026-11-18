"""Question 2, Family A: the complete-bipartite ratio sweep, with inverses.

Trains the three IQL variants on twelve graphs K_{m,n} (m signallers, n guessers):
the nine Family-A conditions of the methodology chapter plus the three inverse
graphs they lack, so every asymmetric graph has its counterpart:

    K_{1,2} <-> K_{2,1}    K_{1,3} <-> K_{3,1}    K_{1,4} <-> K_{4,1}
    K_{2,3} <-> K_{3,2}    K_{2,4} <-> K_{4,2}    K_{2,2}, K_{3,3} (self-inverse)

What counts as success
----------------------
With fewer signallers than guessers R = 1.0 is impossible, so "success" means
reaching the graph's own best achievable reward. That ceiling is computed exactly
(gsg.baselines.oracle_baseline.complete_bipartite_optimum) and verified by playing
the oracle. Each graph is read against four reference points: random play and
no-communication (both 0.5), the best a pure tracking code can do, and the oracle.

Protocol
--------
Hyperparameters as in Question 1. The schedule scales with the size of the policy
table the population must learn, E = m*2^n + n*2^m entries: epsilon decays over
156.25*E episodes (2,500 on K_{2,2}). E is the same for K_{m,n} and K_{n,m}, so
counterparts always get identical schedules. Greedy policies are read exactly every
1/48 of training (global RNG saved and restored, so measuring never alters training).
Two thirds of the way through is the "primary" checkpoint, the analogue of episode
4,000 of 6,000 in Question 1; the last third measures stability.

Outputs (results/question2/family_a/ unless GSG_Q2_OUT is set)
--------------------------------------------------------------
    runs.jsonl          checkpoint of every finished run; a restart resumes from it
    runs.csv            one row per run
    trajectories.csv    every change of convention or reward, per run
    summary.csv         every metric with its CI and p-value
    figures/<cond>/     topology, learning curves, reward bands, conventions, policy
                        tables, action-response matrices (settled and during training)
    figures/summary/    cross-graph and counterpart comparisons

Run:  python experiments/q2_family_a.py [--stage train|analyse|all]
      env: GSG_Q2_SEEDS, GSG_Q2_SEED_START, GSG_Q2_CONDITIONS, GSG_Q2_VARIANTS,
           GSG_Q2_SCALE, GSG_Q2_OUT, GSG_WORKERS, GSG_BLOCKS
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
import random
import sys
import time
import traceback
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)

from gsg.baselines.oracle_baseline import complete_bipartite_optimum, complete_bipartite_oracle
from gsg.baselines.random_baseline import RandomGuesser, RandomSignaller
from gsg.environment.graphs import SignallingGraph
from gsg.environment.pettingzoo_env import GraphSignallingParallelEnv
from gsg.evaluation.conventions import (
    WorldOutcome,
    canonical_id,
    exhaustive_joint_table,
    greedy_policy,
)
from gsg.evaluation.information_theory import guesser_info_metrics, signaller_info_metrics
from gsg.evaluation.metrics import evaluate
from gsg.training.trainer import train_independent_iql, train_shared_iql, train_two_brain_iql

from q1_hypothesis_tests_k22 import (
    _weights_hash,
    binom_test,
    entropy_from_counts,
    fisher_exact,
    holm,
    mc_pvalue,
    permutation_homogeneity,
    wilson,
)

# --- configuration -----------------------------------------------------


@dataclass(frozen=True)
class Condition:
    label: str
    m: int                # signallers
    n: int                # guessers
    note: str
    counterpart: str      # label of K_{n,m}

    @property
    def name(self) -> str:
        return f"K{self.m},{self.n}"

    @property
    def rho(self) -> float:
        return self.m / self.n

    @property
    def table_entries(self) -> int:
        return self.m * 2 ** self.n + self.n * 2 ** self.m


# "r" marks the inverse graphs added to the methodology's nine conditions.
ALL_CONDITIONS = [
    Condition("A3", 1, 4, "extreme bottleneck", "A3r"),
    Condition("A2", 1, 3, "", "A9"),
    Condition("A1", 1, 2, "scale pair with A6", "A1r"),
    Condition("A6", 2, 4, "scale pair with A1", "A6r"),
    Condition("A5", 2, 3, "", "A8"),
    Condition("A4", 2, 2, "Question 1 baseline", "A4"),
    Condition("A7", 3, 3, "scale pair with A4", "A7"),
    Condition("A8", 3, 2, "", "A5"),
    Condition("A1r", 2, 1, "inverse of A1", "A1"),
    Condition("A6r", 4, 2, "inverse of A6", "A6"),
    Condition("A9", 3, 1, "signaller surplus", "A2"),
    Condition("A3r", 4, 1, "inverse of A3", "A3"),
]
BY_LABEL = {c.label: c for c in ALL_CONDITIONS}
SCALE_PAIRS = [("A1", "A6"), ("A1r", "A6r"), ("A4", "A7")]

_selected = os.environ.get("GSG_Q2_CONDITIONS")
CONDITIONS = [BY_LABEL[x] for x in _selected.split(",")] if _selected else ALL_CONDITIONS
VARIANTS = os.environ.get("GSG_Q2_VARIANTS", "shared,two_brain,independent").split(",")
NUM_SEEDS = int(os.environ.get("GSG_Q2_SEEDS", 100))
SEED_START = int(os.environ.get("GSG_Q2_SEED_START", 0))
NUM_BLOCKS = int(os.environ.get("GSG_BLOCKS", 5))
SCALE = float(os.environ.get("GSG_Q2_SCALE", 1.0))      # < 1 only for smoke tests
WORKERS = int(os.environ.get("GSG_WORKERS") or os.environ.get("SLURM_CPUS_PER_TASK") or os.cpu_count() or 1)

DECAY_PER_TABLE_ENTRY = 156.25        # 2,500 episodes on K_{2,2}
# Total training, as a multiple of the epsilon-decay length, by policy-table size.
# Set from a pilot: the small graphs had settled by 2-4 decay lengths, K_{1,4} needed
# about 6, and K_{2,4} had not fully reached its ceiling at 6.
TRAIN_MULTIPLIER = {8: 6, 14: 6, 16: 6, 24: 8, 28: 8, 48: 12}
SNAPSHOTS_PER_RUN = 48
PRIMARY_SNAPSHOT = 32                 # 2/3 of training
REPRO_SEEDS = 2                       # seeds per (graph, variant) re-run without instrumentation
AR_WINDOWS = 4                        # action-response windows during training
SNAPSHOT_DETAIL_SEEDS = 3             # seeds whose full policies are logged during training
CROSSPLAY_RUNS = 15

ALPHA = 0.05
STATS_SEED = 20261118
MC_SIMS = 5_000
PERM_SIMS = 5_000
ENUMERATE_LIMIT = 2 ** 25             # joint signaller rules; larger spaces are sampled
SAMPLE_CONVENTIONS = 2_000_000
EPS = 1e-9

OUT_DIR = os.environ.get("GSG_Q2_OUT") or os.path.join(HERE, "..", "results", "question2", "family_a")
FIG_DIR = os.path.join(OUT_DIR, "figures")

ITEM_LETTER = {0: "C", 1: "D"}
FAMILIES = ["tracking", "XOR-type", "nonlinear", "partial tracking", "silent"]


def schedule(cond: Condition):
    """(total_episodes, epsilon_decay_episodes, snapshot_every) for a condition."""
    decay = max(10, int(DECAY_PER_TABLE_ENTRY * cond.table_entries * SCALE))
    every = max(1, decay * TRAIN_MULTIPLIER[cond.table_entries] // SNAPSHOTS_PER_RUN)
    return every * SNAPSHOTS_PER_RUN, decay, every


# --- the game as lookup tables -------------------------------------------
#
# A greedy policy is stored as one integer per agent: bit i is its action for the
# i-th observation in itertools.product order (gsg.evaluation.conventions.canonical_id).
# Everything downstream (rewards, conventions, information metrics, cross-play) is
# computed exactly from those integers, so no network is needed after training.


class Game:
    def __init__(self, m: int, n: int) -> None:
        self.m, self.n = m, n
        self.graph = SignallingGraph.complete_bipartite(m, n)
        self.worlds = list(itertools.product((0, 1), repeat=n))
        self.patterns = list(itertools.product((0, 1), repeat=m))

    def lights(self, s_ids, world: int):
        return tuple((s_ids[s] >> world) & 1 for s in range(self.m))

    def pattern_index(self, lights) -> int:
        return sum(bit << (self.m - 1 - s) for s, bit in enumerate(lights))

    def guesses(self, g_ids, lights):
        index = self.pattern_index(lights)
        return tuple((g_ids[g] >> index) & 1 for g in range(self.n))

    def correct_counts(self, s_ids, g_ids):
        counts = [0] * self.n
        for w, items in enumerate(self.worlds):
            for g, guess in enumerate(self.guesses(g_ids, self.lights(s_ids, w))):
                counts[g] += guess == items[g]
        return counts

    def reward(self, s_ids, g_ids) -> float:
        return sum(self.correct_counts(s_ids, g_ids)) / (self.n * len(self.worlds))

    def accuracies(self, s_ids, g_ids):
        return [c / len(self.worlds) for c in self.correct_counts(s_ids, g_ids)]

    def outcomes(self, s_ids, g_ids):
        result = []
        for w, items in enumerate(self.worlds):
            lights = self.lights(s_ids, w)
            guesses = self.guesses(g_ids, lights)
            correct = {g: guesses[g] == items[g] for g in range(self.n)}
            result.append(WorldOutcome(dict(enumerate(items)), dict(enumerate(lights)),
                                       dict(enumerate(guesses)), correct,
                                       sum(correct.values()) / self.n))
        return result

    def needed_signallers(self, s_ids, g_ids):
        """Signallers that cannot be silenced: going through them in order, each
        light is pinned to a constant if no guess in any world changes."""
        reference = [self.guesses(g_ids, self.lights(s_ids, w)) for w in range(len(self.worlds))]
        pinned = {}
        for s in range(self.m):
            for constant in (0, 1):
                trial = {**pinned, s: constant}
                if all(self.guesses(g_ids, tuple(trial.get(i, b) for i, b in
                                                 enumerate(self.lights(s_ids, w)))) == reference[w]
                       for w in range(len(self.worlds))):
                    pinned = trial
                    break
        return [s for s in range(self.m) if s not in pinned]


@lru_cache(maxsize=None)
def game(m: int, n: int) -> Game:
    return Game(m, n)


@lru_cache(maxsize=None)
def ceiling(m: int, n: int) -> float:
    return complete_bipartite_optimum(m, n)[0]


def tracking_bound(m: int, n: int) -> float:
    """Reward of the best pure tracking code: min(m, n) guessers served, the rest at chance."""
    served = min(m, n)
    return (served + 0.5 * (n - served)) / n


@dataclass(frozen=True)
class Rule:
    kind: str             # constant | track | parity | and-or | majority | threshold | other
    support: tuple        # input positions the rule depends on
    name: str


@lru_cache(maxsize=None)
def describe_rule(rule_id: int, k: int, prefix: str = "x") -> Rule:
    """Structural description of a Boolean rule on k binary inputs."""
    inputs = list(itertools.product((0, 1), repeat=k))
    table = {obs: (rule_id >> i) & 1 for i, obs in enumerate(inputs)}

    def flipped(obs, j):
        return obs[:j] + (1 - obs[j],) + obs[j + 1:]

    support = tuple(j for j in range(k) if any(table[o] != table[flipped(o, j)] for o in inputs))
    if not support:
        return Rule("constant", (), f"always {'ON' if table[inputs[0]] else 'OFF'}" if prefix == "x"
                    else f"always {'DOG' if table[inputs[0]] else 'CAT'}")
    if len(support) == 1:
        j = support[0]
        direct = table[tuple(1 if i == j else 0 for i in range(k))] == 1
        return Rule("track", support, f"{'' if direct else 'NOT '}{prefix}{j}")

    labels = [f"{prefix}{j}" for j in support]
    base = table[tuple(0 for _ in range(k))]
    if all(table[o] == (base ^ (sum(o[j] for j in support) % 2)) for o in inputs):
        return Rule("parity", support, f"{'XNOR' if base else 'XOR'}({','.join(labels)})")

    r = len(support)
    for polarity in itertools.product((0, 1), repeat=r):
        for t in range(1, r + 1):
            if all(table[o] == (sum(o[j] ^ p for j, p in zip(support, polarity)) >= t) for o in inputs):
                literals = ",".join(("!" if p else "") + lab for lab, p in zip(labels, polarity))
                if t == r:
                    return Rule("and-or", support, f"AND({literals})")
                if t == 1:
                    return Rule("and-or", support, f"OR({literals})")
                if r % 2 == 1 and 2 * t == r + 1:
                    return Rule("majority", support, f"MAJ({literals})")
                return Rule("threshold", support, f"AT-LEAST-{t}({literals})")
    bits = "".join(str(table[o]) for o in inputs)
    return Rule("other", support, f"other[{bits}]")


def family_of(s_ids, m: int, n: int) -> str:
    """Family of a joint signaller convention, defined for every K_{m,n}."""
    rules = [describe_rule(r, n) for r in s_ids]
    active = [r for r in rules if r.kind != "constant"]
    if not active:
        return "silent"
    if all(r.kind == "track" for r in active):
        return "tracking" if len({r.support[0] for r in active}) == min(m, n) else "partial tracking"
    if all(r.kind in ("track", "parity") for r in active):
        return "XOR-type"
    return "nonlinear"


def signature_of(s_ids, n: int) -> str:
    rules = [describe_rule(r, n) for r in s_ids]
    kinds = Counter(r.kind for r in rules)
    items = {r.support[0] for r in rules if r.kind == "track"}
    text = " + ".join(f"{c} {k}" for k, c in sorted(kinds.items()))
    return f"{text} ({len(items)} items relayed)"


def convention_text(s_ids, n: int) -> str:
    return "; ".join(f"S{s}: {describe_rule(r, n).name}" for s, r in enumerate(s_ids))


def guesser_text(g_ids, m: int) -> str:
    return "; ".join(f"G{g}: {describe_rule(r, m, 'S').name}" for g, r in enumerate(g_ids))


def world_lines(g: Game, s_ids, g_ids):
    """One line per world: items > lights > guesses (wrong guesses marked '!')."""
    lines = []
    for w, items in enumerate(g.worlds):
        lights = g.lights(s_ids, w)
        guesses = g.guesses(g_ids, lights)
        said = "".join(ITEM_LETTER[a] + ("" if a == b else "!") for a, b in zip(guesses, items))
        lines.append(f"{''.join(ITEM_LETTER[x] for x in items)} > {''.join(map(str, lights))} > {said}")
    return lines


# --- one training run (executed in a worker process) --------------------

TRAINERS = {"shared": train_shared_iql, "two_brain": train_two_brain_iql,
            "independent": train_independent_iql}


def read_tables(g: Game, signallers, guessers):
    s_ids = [canonical_id(greedy_policy(signallers[s], 2, g.n), 2, 2, g.n) for s in g.graph.signallers]
    g_ids = [canonical_id(greedy_policy(guessers[x], 2, g.m), 2, 2, g.m) for x in g.graph.guessers]
    return s_ids, g_ids


def run_key(label, variant, seed, mode):
    total, decay, _every = schedule(BY_LABEL[label])
    return f"{label}|{variant}|{seed}|{mode}|{total}|{decay}"


def run_one(job):
    """Train one population. ``job = (label, variant, seed, mode)``; mode "full" is
    instrumented, mode "plain" has no instrumentation and exists to prove that the
    instrumentation and the seeding leave the trained weights unchanged."""
    label, variant, seed, mode = job
    torch.set_num_threads(1)
    t0 = time.time()
    cond = BY_LABEL[label]
    total, decay, every = schedule(cond)
    g = game(cond.m, cond.n)
    env = GraphSignallingParallelEnv(g.graph, seed=seed)
    edges = list(g.graph.edges)
    snaps = []
    ar = [[[0, 0, 0, 0] for _ in edges] for _ in range(AR_WINDOWS)]
    window_reward = [0.0] * AR_WINDOWS
    window_count = [0] * AR_WINDOWS

    def on_log(episode, _rewards, *rest):
        # act() draws random numbers even at epsilon 0, so the RNG state is restored
        py_state, torch_state = random.getstate(), torch.get_rng_state()
        try:
            s_ids, g_ids = read_tables(g, rest[-2], rest[-1])
        finally:
            random.setstate(py_state)
            torch.set_rng_state(torch_state)
        snaps.append([episode, s_ids, g_ids, g.reward(s_ids, g_ids)])

    def on_episode(index, record):
        w = min(AR_WINDOWS - 1, index * AR_WINDOWS // total)
        window_reward[w] += record.reward
        window_count[w] += 1
        for e, (s, x) in enumerate(edges):
            ar[w][e][2 * record.signals[s] + record.guesses[x]] += 1

    kwargs = dict(num_episodes=total, epsilon_decay_episodes=decay, seed=seed)
    if mode == "full":
        kwargs.update(log_every=every, on_log=on_log, on_episode=on_episode)
    out = TRAINERS[variant](env, **kwargs)
    signallers, guessers = out[-3], out[-2]
    result = dict(key=run_key(label, variant, seed, mode), label=label, variant=variant, seed=seed,
                  mode=mode, weights=_weights_hash(signallers, guessers), seconds=time.time() - t0)
    if mode == "full":
        result.update(snaps=snaps, ar=ar,
                      train_reward=[r / max(c, 1) for r, c in zip(window_reward, window_count)])
    return result


def planned_jobs():
    jobs = []
    for seed in range(SEED_START, SEED_START + NUM_SEEDS):   # seed-first: a partial run stays balanced
        for cond in CONDITIONS:
            for variant in VARIANTS:
                jobs.append((cond.label, variant, seed, "full"))
    for seed in range(SEED_START, SEED_START + REPRO_SEEDS):
        for cond in CONDITIONS:
            for variant in VARIANTS:
                jobs.append((cond.label, variant, seed, "plain"))
    return jobs


def load_records():
    path = os.path.join(OUT_DIR, "runs.jsonl")
    records = {}
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:      # a line cut off by a timeout
                    continue
                records[rec["key"]] = rec
    return records


def train_stage() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    done = load_records()
    jobs = [j for j in planned_jobs() if run_key(*j) not in done]
    print("=" * 78)
    print(f"TRAINING: {len(jobs)} runs to do, {len(done)} already finished, {WORKERS} workers")
    print(f"{'cond':5s}{'graph':7s}{'rho':>6s}{'table':>7s}{'episodes':>10s}{'decay':>8s}{'snapshot':>10s}{'ceiling':>9s}")
    for c in CONDITIONS:
        total, decay, every = schedule(c)
        print(f"{c.label:5s}{c.name:7s}{c.rho:6.2f}{c.table_entries:7d}{total:10d}{decay:8d}{every:10d}"
              f"{ceiling(c.m, c.n):9.4f}")
    print("=" * 78, flush=True)
    if not jobs:
        return
    t0 = time.time()
    failures = 0
    with ProcessPoolExecutor(max_workers=WORKERS) as pool, \
            open(os.path.join(OUT_DIR, "runs.jsonl"), "a", encoding="utf-8") as out:
        futures = {pool.submit(run_one, j): j for j in jobs}
        for i, future in enumerate(as_completed(futures), 1):
            job = futures[future]
            try:
                rec = future.result()
            except Exception:
                failures += 1
                print(f"  FAILED {job}\n{traceback.format_exc()}", flush=True)
                continue
            out.write(json.dumps(rec, separators=(",", ":")) + "\n")
            out.flush()
            if i % max(1, len(jobs) // 200) == 0 or i == len(jobs):
                elapsed = time.time() - t0
                reward = f"R={rec['snaps'][-1][3]:.3f}" if rec["mode"] == "full" else "plain re-run"
                print(f"  [{i}/{len(jobs)}] {rec['label']} {rec['variant']} seed {rec['seed']} {reward} "
                      f"({rec['seconds']:.0f}s)  elapsed {elapsed / 60:.1f} min, "
                      f"about {elapsed / i * (len(jobs) - i) / 60:.0f} min left", flush=True)
    print(f"training finished in {(time.time() - t0) / 60:.1f} min, {failures} failed runs", flush=True)


# --- per-run derived quantities ------------------------------------------


def derive(rec, cond: Condition, ceil: float) -> dict:
    g = game(cond.m, cond.n)
    total, decay, every = schedule(cond)
    snaps = rec["snaps"]
    final = snaps[-1]
    s_ids, g_ids, reward = final[1], final[2], final[3]
    success = reward >= ceil - EPS
    d = dict(rec)
    primary = snaps[PRIMARY_SNAPSHOT - 1]
    d.update(s_ids=tuple(s_ids), g_ids=tuple(g_ids), reward=reward, success=success,
             success_primary=primary[3] >= ceil - EPS, reward_primary=primary[3],
             block=(rec["seed"] - SEED_START) * NUM_BLOCKS // NUM_SEEDS)
    d["first_success"] = next((e for e, _s, _g, r in snaps if r >= ceil - EPS), None)
    d["reward_lockin"] = d["convention_lockin"] = None
    if success:
        for e, sn, _gn, r in reversed(snaps):
            if r < ceil - EPS:
                break
            d["reward_lockin"] = e
        for e, sn, _gn, r in reversed(snaps):
            if r < ceil - EPS or sn != s_ids:
                break
            d["convention_lockin"] = e
    changes = [b[0] for a, b in zip(snaps, snaps[1:]) if a[1] != b[1]]
    d["changes"] = len(changes)
    d["changes_after_decay"] = sum(e > decay for e in changes)
    tail = [tuple(x[1]) for x in snaps[PRIMARY_SNAPSHOT - 1:]]
    if d["success_primary"]:
        d["dynamics"] = "held optimum" if success and all(x[3] >= ceil - EPS for x in snaps[PRIMARY_SNAPSHOT:]) \
            else "left optimum"
    elif success:
        d["dynamics"] = "escaped"
    elif len(set(tail)) == 1:
        d["dynamics"] = "frozen"
    elif any(c in tail[:i] and c != tail[i - 1] for i, c in enumerate(tail) if i):
        d["dynamics"] = "oscillating"
    else:
        d["dynamics"] = "drifting"
    needed = g.needed_signallers(s_ids, g_ids)
    d["needed"] = needed
    d["effective"] = tuple(s_ids[s] if s in needed else None for s in range(cond.m))
    d["family"] = family_of(s_ids, cond.m, cond.n)
    d["signature"] = signature_of(s_ids, cond.n)
    d["accuracies"] = g.accuracies(s_ids, g_ids)
    outcomes = g.outcomes(s_ids, g_ids)
    sinfo, ginfo = signaller_info_metrics(g.graph, outcomes), guesser_info_metrics(g.graph, outcomes)
    d["channel"] = [sinfo[s].channel_utilisation for s in range(cond.m)]
    d["mi"] = [[sinfo[s].mutual_information[x] for x in range(cond.n)] for s in range(cond.m)]
    d["entropy_reduction"] = [ginfo[x].entropy_reduction for x in range(cond.n)]
    return d


def trajectory_points(rec):
    points, previous = [], None
    for e, s_ids, _g, r in rec["snaps"]:
        state = (tuple(s_ids), round(r, 6))
        if state != previous:
            points.append((e, tuple(s_ids), r))
            previous = state
    return points


def trajectory_text(rec) -> str:
    return " -> ".join(f"ep{e} ({','.join(map(str, s))}) {r:.3f}" for e, s, r in trajectory_points(rec))


# --- convention space ------------------------------------------------------


def convention_space(cond: Condition, ceil: float, rng) -> dict:
    """Reward distribution over all joint signaller conventions (each paired with
    its best possible guessers), and the conventions that reach the ceiling."""
    m, n = cond.m, cond.n
    W = 2 ** n
    rules = 2 ** W
    total = rules ** m
    enumerated = total <= ENUMERATE_LIMIT
    X = np.array(game(m, n).worlds, dtype=bool)
    world_index = np.arange(W, dtype=np.int64)
    target = int(round(ceil * n * W))
    counts, optimal_rows = Counter(), []
    drawn = total if enumerated else SAMPLE_CONVENTIONS
    chunk = 100_000
    for start in range(0, drawn, chunk):
        size = min(chunk, drawn - start)
        if enumerated:
            index = np.arange(start, start + size, dtype=np.int64)
            R = np.stack([(index // rules ** s) % rules for s in range(m)], axis=1)
        else:
            R = rng.integers(0, rules, size=(size, m), dtype=np.int64)
        M = np.zeros((size, W), dtype=np.int64)
        for s in range(m):
            M |= ((R[:, s, None] >> world_index) & 1) << s
        correct = np.zeros(size, dtype=np.int64)
        for message in range(2 ** m):
            mask = M == message
            in_cell = mask.sum(axis=1)
            for g in range(n):
                ones = (mask & X[:, g, None].T).sum(axis=1)
                correct += np.maximum(ones, in_cell - ones)
        values, freq = np.unique(correct, return_counts=True)
        for v, c in zip(values, freq):
            counts[int(v)] += int(c)
        if enumerated:
            optimal_rows.append(R[correct == target])
    bands = {v / (n * W): c for v, c in sorted(counts.items())}
    result = dict(total=total, enumerated=enumerated, sampled=drawn, bands=bands)
    if enumerated:
        rows = np.concatenate(optimal_rows)
        result["n_optimal"] = len(rows)
        result["families"] = Counter(family_of([int(x) for x in row], m, n) for row in rows)
    else:
        result["n_optimal"], result["families"] = None, None
    return result


def verified_baselines(cond: Condition) -> dict:
    g = game(cond.m, cond.n)
    env = GraphSignallingParallelEnv(g.graph, seed=STATS_SEED)
    signallers, guessers = complete_bipartite_oracle(g.graph)
    outcomes = exhaustive_joint_table(g.graph, 2, signallers, guessers)
    oracle_exact = sum(o.reward for o in outcomes) / len(outcomes)
    oracle_sampled = evaluate(env, signallers, guessers, episodes=2_000, seed=STATS_SEED).mean_reward
    randoms = {s: RandomSignaller(seed=STATS_SEED + s) for s in g.graph.signallers}
    random_g = {x: RandomGuesser(2, seed=STATS_SEED + 100 + x) for x in g.graph.guessers}
    random_sampled = evaluate(env, randoms, random_g, episodes=10_000, seed=STATS_SEED).mean_reward
    _reward, code = complete_bipartite_optimum(cond.m, cond.n)
    return dict(oracle=oracle_exact, oracle_sampled=oracle_sampled, random=random_sampled,
                oracle_lines=world_lines_oracle(g, outcomes), code=code)


def world_lines_oracle(g: Game, outcomes):
    lines = []
    for o in outcomes:
        items = "".join(ITEM_LETTER[o.items[x]] for x in range(g.n))
        lights = "".join(str(o.signals[s]) for s in range(g.m))
        said = "".join(ITEM_LETTER[o.guesses[x]] + ("" if o.correct[x] else "!") for x in range(g.n))
        lines.append(f"{items} > {lights} > {said}")
    return lines


# --- statistics helpers ----------------------------------------------------

SUMMARY_ROWS = []


def summary(cond, variant, metric, value, ci=(None, None), p=None, note=""):
    SUMMARY_ROWS.append([cond, variant, metric, value, ci[0], ci[1], p, note])


def pct(k, n):
    return 100 * k / n if n else float("nan")


def ci_text(k, n):
    lo, hi = wilson(k, n)
    return f"{k}/{n} = {pct(k, n):.1f}% [{100 * lo:.1f}, {100 * hi:.1f}]"


def mean_ci(values):
    a = np.asarray(values, dtype=float)
    half = 1.96 * a.std(ddof=1) / math.sqrt(len(a)) if len(a) > 1 else float("nan")
    return float(a.mean()), float(a.mean() - half), float(a.mean() + half)


def uniform_convention_null(n, K, rng, sims=MC_SIMS):
    H, top, distinct = np.zeros(sims), np.zeros(sims), np.zeros(sims)
    for i in range(sims):
        _values, counts = np.unique(rng.integers(0, K, size=n), return_counts=True)
        H[i], top[i], distinct[i] = entropy_from_counts(counts), counts.max(), len(counts)
    return H, top, distinct


def spearman(x, y, rng, sims=PERM_SIMS):
    def ranks(values):
        v = np.asarray(values, dtype=float)
        order = np.argsort(v)
        r = np.empty(len(v))
        r[order] = np.arange(len(v), dtype=float)
        for value in np.unique(v):
            r[v == value] = r[v == value].mean()
        return r

    rx, ry = ranks(x), ranks(y)
    observed = float(np.corrcoef(rx, ry)[0, 1])
    hits = sum(abs(np.corrcoef(rx, rng.permutation(ry))[0, 1]) >= abs(observed) - 1e-12 for _ in range(sims))
    return observed, (1 + hits) / (1 + sims)


def conv_str(ids):
    return "(" + ",".join("*" if x is None else str(x) for x in ids) + ")"


# --- analysis of one condition ---------------------------------------------


def analyse_group(cond, name, runs, space, ceil, rng, verbose=True) -> dict:
    """All per-group statistics; ``name`` is a variant or 'POOLED'."""
    P = print if verbose else (lambda *a, **k: None)
    total, decay, every = schedule(cond)
    N = len(runs)
    wins = [r for r in runs if r["success"]]
    k = len(wins)
    rewards = [r["reward"] for r in runs]
    mean, lo, hi = mean_ci(rewards)
    span = ceil - 0.5
    res = dict(n=N, k=k, rate=k / N, rate_ci=wilson(k, N), mean=mean, mean_ci=(lo, hi),
               normalised=(mean - 0.5) / span, runs=runs)
    P(f"\n  --- {cond.label} {cond.name} / {name}: {N} runs ---")
    P(f"  A. REWARD AND SUCCESS (success = final greedy reward equals the ceiling {ceil:.4f})")
    P(f"     success rate              : {ci_text(k, N)}")
    P(f"     mean final reward         : {mean:.4f} [{lo:.4f}, {hi:.4f}]")
    P(f"     vs random 0.5             : +{mean - 0.5:.4f}; {sum(r > 0.5 + EPS for r in rewards)}/{N} runs above 0.5, "
      f"{sum(abs(r - 0.5) <= EPS for r in rewards)} at 0.5, {sum(r < 0.5 - EPS for r in rewards)} below")
    P(f"     vs oracle {ceil:.4f}         : gap {ceil - mean:.4f}; "
      f"share of the achievable gain realised {100 * res['normalised']:.1f}%")
    track = tracking_bound(cond.m, cond.n)
    P(f"     vs tracking-only {track:.4f}  : {sum(r > track + EPS for r in rewards)} above, "
      f"{sum(abs(r - track) <= EPS for r in rewards)} equal, {sum(r < track - EPS for r in rewards)} below")
    bands = Counter(round(r, 4) for r in rewards)
    res["bands"] = bands
    P("     final reward bands        : " + "  ".join(f"{b:.4f}: {c}" for b, c in sorted(bands.items(), reverse=True)))
    checkpoints = {}
    for frac in (0.25, 0.5, 2 / 3, 0.75, 1.0):
        i = max(0, int(round(SNAPSHOTS_PER_RUN * frac)) - 1)
        checkpoints[frac] = sum(r["snaps"][i][3] >= ceil - EPS for r in runs) / N
    res["checkpoints"] = checkpoints
    P("     success by training stage : " + "  ".join(
        f"{int(round(f * total))} eps: {100 * v:.1f}%" for f, v in checkpoints.items()))
    summary(cond.label, name, "success_rate", k / N, wilson(k, N), note=f"{k}/{N}")
    summary(cond.label, name, "mean_reward", mean, (lo, hi))
    summary(cond.label, name, "share_of_achievable_gain", res["normalised"])

    blocks = [r["block"] for r in runs]
    if len(set(blocks)) > 1:
        chi, p, _table = permutation_homogeneity(blocks, [r["success"] for r in runs], rng, sims=PERM_SIMS)
        rates = [np.mean([r["success"] for r in runs if r["block"] == b]) for b in sorted(set(blocks))]
        res["block_rates"], res["block_p"] = rates, p
        P(f"     seed-block reproducibility: rates {' '.join(f'{100 * x:.0f}%' for x in rates)}; "
          f"permutation chi2 = {chi:.2f}, p = {p:.3f}  (H0: same rate in every block)")
        summary(cond.label, name, "block_homogeneity", chi, p=p)

    P("  B. TIMING AND STABILITY")
    lock = [r["reward_lockin"] for r in wins]
    conv_lock = [r["convention_lockin"] for r in wins if r["convention_lockin"] is not None]
    res["lock_median"] = float(np.median(lock)) if lock else float("nan")
    if lock:
        q = np.percentile(lock, [25, 50, 75])
        P(f"     reward reaches its final optimum for good at: median ep {q[1]:.0f} (IQR {q[0]:.0f}-{q[2]:.0f}); "
          f"{100 * np.mean(np.array(lock) <= decay):.0f}% by the end of epsilon decay (ep {decay})")
    if conv_lock:
        q = np.percentile(conv_lock, [25, 50, 75])
        P(f"     exact convention last changes at            : median ep {q[1]:.0f} (IQR {q[0]:.0f}-{q[2]:.0f})")
    dyn = Counter(r["dynamics"] for r in runs)
    res["dynamics"] = dyn
    P(f"     from {PRIMARY_SNAPSHOT * every} (2/3 of training) to the end: " + ", ".join(
        f"{key} {dyn[key]}" for key in ("held optimum", "left optimum", "escaped", "frozen", "oscillating", "drifting")
        if dyn[key]))
    win_changes = [r["changes_after_decay"] for r in wins]
    lose_changes = [r["changes_after_decay"] for r in runs if not r["success"]]
    P(f"     convention changes after epsilon decay: successful {np.mean(win_changes) if win_changes else 0:.2f}, "
      f"unsuccessful {np.mean(lose_changes) if lose_changes else 0:.2f} per run")

    P("  C. CONVENTIONS REACHING THE CEILING")
    res["family_success"] = Counter(r["family"] for r in wins)
    res["family_all"] = Counter(r["family"] for r in runs)
    if wins:
        exact = Counter(r["s_ids"] for r in wins)
        effective = Counter(r["effective"] for r in wins)
        H = entropy_from_counts(list(exact.values()))
        top_conv, top_count = exact.most_common(1)[0]
        singletons = sum(c == 1 for c in exact.values())
        res.update(distinct=len(exact), effective_distinct=len(effective), H=H,
                   H_effective=entropy_from_counts(list(effective.values())),
                   top_share=top_count / k, coverage=1 - singletons / k)
        P(f"     distinct conventions      : {len(exact)} exact; {len(effective)} after ignoring signallers "
          f"the guessers can do without")
        P(f"     convention entropy        : {H:.3f} bits exact, {res['H_effective']:.3f} bits effective")
        P(f"     most common convention    : {conv_str(top_conv)}  {ci_text(top_count, k)}")
        P(f"     Good-Turing coverage      : {100 * res['coverage']:.1f}% (chance the next success is a new "
          f"convention: {100 * (1 - res['coverage']):.1f}%)")
        summary(cond.label, name, "distinct_conventions", len(exact), note=f"{k} successful runs")
        summary(cond.label, name, "convention_entropy_bits", H)
        summary(cond.label, name, "dominant_convention_share", top_count / k, wilson(top_count, k), note=conv_str(top_conv))
        if space["n_optimal"]:
            K = space["n_optimal"]
            H0, T0, D0 = uniform_convention_null(k, K, rng)
            tests = {"entropy below uniform": mc_pvalue(H0, H, greater=False),
                     "dominant share above uniform": mc_pvalue(T0, top_count, greater=True),
                     "fewer distinct than uniform": mc_pvalue(D0, len(exact), greater=False)}
            adjusted = holm(tests)
            res["uniform_p"] = adjusted["entropy below uniform"]
            P(f"     H0: all {K} optimal conventions equally likely (uniform would give H = {H0.mean():.2f} bits, "
              f"{D0.mean():.1f} distinct)")
            for t, p in tests.items():
                P(f"        {t:30s}: p = {p:.4g}, Holm p = {adjusted[t]:.4g}")
                summary(cond.label, name, f"uniform_test:{t}", None, p=adjusted[t])
        P("     families among successful runs (expected share if every optimal convention were equally likely):")
        family_p = {}
        for fam, count in res["family_success"].most_common():
            expected = space["families"].get(fam, 0) / K if space["families"] and space["n_optimal"] else None
            line = f"        {fam:18s}: {ci_text(count, k)}"
            if expected is not None:
                if 0 < expected < 1:
                    family_p[fam] = binom_test(count, k, expected)
                    line += f"   expected {100 * expected:.1f}%"
                else:
                    line += f"   expected {100 * expected:.1f}% (no test)"
            P(line)
            summary(cond.label, name, f"family_share:{fam}", count / k, wilson(count, k))
        if family_p:
            adjusted = holm(family_p)
            P("        binomial tests against those expected shares (Holm): " + ", ".join(
                f"{fam} p = {adjusted[fam]:.3g}" for fam in family_p))
        P("     most common successful conventions:")
        first_seen = {}
        for r in wins:
            first_seen.setdefault(r["s_ids"], r)
        for conv, c in exact.most_common(6):
            P(f"        {conv_str(conv):20s} {c:4d} runs  [{first_seen[conv]['family']}]  "
              f"{convention_text(conv, cond.n)}")
    else:
        P("     no run reached the ceiling")

    misses = [r for r in runs if not r["success"]]
    if misses:
        P(f"  D. RUNS BELOW THE CEILING ({len(misses)})")
        P("     families: " + ", ".join(f"{f} {c}" for f, c in Counter(r["family"] for r in misses).most_common()))
        for sig, c in Counter(r["signature"] for r in misses).most_common(4):
            P(f"        {c:4d} runs  {sig}")
        for conv, c in Counter(r["s_ids"] for r in misses).most_common(3):
            example = next(r for r in misses if r["s_ids"] == conv)
            P(f"        {conv_str(conv):20s} {c:4d} runs  R = {example['reward']:.4f}  {convention_text(conv, cond.n)}")

    P("  E. INFORMATION AND ROLES (means over all runs, exact)")
    channel = np.mean([r["channel"] for r in runs], axis=0)
    reduction = np.mean([r["entropy_reduction"] for r in runs], axis=0)
    sorted_acc = np.mean([sorted(r["accuracies"], reverse=True) for r in runs], axis=0)
    res["sorted_accuracy"] = sorted_acc
    res["needed"] = float(np.mean([len(r["needed"]) for r in runs]))
    res["channel"] = float(channel.mean())
    res["entropy_reduction"] = float(reduction.mean())
    P("     channel use H(M_s), bits   : " + " ".join(f"S{i}={v:.3f}" for i, v in enumerate(channel)))
    P("     entropy reduction I(o;x_g) : " + " ".join(f"G{i}={v:.3f}" for i, v in enumerate(reduction)))
    P("     guesser accuracy, best to worst guesser: " + " ".join(f"{v:.3f}" for v in sorted_acc))
    P(f"     signallers actually needed : {res['needed']:.2f} of {cond.m}")
    return res


def crossplay(cond, runs, ceil):
    g = game(cond.m, cond.n)
    out = {}
    for mode, pool in (("all", runs), ("successful", [r for r in runs if r["success"]])):
        chosen = sorted(pool, key=lambda r: r["seed"])[:CROSSPLAY_RUNS]
        if len(chosen) < 2:
            continue
        matrix = np.array([[g.reward(b["s_ids"], a["g_ids"]) for b in chosen] for a in chosen])
        off = matrix[~np.eye(len(chosen), dtype=bool)]
        out[mode] = dict(matrix=matrix, seeds=[r["seed"] for r in chosen], self=float(np.diag(matrix).mean()),
                         cross=float(off.mean()), values=Counter(np.round(off, 4).tolist()),
                         share_at_ceiling=float(np.mean(off >= ceil - EPS)))
    return out


def analyse_condition(cond, by_variant, plain, rng) -> dict:
    ceil = ceiling(cond.m, cond.n)
    total, decay, every = schedule(cond)
    g = game(cond.m, cond.n)
    print(f"\n\n{'#' * 78}\n# {cond.label}  {cond.name}   {cond.m} signaller(s), {cond.n} guesser(s), "
          f"{cond.m * cond.n} edges, rho = m/n = {cond.rho:.2f}"
          f"{'   (' + cond.note + ')' if cond.note else ''}\n# counterpart: {cond.counterpart} "
          f"{BY_LABEL[cond.counterpart].name}\n{'#' * 78}")
    print(f"training: {total} episodes, epsilon 1.0 -> 0.05 over {decay}, snapshots every {every}, "
          f"policy table entries {cond.table_entries}")

    base = verified_baselines(cond)
    track = tracking_bound(cond.m, cond.n)
    print("\nBASELINES (team reward)")
    print(f"  random play               : 0.5000 exact expectation; {base['random']:.4f} over 10,000 played episodes")
    print("  no communication          : 0.5000 (a constant guess is right half the time)")
    print(f"  best pure tracking code   : {track:.4f} ({min(cond.m, cond.n)} guesser(s) served, any others at chance)")
    print(f"  oracle (best possible)    : {base['oracle']:.4f} exact; {base['oracle_sampled']:.4f} over 2,000 played episodes")
    print(f"  oracle codebook           : {' '.join(''.join(ITEM_LETTER[b] for b in w) for w in base['code'])}")
    print("  oracle play (items > lights > guesses, ! = wrong): " + " | ".join(base["oracle_lines"]))

    space = convention_space(cond, ceil, rng)
    how = "all enumerated" if space["enumerated"] else f"estimated from {space['sampled']:,} random conventions"
    print(f"\nCONVENTION SPACE: {space['total']:,} joint signaller conventions ({how}), "
          f"each scored with its best possible guessers")
    for value, count in sorted(space["bands"].items(), reverse=True):
        print(f"  R = {value:.4f}: {count:>12,}  ({100 * count / space['sampled']:.3f}%)")
    if space["n_optimal"] is not None:
        print(f"  conventions reaching the ceiling: {space['n_optimal']:,}; by family: " + ", ".join(
            f"{f} {c:,}" for f, c in space["families"].most_common()))
    else:
        print("  conventions reaching the ceiling: too many joint rules to enumerate; "
              "learned conventions are classified structurally instead")

    results = {}
    for variant, runs in by_variant.items():
        results[variant] = analyse_group(cond, variant, runs, space, ceil, rng)
        results[variant]["crossplay"] = crossplay(cond, runs, ceil)
    pooled = [r for runs in by_variant.values() for r in runs]
    results["POOLED"] = analyse_group(cond, "POOLED", pooled, space, ceil, rng)

    print(f"\n  --- {cond.label} {cond.name}: do the variants differ? ---")
    labels = [r["variant"] for r in pooled]
    if len(by_variant) > 1:
        chi, p, _t = permutation_homogeneity(labels, [r["success"] for r in pooled], rng, sims=PERM_SIMS)
        print(f"  success rate: permutation chi2 = {chi:.2f}, p = {p:.3f}  (H0: same rate in every variant)")
        results["variant_rate_p"] = p
        summary(cond.label, "all", "variant_rate_homogeneity", chi, p=p)
        wins = [r for r in pooled if r["success"]]
        if len({r["variant"] for r in wins}) > 1 and len({r["family"] for r in wins}) > 1:
            chi, p, _t = permutation_homogeneity([r["variant"] for r in wins], [r["family"] for r in wins], rng, sims=PERM_SIMS)
            print(f"  family mix of successful runs: permutation chi2 = {chi:.2f}, p = {p:.3f}")
            summary(cond.label, "all", "variant_family_homogeneity", chi, p=p)
        pairs = {}
        for a, b in itertools.combinations(by_variant, 2):
            ra, rb = results[a], results[b]
            pairs[f"{a} vs {b}"] = fisher_exact(ra["k"], ra["n"] - ra["k"], rb["k"], rb["n"] - rb["k"])
        adjusted = holm(pairs)
        print("  pairwise Fisher tests on success (Holm): " + ", ".join(f"{k} p = {adjusted[k]:.3f}" for k in pairs))

    print(f"\n  --- {cond.label} {cond.name}: cross-play (exact; rows = guessers' run, columns = signallers' run) ---")
    for variant in by_variant:
        for mode, c in results[variant]["crossplay"].items():
            values = "  ".join(f"{v:.3f}: {n}" for v, n in sorted(c["values"].items(), reverse=True))
            print(f"  {variant:12s} {mode:10s} {len(c['seeds']):2d} runs: self-play {c['self']:.3f}, cross-play "
                  f"{c['cross']:.3f}, gap {c['self'] - c['cross']:.3f}; {100 * c['share_at_ceiling']:.1f}% of "
                  f"cross pairs reach the ceiling")
            print(f"      cross-play cell values: {values}")
            summary(cond.label, variant, f"crossplay_{mode}:self", c["self"])
            summary(cond.label, variant, f"crossplay_{mode}:cross", c["cross"])

    same = sum(plain.get((v, r["seed"]), {}).get("weights") == r["weights"]
               for v, runs in by_variant.items() for r in runs if (v, r["seed"]) in plain)
    print(f"\n  reproducibility: {same}/{len(plain)} uninstrumented re-runs gave bit-identical network weights")
    results["repro"] = (same, len(plain))

    print(f"\n  --- {cond.label} {cond.name}: policies during training (first {SNAPSHOT_DETAIL_SEEDS} seeds; "
          f"each world is items > lights > guesses, ! = wrong guess) ---")
    stages = [(0, "early"), (SNAPSHOTS_PER_RUN // 8 - 1, "1/8"), (SNAPSHOTS_PER_RUN // 4 - 1, "1/4"),
              (SNAPSHOTS_PER_RUN // 2 - 1, "1/2"), (PRIMARY_SNAPSHOT - 1, "2/3"), (SNAPSHOTS_PER_RUN - 1, "final")]
    for variant, runs in by_variant.items():
        for r in sorted(runs, key=lambda x: x["seed"])[:SNAPSHOT_DETAIL_SEEDS]:
            print(f"\n  {variant} seed {r['seed']}")
            for index, tag in stages:
                e, s_ids, g_ids, rew = r["snaps"][index]
                print(f"    [{tag:5s}] ep {e:6d} R={rew:.4f} {family_of(s_ids, cond.m, cond.n):16s} "
                      f"{convention_text(s_ids, cond.n)}")
                print(f"            guessers: {guesser_text(g_ids, cond.m)}")
                print("            " + " | ".join(world_lines(g, s_ids, g_ids)))

    print(f"\n  --- {cond.label} {cond.name}: per-run records ---")
    for variant, runs in by_variant.items():
        for r in sorted(runs, key=lambda x: x["seed"]):
            mark = "OPTIMAL" if r["success"] else "below ceiling"
            lock = r["reward_lockin"] if r["reward_lockin"] is not None else "-"
            print(f"  {variant:11s} seed {r['seed']:3d} blk{r['block']} R={r['reward']:.4f} {mark:13s} "
                  f"{r['family']:16s} conv={conv_str(r['s_ids'])} lock-in={lock} {r['dynamics']} ({r['seconds']:.0f}s)")
            print(f"      {convention_text(r['s_ids'], cond.n)}  ||  {guesser_text(r['g_ids'], cond.m)}")
            print(f"      accuracy per guesser: {' '.join(f'{a:.3f}' for a in r['accuracies'])}; "
                  f"needed signallers: {r['needed']}; train R by quarter: "
                  f"{' '.join(f'{x:.3f}' for x in r['train_reward'])}")
            print(f"      traj: {trajectory_text(r)}")
    results.update(space=space, baselines=base, ceiling=ceil, tracking=track)
    return results


# --- cross-graph analysis ------------------------------------------------------


def score_row(res, variants):
    return "  ".join(f"{res[v]['mean']:.3f}/{100 * res[v]['rate']:5.1f}%" for v in variants)


def cross_graph_report(all_results, rng) -> None:
    conds = [c for c in ALL_CONDITIONS if c.label in all_results]
    variants = [v for v in VARIANTS]
    print(f"\n\n{'=' * 78}\nFAMILY A: ALL GRAPHS, ORDERED BY RATIO rho = signallers / guessers\n{'=' * 78}")
    print("Each variant column is mean final reward / share of runs reaching the ceiling.")
    print(f"{'cond':5s}{'graph':6s}{'rho':>5s}{'random':>7s}{'track':>7s}{'oracle':>8s}   " +
          "   ".join(f"{v:^14s}" for v in variants))
    for c in conds:
        r = all_results[c.label]
        print(f"{c.label:5s}{c.name:6s}{c.rho:5.2f}{0.5:7.3f}{r['tracking']:7.3f}{r['ceiling']:8.4f}   {score_row(r, variants)}")

    print("\nShare of the achievable gain realised, (mean reward - 0.5) / (oracle - 0.5), and top family of successes")
    for c in conds:
        r = all_results[c.label]
        parts = []
        for v in variants:
            fam = r[v]["family_success"].most_common(1)
            parts.append(f"{v} {100 * r[v]['normalised']:5.1f}% {fam[0][0] if fam else '-'}")
        print(f"  {c.label:4s}{c.name:6s} " + " | ".join(parts))

    print(f"\n{'=' * 78}\nCOUNTERPARTS: K_(a,b) AGAINST K_(b,a) (identical policy-table size and schedule)\n{'=' * 78}")
    seen = set()
    for c in conds:
        other = BY_LABEL[c.counterpart]
        if c.label == other.label or c.label in seen or other.label not in all_results:
            continue
        seen.update((c.label, other.label))
        ra, rb = all_results[c.label], all_results[other.label]
        print(f"\n{c.label} {c.name} (ceiling {ra['ceiling']:.4f}) vs {other.label} {other.name} (ceiling {rb['ceiling']:.4f})")
        for v in variants + ["POOLED"]:
            a, b = ra[v], rb[v]
            p = fisher_exact(a["k"], a["n"] - a["k"], b["k"], b["n"] - b["k"])
            print(f"  {v:12s} success {100 * a['rate']:5.1f}% vs {100 * b['rate']:5.1f}% (Fisher p = {p:.3g}); "
                  f"mean R {a['mean']:.4f} vs {b['mean']:.4f}; gain realised {100 * a['normalised']:.1f}% vs "
                  f"{100 * b['normalised']:.1f}%; median lock-in {a['lock_median']:.0f} vs {b['lock_median']:.0f}; "
                  f"signallers needed {a['needed']:.2f}/{c.m} vs {b['needed']:.2f}/{other.m}")
            summary(f"{c.label}|{other.label}", v, "counterpart_success_fisher", a["rate"] - b["rate"], p=p)
        for label, res in ((c.label, ra), (other.label, rb)):
            print(f"  {label}: successful families (pooled) " + ", ".join(
                f"{f} {n}" for f, n in res["POOLED"]["family_success"].most_common()))

    print(f"\n{'=' * 78}\nSCALE PAIRS: SAME RATIO, MORE AGENTS\n{'=' * 78}")
    for a_label, b_label in SCALE_PAIRS:
        if a_label not in all_results or b_label not in all_results:
            continue
        ra, rb = all_results[a_label], all_results[b_label]
        print(f"{a_label} {BY_LABEL[a_label].name} vs {b_label} {BY_LABEL[b_label].name} (rho = {BY_LABEL[a_label].rho:.2f})")
        for v in variants + ["POOLED"]:
            a, b = ra[v], rb[v]
            p = fisher_exact(a["k"], a["n"] - a["k"], b["k"], b["n"] - b["k"])
            print(f"  {v:12s} success {100 * a['rate']:5.1f}% vs {100 * b['rate']:5.1f}% (Fisher p = {p:.3g}); "
                  f"gain realised {100 * a['normalised']:.1f}% vs {100 * b['normalised']:.1f}%")

    print(f"\n{'=' * 78}\nTRENDS ACROSS THE {len(conds)} GRAPHS (Spearman rank correlation, permutation p)\n{'=' * 78}")
    if len(conds) >= 4:
        predictors = {"ratio rho": [c.rho for c in conds], "edges m*n": [c.m * c.n for c in conds],
                      "policy-table entries": [c.table_entries for c in conds]}
        for v in variants + ["POOLED"]:
            for outcome, key in (("success rate", "rate"), ("share of gain realised", "normalised")):
                y = [all_results[c.label][v][key] for c in conds]
                texts = []
                for name, x in predictors.items():
                    r, p = spearman(x, y, rng)
                    texts.append(f"{name}: r = {r:+.2f} (p = {p:.3f})")
                    summary("family", v, f"spearman:{outcome}~{name}", r, p=p)
                print(f"  {v:12s} {outcome:24s} " + "; ".join(texts))


# --- output files ------------------------------------------------------------


def write_csvs(all_runs) -> None:
    with open(os.path.join(OUT_DIR, "runs.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["condition", "graph", "signallers", "guessers", "rho", "variant", "seed", "block", "ceiling",
                    "final_reward", "gap_to_oracle", "success", "reward_at_two_thirds", "success_at_two_thirds",
                    "first_success_ep", "reward_lockin_ep", "convention_lockin_ep", "convention_changes",
                    "changes_after_decay", "dynamics", "family", "signature", "signaller_ids", "guesser_ids",
                    "effective_convention", "signaller_rules", "guesser_rules", "needed_signallers",
                    "guesser_accuracy", "channel_use", "entropy_reduction", "train_reward_by_quarter",
                    "weights_sha256_16", "seconds"])
        for r in all_runs:
            c = BY_LABEL[r["label"]]
            ceil = ceiling(c.m, c.n)
            w.writerow([c.label, c.name, c.m, c.n, f"{c.rho:.4f}", r["variant"], r["seed"], r["block"], f"{ceil:.6f}",
                        f"{r['reward']:.6f}", f"{ceil - r['reward']:.6f}", int(r["success"]), f"{r['reward_primary']:.6f}",
                        int(r["success_primary"]), r["first_success"], r["reward_lockin"], r["convention_lockin"],
                        r["changes"], r["changes_after_decay"], r["dynamics"], r["family"], r["signature"],
                        conv_str(r["s_ids"]), conv_str(r["g_ids"]), conv_str(r["effective"]),
                        convention_text(r["s_ids"], c.n), guesser_text(r["g_ids"], c.m),
                        " ".join(f"S{s}" for s in r["needed"]), " ".join(f"{a:.4f}" for a in r["accuracies"]),
                        " ".join(f"{a:.4f}" for a in r["channel"]), " ".join(f"{a:.4f}" for a in r["entropy_reduction"]),
                        " ".join(f"{a:.4f}" for a in r["train_reward"]), r["weights"], f"{r['seconds']:.1f}"])
    with open(os.path.join(OUT_DIR, "trajectories.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["condition", "graph", "variant", "seed", "episode", "signaller_ids", "exact_reward", "family"])
        for r in all_runs:
            c = BY_LABEL[r["label"]]
            for e, s_ids, reward in trajectory_points(r):
                w.writerow([c.label, c.name, r["variant"], r["seed"], e, conv_str(s_ids), f"{reward:.6f}",
                            family_of(s_ids, c.m, c.n)])
    with open(os.path.join(OUT_DIR, "summary.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["condition", "variant", "metric", "value", "ci_low", "ci_high", "p_value", "note"])
        w.writerows(SUMMARY_ROWS)


# --- figures -------------------------------------------------------------------

VARIANT_COLOR = {"shared": "#2a78d6", "two_brain": "#eb6834", "independent": "#1baf7a"}
FAMILY_COLOR = {"tracking": "#2a78d6", "XOR-type": "#eb6834", "nonlinear": "#1baf7a",
                "partial tracking": "#eda100", "silent": "#e87ba4"}
DYNAMICS = ["held optimum", "escaped", "left optimum", "frozen", "oscillating", "drifting"]
DYNAMICS_COLOR = dict(zip(DYNAMICS, ["#2a78d6", "#1baf7a", "#e87ba4", "#8f8e88", "#eb6834", "#eda100"]))
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def _style(ax, grid="y"):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    if grid:
        ax.grid(axis=grid, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def _save(fig, *path):
    target = os.path.join(FIG_DIR, *path)
    os.makedirs(os.path.dirname(target), exist_ok=True)
    fig.savefig(target, dpi=150, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)


def _folder(c):
    return f"{c.label}_K{c.m}_{c.n}"


def _reference_lines(ax, res, x_right=None):
    same = abs(res["ceiling"] - res["tracking"]) < EPS
    lines = [(res["ceiling"], "oracle = tracking only" if same else "oracle", "-")]
    if not same:
        lines.append((res["tracking"], "tracking only", ":"))
    for value, text, style in lines + [(0.5, "random", "--")]:
        ax.axhline(value, color=INK2, linestyle=style, linewidth=1)
        if x_right is not None:
            ax.text(x_right, value, f" {text} {value:.3f}", fontsize=7, color=INK2, va="bottom", ha="right")


def policy_figure(c, s_ids, g_ids, title, path):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    g = game(c.m, c.n)
    item_color = {0: "#9ecae1", 1: "#fdae61"}
    light_color = {0: "#d8d7d2", 1: "#ffd23f"}
    item_word, light_word = {0: "CAT", 1: "DOG"}, {0: "OFF", 1: "ON"}
    columns = c.n + c.m + c.n
    rows = len(g.worlds)
    used = {g.lights(s_ids, w) for w in range(rows)}
    fig, (left, right) = plt.subplots(1, 2, figsize=(0.62 * (columns + c.m + c.n) + 2.4, 0.34 * max(rows, len(g.patterns)) + 1.8),
                                      gridspec_kw={"width_ratios": [columns + 0.8, c.m + c.n + 0.4]})

    def cell(ax, x, y, color, text, wrong=False):
        ax.add_patch(Rectangle((x, y), 1, 1, facecolor=color, edgecolor="white", linewidth=1.5,
                               hatch="////" if wrong else None))
        ax.text(x + 0.5, y + 0.5, text, ha="center", va="center", fontsize=7, color=INK,
                fontweight="bold" if wrong else "normal")

    for row, items in enumerate(g.worlds):
        y = rows - 1 - row
        lights = g.lights(s_ids, row)
        guesses = g.guesses(g_ids, lights)
        for i, v in enumerate(items):
            cell(left, i, y, item_color[v], item_word[v])
        for s, v in enumerate(lights):
            cell(left, c.n + 0.4 + s, y, light_color[v], light_word[v])
        for i, v in enumerate(guesses):
            wrong = v != items[i]
            cell(left, c.n + c.m + 0.8 + i, y, item_color[v], item_word[v] + (" x" if wrong else ""), wrong)
    heads = ([(i + 0.5, f"G{i} item") for i in range(c.n)] + [(c.n + 0.9 + s, f"S{s}") for s in range(c.m)]
             + [(c.n + c.m + 1.3 + i, f"G{i} says") for i in range(c.n)])
    for x, text in heads:
        left.text(x, rows + 0.15, text, ha="center", va="bottom", fontsize=7, color=INK2)
    left.set_xlim(0, columns + 0.8)
    left.set_ylim(0, rows + 0.8)
    left.set_title("What happens in every possible world", fontsize=9, color=INK, loc="left")
    n_rows = len(g.patterns)
    for row, pattern in enumerate(g.patterns):
        y = n_rows - 1 - row
        reachable = pattern in used
        for s, v in enumerate(pattern):
            cell(right, s, y, light_color[v] if reachable else "#f1f0ec", light_word[v] if reachable else "-")
        guesses = g.guesses(g_ids, pattern)
        for i, v in enumerate(guesses):
            cell(right, c.m + 0.4 + i, y, item_color[v] if reachable else "#f1f0ec", item_word[v] if reachable else "-")
    for s in range(c.m):
        right.text(s + 0.5, n_rows + 0.15, f"S{s}", ha="center", va="bottom", fontsize=7, color=INK2)
    for i in range(c.n):
        right.text(c.m + 0.9 + i, n_rows + 0.15, f"G{i}", ha="center", va="bottom", fontsize=7, color=INK2)
    right.set_xlim(0, c.m + c.n + 0.4)
    right.set_ylim(0, n_rows + 0.8)
    right.set_title("How guessers decode lights", fontsize=9, color=INK, loc="left")
    for ax in (left, right):
        ax.axis("off")
    fig.suptitle(title, fontsize=9, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    _save(fig, *path)


def exact_action_response(c, s_ids, g_ids):
    g = game(c.m, c.n)
    table = {(s, x): {(a, b): 0.0 for a in (0, 1) for b in (0, 1)} for s in range(c.m) for x in range(c.n)}
    for w in range(len(g.worlds)):
        lights = g.lights(s_ids, w)
        guesses = g.guesses(g_ids, lights)
        for (s, x), cells in table.items():
            cells[(lights[s], guesses[x])] += 1 / len(g.worlds)
    return table


def training_action_response_figure(c, rec, title, path):
    import matplotlib.pyplot as plt
    edges = list(game(c.m, c.n).graph.edges)
    fig, axes = plt.subplots(AR_WINDOWS, len(edges), figsize=(1.28 * len(edges) + 1.5, 1.28 * AR_WINDOWS + 1.0), squeeze=False)
    total, _decay, _every = schedule(c)
    im = None
    for w, window in enumerate(rec["ar"]):
        for e, counts in enumerate(window):
            ax = axes[w][e]
            n = max(sum(counts), 1)
            grid = np.array(counts, dtype=float).reshape(2, 2) * 100 / n
            im = ax.imshow(grid, cmap="Blues", vmin=0, vmax=100)
            for a in range(2):
                for b in range(2):
                    ax.text(b, a, f"{grid[a, b]:.0f}", ha="center", va="center", fontsize=7,
                            color="white" if grid[a, b] > 55 else INK)
            ax.set_xticks([0, 1], ["CAT", "DOG"] if w == AR_WINDOWS - 1 else ["", ""], fontsize=6)
            ax.set_yticks([0, 1], ["OFF", "ON"] if e == 0 else ["", ""], fontsize=6)
            ax.tick_params(length=0)
            if w == 0:
                ax.set_title(f"S{edges[e][0]}→G{edges[e][1]}", fontsize=8, color=INK)
        start, end = w * total // AR_WINDOWS, (w + 1) * total // AR_WINDOWS
        axes[w][0].set_ylabel(f"ep {start}-{end}", fontsize=7, color=INK2)
    fig.suptitle(title, fontsize=9, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 0.94, 0.97))
    fig.colorbar(im, cax=fig.add_axes((0.95, 0.25, 0.012, 0.5)), label="% of training episodes")
    _save(fig, *path)


def representative(runs):
    """A run with the most common convention among the best outcomes, and one failure."""
    wins = [r for r in runs if r["success"]]
    pool = wins or [r for r in runs if abs(r["reward"] - max(x["reward"] for x in runs)) < EPS]
    conv = Counter(r["s_ids"] for r in pool).most_common(1)[0][0]
    best = min((r for r in pool if r["s_ids"] == conv), key=lambda r: r["seed"])
    misses = [r for r in runs if not r["success"]]
    failure = None
    if misses:
        conv = Counter(r["s_ids"] for r in misses).most_common(1)[0][0]
        failure = min((r for r in misses if r["s_ids"] == conv), key=lambda r: r["seed"])
    return best, failure


def condition_figures(c, res, by_variant):
    import matplotlib.pyplot as plt
    from gsg.visualization import render_action_response_matrix, render_matplotlib
    folder = _folder(c)
    total, decay, every = schedule(c)
    variants = list(by_variant)
    title = (f"{c.label} {c.name}: oracle {res['ceiling']:.3f}, tracking only {res['tracking']:.3f}, random 0.500")

    fig, ax = plt.subplots(figsize=(4.6, 1.2 + 1.0 * max(c.m, c.n)))
    render_matplotlib(game(c.m, c.n).graph, ax=ax, title=f"{c.label} {c.name}, rho = {c.rho:.2f}")
    _save(fig, folder, "topology.png")

    episodes = (np.arange(SNAPSHOTS_PER_RUN) + 1) * every
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.9))
    for v in variants:
        runs = by_variant[v]
        rewards = np.array([[s[3] for s in r["snaps"]] for r in runs])
        mean, half = rewards.mean(0), 1.96 * rewards.std(0, ddof=1) / math.sqrt(len(runs)) if len(runs) > 1 else 0
        axes[0].plot(episodes, mean, color=VARIANT_COLOR[v], linewidth=2, label=v)
        axes[0].fill_between(episodes, mean - half, mean + half, color=VARIANT_COLOR[v], alpha=0.15, linewidth=0)
        axes[1].plot(episodes, (rewards >= res["ceiling"] - EPS).mean(0), color=VARIANT_COLOR[v], linewidth=2, label=v)
        moved = np.array([[a[1] != b[1] for a, b in zip(r["snaps"], r["snaps"][1:])] for r in runs], dtype=float)
        axes[2].plot(episodes[1:], moved.mean(0), color=VARIANT_COLOR[v], linewidth=2, label=v)
    _reference_lines(axes[0], res, total)
    axes[0].set_title("Exact greedy reward (mean, 95% CI)", fontsize=9, loc="left", color=INK)
    axes[1].set_title("Share of runs at the ceiling", fontsize=9, loc="left", color=INK)
    axes[1].set_ylim(0, 1.03)
    axes[2].set_title(f"Runs whose convention changed since the last snapshot", fontsize=9, loc="left", color=INK)
    axes[2].set_ylim(0, 1.03)
    for ax in axes:
        ax.axvline(decay, color=GRID, linewidth=1.2)
        ax.set_xlabel("training episode (vertical line: epsilon reaches 0.05)", fontsize=8, color=INK2)
        _style(ax)
    axes[1].legend(fontsize=8, frameon=False)
    fig.suptitle(title, fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    _save(fig, folder, "learning_curves.png")

    bands = sorted({b for v in variants for b in res[v]["bands"]}, reverse=True)
    shades = plt.cm.Blues(np.linspace(0.85, 0.3, max(len(bands), 1)))
    fig, ax = plt.subplots(figsize=(8.5, 0.55 * len(variants) + 1.5))
    for y, v in enumerate(variants):
        left = 0
        for b, color in zip(bands, shades):
            width = res[v]["bands"].get(b, 0) / res[v]["n"]
            ax.barh(y, width, left=left, color=color, edgecolor="white", linewidth=1.5,
                    label=f"R = {b:.3f}" if y == 0 else None)
            if width > 0.06:
                ax.text(left + width / 2, y, f"{100 * width:.0f}%", ha="center", va="center", fontsize=7,
                        color="white" if color[:3].mean() < 0.55 else INK)
            left += width
    ax.set_yticks(range(len(variants)), variants)
    ax.invert_yaxis()
    ax.set_xlabel("share of runs", fontsize=8, color=INK2)
    ax.legend(fontsize=7, frameon=False, ncol=min(5, len(bands)), loc="lower center", bbox_to_anchor=(0.5, 1.0))
    _style(ax, grid=None)
    _save(fig, folder, "reward_bands.png")

    fig, axes = plt.subplots(len(variants), 1, figsize=(10, 2.1 * len(variants)), squeeze=False)
    for ax, v in zip(axes[:, 0], variants):
        wins = [r for r in by_variant[v] if r["success"]]
        ax.set_title(f"{v}: {len(wins)} of {len(by_variant[v])} runs at the ceiling", fontsize=9, loc="left", color=INK)
        _style(ax)
        if not wins:
            continue
        family = {r["s_ids"]: r["family"] for r in wins}
        top = Counter(r["s_ids"] for r in wins).most_common(16)
        shares = [count / len(wins) for _conv, count in top]
        ax.bar(range(len(top)), shares, color=[FAMILY_COLOR[family[conv]] for conv, _count in top],
               edgecolor="white", linewidth=1.5, width=0.75)
        ax.set_xticks(range(len(top)), [conv_str(conv) for conv, _count in top], rotation=35, ha="right", fontsize=6)
        ax.set_ylabel("share of successes", fontsize=8, color=INK2)
    from matplotlib.patches import Patch
    axes[0, 0].legend(handles=[Patch(color=FAMILY_COLOR[f], label=f) for f in FAMILIES], fontsize=7, frameon=False,
                      ncol=5, loc="upper right")
    fig.tight_layout()
    _save(fig, folder, "convention_distribution.png")

    for v in variants:
        best, failure = representative(by_variant[v])
        policy_figure(c, best["s_ids"], best["g_ids"],
                      f"{c.name} {v}: most common best convention, seed {best['seed']}, R = {best['reward']:.3f}, {best['family']}",
                      (folder, f"policy_{v}_best.png"))
        fig = render_action_response_matrix(exact_action_response(c, best["s_ids"], best["g_ids"]),
                                            title=f"{c.name} {v}: settled policy, seed {best['seed']}, R = {best['reward']:.3f}")
        _save(fig, folder, f"action_response_{v}_best.png")
        training_action_response_figure(c, best, f"{c.name} {v}: signal-guess frequencies during training, seed {best['seed']} (exploration on)",
                                        (folder, f"action_response_training_{v}_best.png"))
        if failure is not None:
            policy_figure(c, failure["s_ids"], failure["g_ids"],
                          f"{c.name} {v}: most common convention below the ceiling, seed {failure['seed']}, "
                          f"R = {failure['reward']:.3f}, {failure['family']}", (folder, f"policy_{v}_below_ceiling.png"))
            fig = render_action_response_matrix(exact_action_response(c, failure["s_ids"], failure["g_ids"]),
                                                title=f"{c.name} {v}: settled policy below the ceiling, seed {failure['seed']}, R = {failure['reward']:.3f}")
            _save(fig, folder, f"action_response_{v}_below_ceiling.png")

    modes = ("all", "successful")
    fig, axes = plt.subplots(2, len(variants), figsize=(4.5 * len(variants), 8.4), squeeze=False)
    for j, v in enumerate(variants):
        for i, mode in enumerate(modes):
            ax = axes[i][j]
            data = res[v]["crossplay"].get(mode)
            if data is None:
                ax.axis("off")
                ax.set_title(f"{v}, {mode}: fewer than 2 runs", fontsize=8)
                continue
            matrix = data["matrix"]
            im = ax.imshow(matrix, cmap="viridis", vmin=0, vmax=1)
            for a in range(len(matrix)):
                for b in range(len(matrix)):
                    ax.text(b, a, f"{matrix[a, b]:.2f}"[1:] if matrix[a, b] < 1 else "1", ha="center", va="center",
                            fontsize=4.5, color="white" if matrix[a, b] < 0.62 else "black")
            ax.set_xticks(range(len(matrix)), data["seeds"], fontsize=5, rotation=90)
            ax.set_yticks(range(len(matrix)), data["seeds"], fontsize=5)
            ax.set_title(f"{v}, {mode} runs: self {data['self']:.3f}, cross {data['cross']:.3f}", fontsize=8, loc="left", color=INK)
            ax.set_xlabel("signallers from seed", fontsize=7, color=INK2)
            ax.set_ylabel("guessers from seed", fontsize=7, color=INK2)
    fig.suptitle(f"{c.label} {c.name}: cross-play, exact team reward", fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    _save(fig, folder, "crossplay.png")


def summary_figures(all_results):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    conds = [c for c in ALL_CONDITIONS if c.label in all_results]
    labels = [f"{c.name}\nρ={c.rho:.2f}" for c in conds]
    x = np.arange(len(conds))
    width = 0.25
    offsets = {v: (i - (len(VARIANTS) - 1) / 2) * width for i, v in enumerate(VARIANTS)}

    def grouped(ax, value, interval):
        for v in VARIANTS:
            ys = np.array([value(all_results[c.label][v]) for c in conds])
            low, high = zip(*[interval(all_results[c.label][v]) for c in conds])
            ax.bar(x + offsets[v], ys, width * 0.86, color=VARIANT_COLOR[v], label=v)
            ax.errorbar(x + offsets[v], ys, yerr=np.clip([ys - np.array(low), np.array(high) - ys], 0, None),
                        fmt="none", ecolor=INK2, elinewidth=0.8, capsize=1.5)
        ax.set_xticks(x, labels, fontsize=7)
        ax.set_xlim(-0.6, len(conds) - 0.4)
        _style(ax)

    fig, ax = plt.subplots(figsize=(13, 4.4))
    grouped(ax, lambda r: r["mean"], lambda r: r["mean_ci"])
    for i, c in enumerate(conds):
        r = all_results[c.label]
        ax.hlines(r["ceiling"], i - 0.42, i + 0.42, color=INK, linewidth=1.8, label="oracle" if i == 0 else None)
        ax.hlines(r["tracking"], i - 0.42, i + 0.42, color=INK2, linewidth=1.2, linestyles=":", label="tracking only" if i == 0 else None)
    ax.axhline(0.5, color=INK2, linestyle="--", linewidth=1, label="random")
    ax.set_ylim(0.45, 1.04)
    ax.set_ylabel("mean final team reward", fontsize=8, color=INK2)
    ax.legend(fontsize=7, frameon=False, ncol=6, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    _save(fig, "summary", "reward_vs_baselines.png")

    fig, ax = plt.subplots(figsize=(13, 3.9))
    grouped(ax, lambda r: r["rate"], lambda r: r["rate_ci"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("share of runs at the ceiling", fontsize=8, color=INK2)
    ax.legend(fontsize=8, frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    _save(fig, "summary", "success_rates.png")

    pairs, seen = [], set()
    for c in conds:
        other = BY_LABEL[c.counterpart]
        if c.label != other.label and other.label in all_results and c.label not in seen:
            seen.update((c.label, other.label))
            pairs.append((c, other))
    if pairs:
        fig, axes = plt.subplots(2, len(pairs), figsize=(3.1 * len(pairs), 6.2), squeeze=False)
        for j, (a, b) in enumerate(pairs):
            for i, (key, name) in enumerate((("rate", "share of runs at the ceiling"), ("normalised", "share of achievable gain"))):
                ax = axes[i][j]
                for k, cond in enumerate((a, b)):
                    for v in VARIANTS:
                        ax.bar(k + offsets[v], all_results[cond.label][v][key], width * 0.86, color=VARIANT_COLOR[v],
                               label=v if k == 0 else None)
                ax.set_xticks([0, 1], [f"{a.name}\noracle {all_results[a.label]['ceiling']:.3f}",
                                       f"{b.name}\noracle {all_results[b.label]['ceiling']:.3f}"], fontsize=7)
                ax.set_ylim(0, 1.05)
                if j == 0:
                    ax.set_ylabel(name, fontsize=8, color=INK2)
                _style(ax)
        axes[0][0].legend(fontsize=7, frameon=False)
        fig.suptitle("Counterparts: swapping the numbers of signallers and guessers", fontsize=10, color=INK, x=0.01, ha="left")
        fig.tight_layout()
        _save(fig, "summary", "counterparts.png")

    fig, axes = plt.subplots(1, len(VARIANTS), figsize=(4.3 * len(VARIANTS), 4.4), sharey=True, squeeze=False)
    for ax, v in zip(axes[0], VARIANTS):
        for y, c in enumerate(conds):
            r = all_results[c.label][v]
            left, total = 0, max(r["k"], 1)
            for fam in FAMILIES:
                width = r["family_success"].get(fam, 0) / total
                ax.barh(y, width, left=left, color=FAMILY_COLOR[fam], edgecolor="white", linewidth=1.2)
                left += width
        ax.set_yticks(range(len(conds)), [c.name for c in conds], fontsize=8)
        ax.invert_yaxis() if ax is axes[0][0] else None
        ax.set_title(v, fontsize=9, loc="left", color=INK)
        ax.set_xlabel("share of runs at the ceiling", fontsize=8, color=INK2)
        _style(ax, grid=None)
    fig.legend(handles=[Patch(color=FAMILY_COLOR[f], label=f) for f in FAMILIES], fontsize=8, frameon=False, ncol=5,
               loc="upper center")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _save(fig, "summary", "convention_families.png")

    fig, axes = plt.subplots(1, 2, figsize=(13, 3.8))
    for ax, key, name in ((axes[0], "H_effective", "entropy of successful conventions, bits (idle signallers ignored)"),
                          (axes[1], "effective_distinct", "distinct successful conventions (idle signallers ignored)")):
        for v in VARIANTS:
            ax.bar(x + offsets[v], [all_results[c.label][v].get(key, 0) for c in conds], width * 0.86, color=VARIANT_COLOR[v], label=v)
        ax.set_xticks(x, [c.name for c in conds], fontsize=7)
        ax.set_title(name, fontsize=9, loc="left", color=INK)
        _style(ax)
    axes[0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    _save(fig, "summary", "convention_diversity.png")

    fig, ax = plt.subplots(figsize=(13, 3.6))
    for v in VARIANTS:
        ys = [all_results[c.label][v]["lock_median"] / schedule(c)[1] for c in conds]
        ax.bar(x + offsets[v], ys, width * 0.86, color=VARIANT_COLOR[v], label=v)
    ax.axhline(1, color=INK2, linestyle="--", linewidth=1)
    ax.text(len(conds) - 0.45, 1, "epsilon reaches 0.05", ha="right", va="bottom", fontsize=7, color=INK2)
    ax.set_xticks(x, labels, fontsize=7)
    ax.set_ylabel("median lock-in episode / decay length", fontsize=8, color=INK2)
    ax.legend(fontsize=8, frameon=False)
    _style(ax)
    _save(fig, "summary", "lockin_times.png")

    fig, ax = plt.subplots(figsize=(11, 4.0))
    for y, c in enumerate(conds):
        counts = all_results[c.label]["POOLED"]["dynamics"]
        total, left = sum(counts.values()), 0
        for key in DYNAMICS:
            width = counts.get(key, 0) / total
            ax.barh(y, width, left=left, color=DYNAMICS_COLOR[key], edgecolor="white", linewidth=1.2)
            left += width
    ax.set_yticks(range(len(conds)), [c.name for c in conds], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("share of runs, from 2/3 of training to the end", fontsize=8, color=INK2)
    ax.legend(handles=[Patch(color=DYNAMICS_COLOR[k], label=k) for k in DYNAMICS], fontsize=7, frameon=False, ncol=6,
              loc="lower center", bbox_to_anchor=(0.5, 1.0))
    _style(ax, grid=None)
    _save(fig, "summary", "late_training_dynamics.png")

    fig, axes = plt.subplots(2, 1, figsize=(13, 6.2), sharex=True)
    for ax, mode in zip(axes, ("all", "successful")):
        for v in VARIANTS:
            for i, c in enumerate(conds):
                data = all_results[c.label][v]["crossplay"].get(mode)
                if data is None:
                    continue
                xpos = i + offsets[v]
                ax.plot([xpos, xpos], [data["cross"], data["self"]], color=VARIANT_COLOR[v], linewidth=1.5)
                ax.plot(xpos, data["self"], "o", color=VARIANT_COLOR[v], markersize=5, label=f"{v} self-play" if i == 0 else None)
                ax.plot(xpos, data["cross"], "D", markerfacecolor="white", markeredgecolor=VARIANT_COLOR[v], markersize=5,
                        label=f"{v} cross-play" if i == 0 else None)
        ax.axhline(0.5, color=INK2, linestyle="--", linewidth=1)
        ax.set_title(f"Self-play and cross-play, {mode} runs", fontsize=9, loc="left", color=INK)
        ax.set_ylabel("team reward", fontsize=8, color=INK2)
        _style(ax)
    axes[1].set_xticks(x, labels, fontsize=7)
    axes[0].legend(fontsize=7, frameon=False, ncol=6)
    fig.tight_layout()
    _save(fig, "summary", "crossplay_gaps.png")

    bottleneck = [c for c in conds if c.m < c.n]
    if bottleneck:
        fig, axes = plt.subplots(1, len(bottleneck), figsize=(2.9 * len(bottleneck), 3.5), sharey=True, squeeze=False)
        for ax, c in zip(axes[0], bottleneck):
            for v in VARIANTS:
                values = all_results[c.label][v]["sorted_accuracy"]
                ax.plot(range(1, c.n + 1), values, "o-", color=VARIANT_COLOR[v], label=v, markersize=5)
            ax.set_xticks(range(1, c.n + 1))
            ax.set_title(c.name, fontsize=9, loc="left", color=INK)
            ax.set_xlabel("guesser, best to worst", fontsize=8, color=INK2)
            _style(ax)
        axes[0][0].set_ylabel("mean accuracy", fontsize=8, color=INK2)
        axes[0][0].legend(fontsize=7, frameon=False)
        fig.suptitle("When there are too few lights, who gets served?", fontsize=10, color=INK, x=0.01, ha="left")
        fig.tight_layout()
        _save(fig, "summary", "who_gets_served.png")


def safely(what, function, *args):
    try:
        function(*args)
    except Exception:
        print(f"  WARNING: {what} failed; results above are unaffected\n{traceback.format_exc()}", flush=True)


# --- analysis driver -------------------------------------------------------------


def analyse_stage() -> None:
    import matplotlib
    matplotlib.use("Agg")
    records = load_records()
    rng = np.random.default_rng(STATS_SEED)
    all_results, all_runs = {}, []
    print("\n" + "=" * 78)
    print("HOW TO READ THIS LOG")
    print("  Items: C = CAT, D = DOG. Lights: 0 = OFF, 1 = ON. A world line 'CD > 10 > C!D' means guessers")
    print("  hold CAT, DOG; the lights are S0 ON, S1 OFF; G0 says DOG (wrong, marked !) and G1 says DOG.")
    print("  x0, x1, ... are the guessers' items. Success = reaching this graph's oracle reward.")
    print("  Families: tracking = every active light relays one item and as many items as possible are relayed;")
    print("  partial tracking = relays only, but fewer items; XOR-type = relays plus parity lights;")
    print("  nonlinear = at least one light combines items (AND/OR, majority, ...); silent = constant lights.")
    print("=" * 78, flush=True)
    for cond in CONDITIONS:
        by_variant, plain = {}, {}
        for variant in VARIANTS:
            runs = []
            for seed in range(SEED_START, SEED_START + NUM_SEEDS):
                rec = records.get(run_key(cond.label, variant, seed, "full"))
                if rec is not None:
                    runs.append(derive(rec, cond, ceiling(cond.m, cond.n)))
                other = records.get(run_key(cond.label, variant, seed, "plain"))
                if other is not None:
                    plain[(variant, seed)] = other
            if runs:
                by_variant[variant] = runs
                all_runs.extend(runs)
        if len(by_variant) != len(VARIANTS):
            print(f"\nSKIPPING {cond.label} {cond.name}: not every variant has finished runs")
            continue
        missing = NUM_SEEDS - min(len(r) for r in by_variant.values())
        if missing:
            print(f"\nNOTE: {cond.label} {cond.name} is missing up to {missing} seeds; statistics use finished runs")
        results = analyse_condition(cond, by_variant, plain, rng)
        all_results[cond.label] = results
        safely(f"figures for {cond.label}", condition_figures, cond, results, by_variant)
    if not all_results:
        print("nothing to analyse")
        return
    cross_graph_report(all_results, rng)
    same = sum(r["repro"][0] for r in all_results.values())
    checked = sum(r["repro"][1] for r in all_results.values())
    print(f"\nREPRODUCIBILITY: {same}/{checked} uninstrumented re-runs matched bit for bit")
    write_csvs(all_runs)
    safely("summary figures", summary_figures, all_results)
    print(f"\ntables in  : {os.path.normpath(OUT_DIR)}\nfigures in : {os.path.normpath(FIG_DIR)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stage", choices=("train", "analyse", "all"), default="all")
    stage = parser.parse_args().stage
    t0 = time.time()
    print(f"seeds {SEED_START}..{SEED_START + NUM_SEEDS - 1}; conditions "
          f"{' '.join(c.label for c in CONDITIONS)}; variants {' '.join(VARIANTS)}; scale {SCALE}", flush=True)
    if stage in ("train", "all"):
        train_stage()
    if stage in ("analyse", "all"):
        analyse_stage()
    print(f"\nq2_family_a.py finished in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
