"""Question 1 / Objective 4 on K_{2,2}: formal, reproducible hypothesis tests.

Research Question 1 asks whether IQL agents restricted to a single binary
signal converge on *identifiable and consistent* conventions, and what their
character is. Objective 4 asks for the empirical frequency distribution over
emergent conventions -- convention entropy and dominant-convention frequency
-- to decide whether some conventions are *structurally preferred* or whether
the outcome is *arbitrary*. The earlier experiments answered these
descriptively on P = 30 seeds; this script answers them as statistical
claims, with confidence intervals and pre-stated null hypotheses, on a much
larger sample (default N = 100 seeds per variant), and checks that the
results themselves reproduce.

Three questions, each turned into testable hypotheses
-----------------------------------------------------
Q-A  Does training always produce the SAME convention?
     H0_A1 (one convention): every perfect run lands on the same joint
           convention. Falsified deterministically by >= 2 distinct ones;
           quantified by the dominant-convention frequency and its 95% CI.
     H0_A2 (arbitrary): the perfect runs are uniform over all 24 perfect
           conventions. Tested by an exact Monte-Carlo multinomial
           goodness-of-fit test, and by the entropy / dominant-frequency
           null distributions under uniform sampling at the same n.
     H0_A3 (no compositional bias): P(tracking) = 8/24 = 1/3. Exact
           binomial test.
     H0_A4.. (symmetry): K_{2,2}'s game symmetries (swap signallers, swap
           guessers, relabel the signal, relabel the items) map conventions
           onto each other, so a symmetric learner should pick each member
           of a symmetry pair with probability 1/2: which item S0 tracks,
           which signaller carries the parity bit, ON=DOG vs ON=CAT, XOR vs
           XNOR. The *learner* is not perfectly symmetric (0/1 network
           inputs are not interchangeable), so these are genuine tests.
     All tests within a variant are Holm-Bonferroni corrected.
Q-B  How MANY different conventions does it produce?
     Observed richness, Chao1 estimate of the total reachable richness
     (with the 24-convention ceiling), Good-Turing sample coverage (the
     probability that one more run finds a convention not yet seen), and an
     accumulation curve compared with the uniform-24 expectation.
Q-C  Does it always produce the same SUCCESS RATE?
     Convergence rate with Wilson and Clopper-Pearson 95% CIs; homogeneity
     of that rate across independent blocks of seeds (permutation
     chi-square test) -- the direct "re-run it and get the same number"
     test; reward-band distribution; mean reward with a bootstrap CI;
     cross-variant comparison (permutation chi-square + Holm-corrected
     pairwise Fisher exact tests).

Plus two checks that the *measurements themselves* reproduce:
  * determinism: re-running a seed reproduces the identical convention,
    reward, full training trajectory and network weights (SHA-256);
  * non-invasive instrumentation: the mid-training snapshots taken below
    do not perturb training (a plain run without snapshots gives
    bit-identical weights at the primary endpoint).

Protocol
--------
Exactly compare_iql_variants_k22.py's: K_{2,2}, 4000 training episodes,
epsilon 1.0 -> 0.05 linearly over 2500 episodes, seed s seeds torch, Python
`random` and the env. The PRIMARY endpoint is the greedy population at
episode 4000. Training is then continued (same run, same RNG stream) to
episode 6000 purely to test *stability*: does a convention, once found,
persist? Because epsilon's schedule does not depend on the total episode
count, the first 4000 episodes of the 6000-episode run ARE the standard
4000-episode run (verified by the weight-hash check). Every 250 episodes the
greedy convention and exact reward are snapshotted; the global RNG state is
saved and restored around each snapshot so measuring never changes training.

Run:  python experiments/q1_hypothesis_tests_k22.py
      (env overrides: GSG_NUM_SEEDS, GSG_SEED_START, GSG_WORKERS, GSG_BLOCKS)
"""

from __future__ import annotations

import csv
import hashlib
import math
import os
import random
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)

from gsg.environment.graphs import SignallingGraph
from gsg.environment.pettingzoo_env import GraphSignallingParallelEnv
from gsg.evaluation.conventions import exhaustive_joint_table, joint_signaller_convention_id
from gsg.evaluation.information_theory import guesser_info_metrics, signaller_info_metrics
from gsg.training.trainer import train_independent_iql, train_shared_iql, train_two_brain_iql

from convention_diversity_k22 import _PERFECT_CATALOGUE, signaller_rule_name

# --- configuration -----------------------------------------------------

VARIANTS = ["shared", "two_brain", "independent"]
NUM_SEEDS = int(os.environ.get("GSG_NUM_SEEDS", 200))
SEED_START = int(os.environ.get("GSG_SEED_START", 0))
NUM_BLOCKS = int(os.environ.get("GSG_BLOCKS", 5))          # for the success-rate reproducibility test
NUM_EPISODES = 4_000                                       # primary endpoint (standard protocol)
EXTENDED_EPISODES = 6_000                                  # stability window: 4000 -> 6000
EPSILON_DECAY_EPISODES = 2_500
SNAPSHOT_EVERY = 250
REPRO_SEEDS = 5                                            # seeds per variant re-run for determinism
WORKERS = int(os.environ.get("GSG_WORKERS") or os.environ.get("SLURM_CPUS_PER_TASK") or os.cpu_count() or 1)

ALPHA = 0.05
Z = 1.959963984540054
STATS_SEED = 20261027
MC_SIMS = 20_000            # Monte-Carlo null simulations
PERM_SIMS = 10_000          # permutation-test shuffles
BOOT_SIMS = 10_000          # bootstrap resamples

N_PERFECT_CONVENTIONS = 24
N_TRACKING = 8

OUT_DIR = os.path.join(HERE, "..", "results", "question1")
FIG_DIR = os.path.join(OUT_DIR, "figures", "hypothesis")
DATA_DIR = os.path.join(OUT_DIR, "hypothesis")

ITEM_LETTER = {0: "C", 1: "D"}  # CAT / DOG

# Catalogue order for plotting/tables: the 8 tracking conventions, then the 16 XOR-type.
CATALOGUE_ORDER = sorted(_PERFECT_CATALOGUE, key=lambda c: (_PERFECT_CATALOGUE[c][1] != "tracking",
                                                            _PERFECT_CATALOGUE[c][0]))

# Single-signaller rules used by the symmetry tests.
TRACK = {12: ("x0", True), 3: ("x0", False), 10: ("x1", True), 5: ("x1", False)}  # id -> (item, ON=DOG?)
PARITY = {6: "XOR", 9: "XNOR"}


def make_env(seed=None) -> GraphSignallingParallelEnv:
    return GraphSignallingParallelEnv(SignallingGraph.complete_bipartite(2, 2), seed=seed)


# --- one training run (executed in a worker process) --------------------

def _weights_hash(signaller_agents, guesser_agents) -> str:
    h = hashlib.sha256()
    for role, agents in (("S", signaller_agents), ("G", guesser_agents)):
        for k in sorted(agents):
            for name, tensor in agents[k].network.state_dict().items():
                h.update(f"{role}{k}.{name}".encode())
                h.update(tensor.detach().cpu().numpy().tobytes())
    return h.hexdigest()[:16]


def _rule_table(outcomes) -> str:
    """Exact 4-world rule, e.g. 'CD>01>CD ok' = items (G0=CAT,G1=DOG) ->
    lights (S0=OFF,S1=ON) -> guesses (CAT,DOG), both correct."""
    parts = []
    for o in outcomes:
        items = "".join(ITEM_LETTER[o.items[g]] for g in sorted(o.items))
        lights = "".join(str(o.signals[s]) for s in sorted(o.signals))
        guesses = "".join(ITEM_LETTER[o.guesses[g]] for g in sorted(o.guesses))
        mark = "ok" if o.reward == 1.0 else ("XX" if o.reward == 0.0 else "x?")
        parts.append(f"{items}>{lights}>{guesses} {mark}")
    return " | ".join(parts)


def run_one(job):
    """Train one population; return a picklable record.

    ``job = (variant, seed, total_episodes, take_snapshots)``.
    """
    variant, seed, total_episodes, take_snapshots = job
    torch.set_num_threads(1)
    t0 = time.time()
    env = make_env(seed=seed)
    graph, n_items = env.graph, env.item_space.n
    snapshots = []      # (episode, conv_id, exact greedy reward)
    primary = {}

    def measure(episode, sig, gue):
        # act() draws random.random() even at epsilon = 0, so the global RNG
        # state is saved and restored: measuring must not alter training.
        py_state, torch_state = random.getstate(), torch.get_rng_state()
        try:
            outcomes = exhaustive_joint_table(graph, n_items, sig, gue)
            conv = joint_signaller_convention_id(graph, n_items, sig)
        finally:
            random.setstate(py_state)
            torch.set_rng_state(torch_state)
        reward = sum(o.reward for o in outcomes) / len(outcomes)
        snapshots.append((episode, conv, reward))
        if episode == NUM_EPISODES:
            sinfo = signaller_info_metrics(graph, outcomes)
            ginfo = guesser_info_metrics(graph, outcomes)
            primary.update(
                conv=conv, reward=reward, rule=_rule_table(outcomes), hash=_weights_hash(sig, gue),
                H_M={s: sinfo[s].channel_utilisation for s in graph.signallers},
                ER={g: ginfo[g].entropy_reduction for g in graph.guessers},
            )

    def on_log(episodes_so_far, _rewards, *rest):
        measure(episodes_so_far, rest[-2], rest[-1])

    trainer = {"shared": train_shared_iql, "two_brain": train_two_brain_iql,
               "independent": train_independent_iql}[variant]
    kwargs = dict(num_episodes=total_episodes, epsilon_decay_episodes=EPSILON_DECAY_EPISODES, seed=seed)
    if take_snapshots:
        kwargs.update(log_every=SNAPSHOT_EVERY, on_log=on_log)
    out = trainer(env, **kwargs)
    sig, gue, history = out[-3], out[-2], out[-1]
    if not take_snapshots:
        measure(total_episodes, sig, gue)

    last = history.episode_rewards[max(0, NUM_EPISODES - 500):NUM_EPISODES]
    return dict(
        variant=variant, seed=seed, snapshots=snapshots, final_hash=_weights_hash(sig, gue),
        train_reward_last500=sum(last) / len(last) if last else float("nan"),
        seconds=time.time() - t0, **primary,
    )


# --- per-run derived quantities ---------------------------------------

def derive(rec) -> dict:
    """Add lock-in / stability fields computed from the snapshot trajectory."""
    snaps = rec["snapshots"]
    primary_snaps = [s for s in snaps if s[0] <= NUM_EPISODES]
    later = [s for s in snaps if s[0] > NUM_EPISODES]
    conv, reward = rec["conv"], rec["reward"]
    rec["perfect"] = reward == 1.0
    rec["catalogue"] = _PERFECT_CATALOGUE.get(conv, (None, None)) if rec["perfect"] else (None, None)

    rec["first_perfect_ep"] = next((e for e, _c, r in primary_snaps if r == 1.0), None)
    lockin = None
    if rec["perfect"]:
        for e, c, r in reversed(primary_snaps):
            if c == conv and r == 1.0:
                lockin = e
            else:
                break
    rec["lockin_ep"] = lockin
    rec["n_changes"] = sum(1 for a, b in zip(snaps, snaps[1:]) if a[1] != b[1])
    rec["n_changes_after_decay"] = sum(1 for a, b in zip(snaps, snaps[1:])
                                       if a[0] >= EPSILON_DECAY_EPISODES and a[1] != b[1])
    if later:
        rec["conv_ext"], rec["reward_ext"] = later[-1][1], later[-1][2]
        rec["stable_ext"] = all(c == conv and r == reward for _e, c, r in later)
    else:
        rec["conv_ext"], rec["reward_ext"], rec["stable_ext"] = conv, reward, True
    return rec


def band_name(reward: float) -> str:
    r = round(reward, 3)
    return {1.0: "perfect (1.000)", 0.875: "joint-channel (0.875)",
            0.75: "single-bit (0.750)", 0.5: "floor (0.500)"}.get(r, f"other ({r:.3f})")


def conv_label(conv) -> str:
    if conv in _PERFECT_CATALOGUE:
        num, kind = _PERFECT_CATALOGUE[conv]
        return f"#{num:>2}/24 {kind:8s}"
    return "not perfect     "


def trajectory_string(snaps) -> str:
    """Change points only: 'ep250 (12,1) .875 -> ep1500 (12,10) 1.000 ...'."""
    out, prev = [], None
    for e, c, r in snaps:
        if (c, r) != prev:
            out.append(f"ep{e} ({c[0]},{c[1]}) {r:.3f}")
            prev = (c, r)
    return " -> ".join(out)


# --- statistics (pure numpy/stdlib: no scipy dependency on the cluster) --

def wilson(k, n, z=Z):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def _binom_pmf(k, n, p):
    if p <= 0:
        return 1.0 if k == 0 else 0.0
    if p >= 1:
        return 1.0 if k == n else 0.0
    return math.exp(math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
                    + k * math.log(p) + (n - k) * math.log(1 - p))


def _binom_cdf(k, n, p):
    return sum(_binom_pmf(i, n, p) for i in range(0, k + 1))


def _bisect(f, lo=0.0, hi=1.0, iters=80):
    """Root of a monotone f on [lo, hi]."""
    f_lo = f(lo)
    for _ in range(iters):
        mid = (lo + hi) / 2
        if (f(mid) > 0) == (f_lo > 0):
            lo, f_lo = mid, f(mid)
        else:
            hi = mid
    return (lo + hi) / 2


def clopper_pearson(k, n, alpha=ALPHA):
    if n == 0:
        return (float("nan"), float("nan"))
    lo = 0.0 if k == 0 else _bisect(lambda p: (1 - _binom_cdf(k - 1, n, p)) - alpha / 2)
    hi = 1.0 if k == n else _bisect(lambda p: _binom_cdf(k, n, p) - alpha / 2)
    return (lo, hi)


def binom_test(k, n, p0):
    """Exact two-sided binomial test (sum of outcomes no more likely than k)."""
    if n == 0:
        return float("nan")
    pk = _binom_pmf(k, n, p0)
    return min(1.0, sum(_binom_pmf(i, n, p0) for i in range(n + 1) if _binom_pmf(i, n, p0) <= pk * (1 + 1e-7)))


def fisher_exact(a, b, c, d):
    """Two-sided Fisher exact test on [[a, b], [c, d]]."""
    r1, c1, n = a + b, a + c, a + b + c + d

    def p_of(x):
        return math.exp(math.lgamma(r1 + 1) - math.lgamma(x + 1) - math.lgamma(r1 - x + 1)
                        + math.lgamma(n - r1 + 1) - math.lgamma(c1 - x + 1) - math.lgamma(n - r1 - c1 + x + 1)
                        - (math.lgamma(n + 1) - math.lgamma(c1 + 1) - math.lgamma(n - c1 + 1)))

    lo, hi = max(0, c1 - (n - r1)), min(r1, c1)
    p_obs = p_of(a)
    return min(1.0, sum(p_of(x) for x in range(lo, hi + 1) if p_of(x) <= p_obs * (1 + 1e-7)))


def entropy_from_counts(counts) -> float:
    counts = np.asarray([c for c in counts if c > 0], dtype=float)
    if counts.size == 0:
        return 0.0
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())


def _row_entropies(mat) -> np.ndarray:
    p = mat / mat.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(p > 0, p * np.log2(p), 0.0)
    return -terms.sum(axis=1)


def uniform_null(n, K, rng, sims=MC_SIMS) -> dict:
    """Monte-Carlo null distributions of (chi2, plug-in entropy, max count,
    distinct count) when n runs are spread uniformly over K conventions."""
    draws = rng.multinomial(n, [1.0 / K] * K, size=sims)
    e = n / K
    return dict(chi2=((draws - e) ** 2 / e).sum(axis=1), H=_row_entropies(draws),
                max=draws.max(axis=1), distinct=(draws > 0).sum(axis=1))


def mc_pvalue(null_values, observed, greater=True) -> float:
    hits = np.sum(null_values >= observed - 1e-12) if greater else np.sum(null_values <= observed + 1e-12)
    return float((1 + hits) / (1 + len(null_values)))


def chi2_stat(table) -> float:
    table = np.asarray(table, dtype=float)
    rows, cols = table.sum(axis=1, keepdims=True), table.sum(axis=0, keepdims=True)
    expected = rows * cols / table.sum()
    mask = expected > 0
    return float((((table - expected) ** 2)[mask] / expected[mask]).sum())


def permutation_homogeneity(groups, categories, rng, sims=PERM_SIMS):
    """Permutation chi-square test that ``categories`` has the same
    distribution in every group. Returns (chi2_observed, p_value, table)."""
    g_vals, c_vals = sorted(set(groups)), sorted(set(categories), key=str)
    gi = np.array([g_vals.index(g) for g in groups])
    ci = np.array([c_vals.index(c) for c in categories])

    def table_of(cidx):
        t = np.zeros((len(g_vals), len(c_vals)))
        np.add.at(t, (gi, cidx), 1)
        return t

    observed_table = table_of(ci)
    obs = chi2_stat(observed_table)
    if len(g_vals) < 2 or len(c_vals) < 2:
        return obs, 1.0, observed_table
    hits = 0
    for _ in range(sims):
        hits += chi2_stat(table_of(rng.permutation(ci))) >= obs - 1e-12
    return obs, (1 + hits) / (1 + sims), observed_table


def holm(pvalues: dict) -> dict:
    """Holm-Bonferroni adjusted p-values for a family {name: p}."""
    items = sorted(((p, k) for k, p in pvalues.items() if not math.isnan(p)))
    m, adjusted, running = len(items), {}, 0.0
    for i, (p, k) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        adjusted[k] = running
    for k, p in pvalues.items():
        adjusted.setdefault(k, float("nan"))
    return adjusted


def bootstrap_ci(values, stat, rng, sims=BOOT_SIMS):
    values = np.asarray(values)
    if values.size == 0:
        return (float("nan"), float("nan"))
    idx = rng.integers(0, values.size, size=(sims, values.size))
    stats = np.array([stat(values[row]) for row in idx])
    return (float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5)))


def chao1(counts) -> float:
    """Bias-corrected Chao1 richness estimate."""
    counts = [c for c in counts if c > 0]
    n, s = sum(counts), len(counts)
    f1, f2 = sum(1 for c in counts if c == 1), sum(1 for c in counts if c == 2)
    if n == 0:
        return 0.0
    return s + ((n - 1) / n) * f1 * (f1 - 1) / (2 * (f2 + 1))


def accumulation_curve(labels, rng, perms=1000):
    """Mean and 95% band of distinct-conventions-seen after k runs, over random orderings."""
    labels = list(labels)
    n = len(labels)
    curves = np.zeros((perms, n))
    for p in range(perms):
        seen = set()
        for i, j in enumerate(rng.permutation(n)):
            seen.add(labels[j])
            curves[p, i] = len(seen)
    return curves.mean(axis=0), np.percentile(curves, 2.5, axis=0), np.percentile(curves, 97.5, axis=0)


# --- formatting helpers ----------------------------------------------

def fmt_ci(lo, hi, pct=True) -> str:
    return f"[{100 * lo:.1f}%, {100 * hi:.1f}%]" if pct else f"[{lo:.3f}, {hi:.3f}]"


def fmt_p(p) -> str:
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "   n/a"
    return f"{p:.4f}" if p >= 1e-4 else f"{p:.1e}"


def verdict(p_adj) -> str:
    if p_adj is None or (isinstance(p_adj, float) and math.isnan(p_adj)):
        return "not testable (n too small)"
    return "REJECT H0" if p_adj < ALPHA else "fail to reject H0"


SUMMARY_ROWS = []   # (variant, metric, value, ci_low, ci_high, p_value, note) -> CSV for the write-up


def summary(variant, metric, value, lo=None, hi=None, p=None, note=""):
    SUMMARY_ROWS.append(dict(variant=variant, metric=metric, value=value, ci_low=lo, ci_high=hi,
                             p_value=p, note=note))


# --- analysis of one variant (or the pooled set) ----------------------

def analyse(name, recs, rng) -> dict:
    n = len(recs)
    perfect = [r for r in recs if r["perfect"]]
    k = len(perfect)
    labels = [r["conv"] for r in perfect]
    counter = Counter(labels)
    res = dict(name=name, n=n, k=k, counter=counter, labels=labels)

    print(f"\n{'#' * 78}\n# ANALYSIS: {name}   (N = {n} runs, {k} perfect)\n{'#' * 78}")

    # ---------------- Q-C: success rate ----------------
    print("\n[Q-C] SUCCESS RATE -- is it the same every time?")
    w, cp = wilson(k, n), clopper_pearson(k, n)
    print(f"  convergence rate (R = 1.000 at ep {NUM_EPISODES}) : {k}/{n} = {100 * k / n:.1f}%")
    print(f"    Wilson 95% CI          : {fmt_ci(*w)}   (half-width {100 * (w[1] - w[0]) / 2:.1f} pp)")
    print(f"    Clopper-Pearson 95% CI : {fmt_ci(*cp)}  (exact, conservative)")
    summary(name, "convergence_rate", k / n, w[0], w[1], note=f"{k}/{n}; Wilson 95% CI")
    res["rate"], res["rate_ci"] = k / n, w

    rewards = np.array([r["reward"] for r in recs])
    mlo, mhi = bootstrap_ci(rewards, np.mean, rng)
    print(f"  mean greedy team reward : {rewards.mean():.4f}  bootstrap 95% CI [{mlo:.4f}, {mhi:.4f}]"
          f"   (sd {rewards.std(ddof=1) if n > 1 else 0:.4f})")
    summary(name, "mean_reward", float(rewards.mean()), mlo, mhi, note="bootstrap 95% CI")

    bands = Counter(band_name(r["reward"]) for r in recs)
    print("  reward-band distribution (every run falls in a discrete band, never in between):")
    for b in ["perfect (1.000)", "joint-channel (0.875)", "single-bit (0.750)", "floor (0.500)"] + \
             sorted(x for x in bands if x.startswith("other")):
        c = bands.get(b, 0)
        if c or not b.startswith("other"):
            lo, hi = wilson(c, n)
            print(f"      {b:24s}: {c:3d}/{n} = {100 * c / n:5.1f}%   Wilson 95% CI {fmt_ci(lo, hi)}")
            summary(name, f"band:{b}", c / n, lo, hi, note=f"{c}/{n}")
    res["bands"] = bands

    # reproducibility of the rate: independent blocks of seeds
    blocks = [r["block"] for r in recs]
    chi_b, p_b, tab_b = permutation_homogeneity(blocks, [r["perfect"] for r in recs], rng)
    print(f"  block reproducibility: seeds split into {len(set(blocks))} independent, disjoint blocks")
    block_rates = []
    for b in sorted(set(blocks)):
        sub = [r for r in recs if r["block"] == b]
        kb = sum(r["perfect"] for r in sub)
        lo, hi = wilson(kb, len(sub))
        seeds = f"{min(r['seed'] for r in sub)}-{max(r['seed'] for r in sub)}"
        print(f"      block {b} (seeds {seeds:>7s}): {kb:3d}/{len(sub)} = {100 * kb / len(sub):5.1f}%  {fmt_ci(lo, hi)}")
        block_rates.append(kb / len(sub))
    print(f"    H0_C1: the convergence rate is the same in every block")
    print(f"      permutation chi2 = {chi_b:.3f}, p = {fmt_p(p_b)}  -> {verdict(p_b)}"
          f"  (rate spread across blocks: {100 * min(block_rates):.0f}%-{100 * max(block_rates):.0f}%)")
    summary(name, "block_homogeneity_chi2", chi_b, p=p_b, note="H0: same convergence rate in every seed block")
    res["block_rates"], res["block_p"] = block_rates, p_b

    # convergence speed and stability
    lock = [r["lockin_ep"] for r in perfect if r["lockin_ep"] is not None]
    if lock:
        lo, hi = bootstrap_ci(lock, np.median, rng)
        q1, q3 = np.percentile(lock, [25, 75])
        print(f"  lock-in episode (final convention held from here to ep {NUM_EPISODES}), perfect runs:")
        print(f"      median {np.median(lock):.0f}  bootstrap 95% CI [{lo:.0f}, {hi:.0f}]  IQR [{q1:.0f}, {q3:.0f}]"
              f"  range [{min(lock)}, {max(lock)}]")
        summary(name, "median_lockin_episode", float(np.median(lock)), lo, hi, note="perfect runs")
    stable = sum(r["stable_ext"] for r in perfect)
    lo, hi = wilson(stable, k)
    print(f"  stability {NUM_EPISODES}->{EXTENDED_EPISODES} (perfect runs keeping the SAME convention at every"
          f" snapshot): {stable}/{k}  Wilson {fmt_ci(lo, hi)}")
    summary(name, "stability_perfect_runs", stable / k if k else float("nan"), lo, hi,
            note=f"same convention through ep {EXTENDED_EPISODES}")
    nonperf = [r for r in recs if not r["perfect"]]
    late = sum(r["reward_ext"] == 1.0 for r in nonperf)
    stuck = sum(r["stable_ext"] for r in nonperf)
    if nonperf:
        print(f"  non-perfect runs at ep {NUM_EPISODES}: {len(nonperf)};  became perfect by ep {EXTENDED_EPISODES}:"
              f" {late};  frozen in the same imperfect convention: {stuck}/{len(nonperf)}")
    summary(name, "late_convergence_4000_6000", late, note=f"of {len(nonperf)} non-perfect runs")
    res["stable"], res["lock"] = (stable, k), lock

    # ---------------- Q-A: same convention? ----------------
    print("\n[Q-A] CONVENTIONS -- does it always produce the same one? are some preferred?")
    unmatched = sum(1 for c in labels if c not in _PERFECT_CATALOGUE)
    print(f"  identifiability: {k - unmatched}/{k} perfect runs map to exactly one of the 24 catalogued"
          f" perfect conventions")
    if k == 0:
        print("  (no perfect runs -- convention tests skipped)")
        res.update(distinct=0, H=float("nan"), pvals={}, adj={})
        return res

    print(f"  empirical distribution over the 24 perfect conventions (Wilson 95% CIs; uniform = {100 / 24:.1f}%):")
    for conv in CATALOGUE_ORDER:
        c = counter.get(conv, 0)
        lo, hi = wilson(c, k)
        num, kind = _PERFECT_CATALOGUE[conv]
        bar = "#" * c
        print(f"      #{num:>2} {kind:8s} ({conv[0]:>2},{conv[1]:>2})  S0={signaller_rule_name(conv[0]):30s}"
              f" S1={signaller_rule_name(conv[1]):30s} {c:3d}  {100 * c / k:5.1f}% {fmt_ci(lo, hi)} {bar}")

    distinct = len(counter)
    dom_conv, dom_c = counter.most_common(1)[0]
    dlo, dhi = wilson(dom_c, k)
    H = entropy_from_counts(counter.values())
    H_mm = H + (distinct - 1) / (2 * k * math.log(2))
    codes = [CATALOGUE_ORDER.index(c) for c in labels]
    # Plug-in entropy is biased downward under resampling, so the percentile
    # interval sits below H; the basic (reflected) bootstrap interval corrects that.
    b_lo, b_hi = bootstrap_ci(codes, lambda a: entropy_from_counts(np.bincount(a)), rng)
    Hlo = min(max(0.0, 2 * H - b_hi), math.log2(24))
    Hhi = min(max(0.0, 2 * H - b_lo), math.log2(24))
    null = uniform_null(k, N_PERFECT_CONVENTIONS, rng)
    res.update(distinct=distinct, H=H, H_ci=(Hlo, Hhi), dom=(dom_conv, dom_c), null=null)

    print(f"\n  H0_A1 (always the same convention):")
    print(f"      distinct perfect conventions observed : {distinct}  -> "
          f"{'REJECTED deterministically (>= 2 distinct conventions observed)' if distinct > 1 else 'NOT rejected'}")
    print(f"      dominant convention: #{_PERFECT_CATALOGUE[dom_conv][0]} {dom_conv} in {dom_c}/{k} = "
          f"{100 * dom_c / k:.1f}%  Wilson 95% CI {fmt_ci(dlo, dhi)}")
    print(f"        (upper CI bound {100 * dhi:.1f}% < 100%: even the most common convention is NOT produced"
          f" every time)" if dhi < 1 else "        (CI includes 100%)")
    summary(name, "distinct_perfect_conventions", distinct, note=f"of 24; {k} perfect runs")
    summary(name, "dominant_convention_freq", dom_c / k, dlo, dhi, note=f"#{_PERFECT_CATALOGUE[dom_conv][0]} {dom_conv}")

    chi_obs = sum((counter.get(c, 0) - k / 24) ** 2 / (k / 24) for c in CATALOGUE_ORDER)
    p_chi = mc_pvalue(null["chi2"], chi_obs)
    p_H = mc_pvalue(null["H"], H, greater=False)
    p_max = mc_pvalue(null["max"], dom_c)
    p_distinct = mc_pvalue(null["distinct"], distinct, greater=False)
    print(f"\n  H0_A2 (arbitrary: uniform over all 24 perfect conventions), Monte-Carlo, {MC_SIMS} sims at n = {k}:")
    print(f"      goodness-of-fit chi2 = {chi_obs:.2f}             p = {fmt_p(p_chi)}")
    print(f"      convention entropy H = {H:.3f} bits (max log2 24 = {math.log2(24):.3f}); Miller-Madow {H_mm:.3f};"
          f" basic-bootstrap 95% CI [{Hlo:.3f}, {Hhi:.3f}]")
    print(f"        normalised H/Hmax = {H / math.log2(24):.3f};  uniform null: mean H = {null['H'].mean():.3f},"
          f" 95% range [{np.percentile(null['H'], 2.5):.3f}, {np.percentile(null['H'], 97.5):.3f}];"
          f"  P(H_null <= H_obs) = {fmt_p(p_H)}")
    print(f"      dominant count {dom_c}: uniform null 95% range of max count "
          f"[{np.percentile(null['max'], 2.5):.0f}, {np.percentile(null['max'], 97.5):.0f}];  p = {fmt_p(p_max)}")
    print(f"      distinct = {distinct}: uniform null 95% range [{np.percentile(null['distinct'], 2.5):.0f},"
          f" {np.percentile(null['distinct'], 97.5):.0f}];  P(distinct_null <= {distinct}) = {fmt_p(p_distinct)}")
    summary(name, "convention_entropy_bits", H, Hlo, Hhi, p_H, note="p = P(H_uniform <= H_obs)")
    summary(name, "convention_entropy_normalised", H / math.log2(24), note="H / log2(24)")

    # orbit split and symmetry tests
    n_track = sum(1 for c in labels if _PERFECT_CATALOGUE[c][1] == "tracking")
    tlo, thi = wilson(n_track, k)
    p_orbit = binom_test(n_track, k, N_TRACKING / N_PERFECT_CONVENTIONS)
    res["track"] = (n_track, k, tlo, thi)
    tests = {"A2 uniform over 24 (chi2)": p_chi, "A2 entropy below uniform": p_H,
             "A2 dominant above uniform": p_max, "A3 P(tracking) = 1/3": p_orbit}
    descr = {}

    track_runs = [c for c in labels if _PERFECT_CATALOGUE[c][1] == "tracking"]
    xor_runs = [c for c in labels if _PERFECT_CATALOGUE[c][1] == "XOR-type"]
    for sub, K, key in ((track_runs, 8, "A2 uniform within tracking"), (xor_runs, 16, "A2 uniform within XOR-type")):
        if len(sub) >= 2:
            sub_c = Counter(sub)
            members = [c for c in CATALOGUE_ORDER if _PERFECT_CATALOGUE[c][1] == ("tracking" if K == 8 else "XOR-type")]
            obs = sum((sub_c.get(c, 0) - len(sub) / K) ** 2 / (len(sub) / K) for c in members)
            tests[key] = mc_pvalue(uniform_null(len(sub), K, rng)["chi2"], obs)
            descr[key] = f"chi2 = {obs:.2f}, n = {len(sub)}, {len(sub_c)}/{K} members seen"

    def binary(key, successes, total, what):
        tests[key] = binom_test(successes, total, 0.5) if total else float("nan")
        lo, hi = wilson(successes, total) if total else (float("nan"),) * 2
        descr[key] = f"{what}: {successes}/{total}" + (f" = {100 * successes / total:.1f}% {fmt_ci(lo, hi)}" if total else "")

    binary("A4 tracking: S0 tracks x0 (vs x1)", sum(1 for a, b in track_runs if TRACK[a][0] == "x0"),
           len(track_runs), "S0 relays G0's item (direct wiring) rather than G1's")
    binary("A5 tracking: S0 polarity ON=DOG", sum(1 for a, b in track_runs if TRACK[a][1]), len(track_runs),
           "S0 lights ON for DOG")
    binary("A6 tracking: S1 polarity ON=DOG", sum(1 for a, b in track_runs if TRACK[b][1]), len(track_runs),
           "S1 lights ON for DOG")
    binary("A7 XOR-type: parity on S1 (vs S0)", sum(1 for a, b in xor_runs if b in PARITY), len(xor_runs),
           "the parity (XOR/XNOR) bit is carried by S1")
    dict_rule = [(a if a in TRACK else b) for a, b in xor_runs]
    par_rule = [(a if a in PARITY else b) for a, b in xor_runs]
    binary("A8 XOR-type: dictator tracks x0", sum(1 for d in dict_rule if TRACK[d][0] == "x0"), len(xor_runs),
           "the tracking signaller relays G0's item")
    binary("A9 XOR-type: dictator ON=DOG", sum(1 for d in dict_rule if TRACK[d][1]), len(xor_runs),
           "the tracking signaller lights ON for DOG")
    binary("A10 XOR-type: XOR (vs XNOR)", sum(1 for p in par_rule if PARITY[p] == "XOR"), len(xor_runs),
           "parity signaller uses XOR (ON iff items differ)")

    print(f"\n  H0_A3 (no compositional bias): tracking = {n_track}/{k} = {100 * n_track / k:.1f}%"
          f"  Wilson 95% CI {fmt_ci(tlo, thi)}  vs 33.3% if arbitrary;  exact binomial p = {fmt_p(p_orbit)}")
    summary(name, "tracking_fraction", n_track / k, tlo, thi, p_orbit, note="H0: 1/3 (uniform over 24)")

    adj = holm(tests)
    print(f"\n  Hypothesis family for '{name}' (Holm-Bonferroni corrected, alpha = {ALPHA}):")
    print(f"      {'test':38s} {'p_raw':>8s} {'p_holm':>8s}  verdict / evidence")
    for key, p in tests.items():
        print(f"      {key:38s} {fmt_p(p):>8s} {fmt_p(adj[key]):>8s}  {verdict(adj[key])}"
              + (f"   [{descr[key]}]" if key in descr else ""))
        summary(name, f"test:{key}", None, p=p, note=f"holm p = {fmt_p(adj[key])}; {descr.get(key, '')}")
    res["pvals"], res["adj"] = tests, adj

    # ---------------- Q-B: how many? ----------------
    print("\n[Q-B] HOW MANY different conventions?")
    est = chao1(counter.values())
    f1 = sum(1 for c in counter.values() if c == 1)
    coverage = 1 - f1 / k
    print(f"  observed richness S_obs          : {distinct} of 24 perfect conventions "
          f"({sum(1 for c in counter if _PERFECT_CATALOGUE[c][1] == 'tracking')} tracking,"
          f" {sum(1 for c in counter if _PERFECT_CATALOGUE[c][1] == 'XOR-type')} XOR-type)")
    print(f"  Chao1 richness estimate          : {est:.1f}  (capped at 24 -> {min(est, 24):.1f});"
          f" singletons f1 = {f1}, doubletons f2 = {sum(1 for c in counter.values() if c == 2)}")
    print(f"  Good-Turing sample coverage      : {100 * coverage:.1f}%  (=> P(next perfect run is a NEW"
          f" convention) ~ {100 * (1 - coverage):.1f}%)")
    never = [c for c in CATALOGUE_ORDER if c not in counter]
    if never:
        print(f"  never observed ({len(never)}): " + ", ".join(f"#{_PERFECT_CATALOGUE[c][0]}{c}" for c in never))
    summary(name, "chao1_richness", est, note="bias-corrected; true ceiling 24")
    summary(name, "good_turing_coverage", coverage)
    res["chao1"], res["coverage"] = est, coverage
    res["accum"] = accumulation_curve(labels, rng)
    return res


# --- figures -----------------------------------------------------------

VARIANT_COLOR = {"shared": "#2a78d6", "two_brain": "#eb6834", "independent": "#1baf7a"}
KIND_COLOR = {"tracking": "#2a78d6", "XOR-type": "#eb6834"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def make_figures(results: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(FIG_DIR, exist_ok=True)
    variants = [v for v in VARIANTS if v in results]

    # 1. convention frequency distribution per variant
    fig, axes = plt.subplots(len(variants), 1, figsize=(11, 2.8 * len(variants)), sharex=True)
    axes = np.atleast_1d(axes)
    x = np.arange(len(CATALOGUE_ORDER))
    for ax, v in zip(axes, variants):
        r = results[v]
        k = max(r["k"], 1)
        freq = np.array([r["counter"].get(c, 0) / k for c in CATALOGUE_ORDER])
        cis = np.array([wilson(r["counter"].get(c, 0), k) for c in CATALOGUE_ORDER])
        colors = [KIND_COLOR[_PERFECT_CATALOGUE[c][1]] for c in CATALOGUE_ORDER]
        ax.bar(x, freq, width=0.75, color=colors, edgecolor="white", linewidth=1)
        yerr = np.clip([freq - cis[:, 0], cis[:, 1] - freq], 0, None)
        ax.errorbar(x, freq, yerr=yerr, fmt="none", ecolor=INK2,
                    elinewidth=0.8, capsize=2)
        ax.axhline(1 / 24, color=INK2, linestyle="--", linewidth=1)
        ax.axvline(N_TRACKING - 0.5, color=INK2, linewidth=0.6)
        ax.text(len(x) - 0.5, 1 / 24, " uniform 1/24", color=INK2, fontsize=7, va="bottom", ha="right")
        title = (f"{v}: {r['k']}/{r['n']} perfect runs, {r.get('distinct', 0)} distinct conventions, "
                 f"H = {r.get('H', float('nan')):.2f} bits")
        ax.set_title(title, fontsize=9, color=INK, loc="left")
        ax.set_ylabel("share of perfect runs", fontsize=8, color=INK2)
        ax.set_xlim(-0.6, len(x) - 0.4)
        _style(ax)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels([f"#{_PERFECT_CATALOGUE[c][0]}\n{c[0]},{c[1]}" for c in CATALOGUE_ORDER], fontsize=6.5)
    from matplotlib.patches import Patch
    axes[0].legend(handles=[Patch(color=KIND_COLOR["tracking"], label="tracking (8)"),
                            Patch(color=KIND_COLOR["XOR-type"], label="XOR-type (16)")],
                   fontsize=8, frameon=False, loc="upper right")
    fig.suptitle("K$_{2,2}$ perfect-convention distribution (error bars: Wilson 95% CI)", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "convention_distribution.png"), dpi=160)
    plt.close(fig)

    # 2. success rate (per variant, with per-block points) and tracking fraction
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8))
    rate_labels, track_labels = [], []
    for i, v in enumerate(variants):
        r = results[v]
        lo, hi = r["rate_ci"]
        a1.errorbar(i, r["rate"], yerr=np.clip([[r["rate"] - lo], [hi - r["rate"]]], 0, None), fmt="o",
                    color=VARIANT_COLOR[v],
                    markersize=8, capsize=4, linewidth=2)
        a1.scatter(i + np.linspace(-0.22, 0.22, len(r["block_rates"])), r["block_rates"], s=14,
                   facecolor="white", edgecolor=VARIANT_COLOR[v], zorder=3)
        rate_labels.append(f"{v}\n{100 * r['rate']:.0f}%, blocks p = {r['block_p']:.2f}")
        track_labels.append(v)
        if "track" in r:
            nt, kk, tlo, thi = r["track"]
            a2.errorbar(i, nt / kk, yerr=np.clip([[nt / kk - tlo], [thi - nt / kk]], 0, None), fmt="o",
                        color=VARIANT_COLOR[v],
                        markersize=8, capsize=4, linewidth=2)
            p = r["adj"].get("A3 P(tracking) = 1/3", float("nan"))
            track_labels[-1] = f"{v}\n{100 * nt / kk:.0f}%, Holm p = {fmt_p(p)}"
    a1.set_title("Convergence rate (95% Wilson CI); hollow = independent seed blocks", fontsize=9, color=INK, loc="left")
    a2.set_title("Share of perfect runs that are tracking conventions", fontsize=9, color=INK, loc="left")
    a2.axhline(1 / 3, color=INK2, linestyle="--", linewidth=1)
    a2.text(len(variants) - 0.6, 1 / 3, "arbitrary (8/24)", fontsize=7, color=INK2, va="bottom", ha="right")
    for ax, labels in ((a1, rate_labels), (a2, track_labels)):
        ax.set_xticks(range(len(variants)))
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 1.05)
        ax.set_xlim(-0.6, len(variants) - 0.4)
        _style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "success_rate_and_tracking.png"), dpi=160)
    plt.close(fig)

    # 3. reward bands (ordinal -> one hue, light to dark)
    band_order = ["perfect (1.000)", "joint-channel (0.875)", "single-bit (0.750)", "floor (0.500)"]
    band_color = ["#0d3b78", "#2a78d6", "#86b6ef", "#cfe1f8"]
    fig, ax = plt.subplots(figsize=(9, 0.9 + 0.6 * len(variants)))
    for i, v in enumerate(variants):
        r, left = results[v], 0.0
        for b, col in zip(band_order, band_color):
            share = r["bands"].get(b, 0) / r["n"]
            if share:
                ax.barh(i, share, left=left, color=col, edgecolor="white", linewidth=2, height=0.6,
                        label=b if i == 0 else None)
                if share > 0.04:
                    ax.text(left + share / 2, i, f"{100 * share:.0f}%", ha="center", va="center", fontsize=7,
                            color="white" if col in band_color[:2] else INK)
                left += share
        other = 1 - left
        if other > 1e-9:
            ax.barh(i, other, left=left, color="#b9b8b2", edgecolor="white", linewidth=2, height=0.6,
                    label="other" if i == 0 else None)
    ax.set_yticks(range(len(variants)))
    ax.set_yticklabels(variants)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of runs", fontsize=8, color=INK2)
    ax.legend(fontsize=7, frameon=False, ncol=5, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    _style(ax)
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "reward_bands.png"), dpi=160)
    plt.close(fig)

    # 4. accumulation curves
    fig, ax = plt.subplots(figsize=(7, 4))
    kmax = 1
    for v in variants:
        r = results[v]
        if r["k"] == 0:
            continue
        mean, lo, hi = r["accum"]
        ks = np.arange(1, len(mean) + 1)
        kmax = max(kmax, len(mean))
        ax.fill_between(ks, lo, hi, color=VARIANT_COLOR[v], alpha=0.15, linewidth=0)
        ax.plot(ks, mean, color=VARIANT_COLOR[v], linewidth=2, label=f"{v} (observed)")
    ks = np.arange(1, kmax + 1)
    ax.plot(ks, 24 * (1 - (23 / 24) ** ks), color=INK2, linestyle="--", linewidth=1.5, label="expected if uniform over 24")
    ax.axhline(24, color=INK2, linewidth=0.6)
    ax.set_xlabel("number of perfect runs", fontsize=8, color=INK2)
    ax.set_ylabel("distinct conventions seen", fontsize=8, color=INK2)
    ax.set_title("Convention accumulation (mean and 95% band over 1000 orderings)", fontsize=9, color=INK, loc="left")
    ax.legend(fontsize=7, frameon=False)
    _style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "convention_accumulation.png"), dpi=160)
    plt.close(fig)

    # 5. lock-in ECDF
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for v in variants:
        lock = np.sort(results[v]["lock"])
        if lock.size:
            ax.step(lock, np.arange(1, lock.size + 1) / results[v]["n"], where="post", color=VARIANT_COLOR[v],
                    linewidth=2, label=v)
    ax.axvline(EPSILON_DECAY_EPISODES, color=INK2, linestyle="--", linewidth=1)
    ax.text(EPSILON_DECAY_EPISODES, 0.02, " epsilon reaches 0.05", fontsize=7, color=INK2)
    ax.set_xlim(0, NUM_EPISODES)
    ax.set_ylim(0, 1)
    ax.set_xlabel("training episode at which the final perfect convention locks in", fontsize=8, color=INK2)
    ax.set_ylabel("cumulative share of ALL runs", fontsize=8, color=INK2)
    ax.set_title("When do runs lock in their convention?", fontsize=9, color=INK, loc="left")
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    _style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "lockin_ecdf.png"), dpi=160)
    plt.close(fig)


# --- main ----------------------------------------------------------------

def write_csvs(all_recs) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    runs_path = os.path.join(DATA_DIR, "runs.csv")
    with open(runs_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variant", "seed", "block", "reward_4000", "band", "perfect", "conv_s0", "conv_s1",
                    "catalogue_num", "kind", "s0_rule", "s1_rule", "first_perfect_ep", "lockin_ep",
                    "n_conv_changes", "n_changes_after_decay", "reward_6000", "conv_6000_s0", "conv_6000_s1",
                    "stable_4000_6000", "train_reward_last500", "H_M0", "H_M1", "I_G0", "I_G1",
                    "weights_sha256_16", "rule_table"])
        for r in all_recs:
            num, kind = r["catalogue"]
            w.writerow([r["variant"], r["seed"], r["block"], f"{r['reward']:.4f}", band_name(r["reward"]),
                        int(r["perfect"]), r["conv"][0], r["conv"][1], num or "", kind or "",
                        signaller_rule_name(r["conv"][0]), signaller_rule_name(r["conv"][1]),
                        r["first_perfect_ep"] or "", r["lockin_ep"] or "", r["n_changes"],
                        r["n_changes_after_decay"], f"{r['reward_ext']:.4f}", r["conv_ext"][0], r["conv_ext"][1],
                        int(r["stable_ext"]), f"{r['train_reward_last500']:.4f}", f"{r['H_M'][0]:.4f}",
                        f"{r['H_M'][1]:.4f}", f"{r['ER'][0]:.4f}", f"{r['ER'][1]:.4f}", r["hash"], r["rule"]])
    sum_path = os.path.join(DATA_DIR, "summary.csv")
    with open(sum_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["variant", "metric", "value", "ci_low", "ci_high", "p_value", "note"])
        w.writeheader()
        w.writerows(SUMMARY_ROWS)
    print(f"\n  per-run records  -> {os.path.relpath(runs_path)}")
    print(f"  summary metrics  -> {os.path.relpath(sum_path)}")


def main() -> None:
    t_start = time.time()
    rng = np.random.default_rng(STATS_SEED)
    seeds = list(range(SEED_START, SEED_START + NUM_SEEDS))
    print("=" * 78)
    print("QUESTION 1 / OBJECTIVE 4 -- HYPOTHESIS TESTS ON K_{2,2}")
    print("=" * 78)
    print(f"variants           : {', '.join(VARIANTS)}")
    print(f"seeds per variant  : N = {NUM_SEEDS} (seeds {seeds[0]}..{seeds[-1]}), {NUM_BLOCKS} disjoint blocks")
    print(f"protocol           : {NUM_EPISODES} episodes (primary endpoint), epsilon 1.0->0.05 over"
          f" {EPSILON_DECAY_EPISODES}; continued to {EXTENDED_EPISODES} for stability;"
          f" greedy snapshot every {SNAPSHOT_EVERY}")
    print(f"statistics         : alpha = {ALPHA}, MC sims = {MC_SIMS}, permutations = {PERM_SIMS},"
          f" bootstrap = {BOOT_SIMS}, stats RNG seed = {STATS_SEED}")
    print(f"workers            : {WORKERS} processes (torch threads = 1 per worker)")
    print("\nRule-table notation: 'CD>01>CD ok' = hidden items (G0=CAT, G1=DOG) -> lights (S0=OFF, S1=ON)"
          " -> guesses (G0=CAT, G1=DOG); ok = both right, x? = one right, XX = both wrong.")
    print("Convention id (a,b): S0's and S1's rule ids, bit k = output on input (x0,x1) in order"
          " CC, CD, DC, DD.  12 = track x0, 10 = track x1, 3/5 = their negations, 6 = XOR, 9 = XNOR.")
    print(flush=True)

    jobs = [(v, s, EXTENDED_EPISODES, True) for v in VARIANTS for s in seeds]
    check_seeds = seeds[:REPRO_SEEDS]
    rerun_jobs = [(v, s, EXTENDED_EPISODES, True) for v in VARIANTS for s in check_seeds]
    plain_jobs = [(v, s, NUM_EPISODES, False) for v in VARIANTS for s in check_seeds]

    records = {v: [] for v in VARIANTS}
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        current = None
        for rec in pool.map(run_one, jobs):
            rec = derive(rec)
            rec["block"] = (rec["seed"] - SEED_START) * NUM_BLOCKS // NUM_SEEDS
            v = rec["variant"]
            if v != current:
                current = v
                print(f"\n{'=' * 78}\nPER-RUN RECORDS: {v}\n{'=' * 78}")
            records[v].append(rec)
            fp = rec["first_perfect_ep"] or "-"
            li = rec["lockin_ep"] or "-"
            print(f"  seed {rec['seed']:3d} blk{rec['block']}  R={rec['reward']:.3f}  {band_name(rec['reward']):22s}"
                  f" conv=({rec['conv'][0]:>2},{rec['conv'][1]:>2}) {conv_label(rec['conv'])}"
                  f" first_perfect={fp!s:>5} lockin={li!s:>5} ep6000:R={rec['reward_ext']:.3f}"
                  f"{' stable' if rec['stable_ext'] else ' CHANGED'}  ({rec['seconds']:.0f}s)")
            print(f"           S0: {signaller_rule_name(rec['conv'][0]):32s} S1: {signaller_rule_name(rec['conv'][1])}")
            print(f"           rule : {rec['rule']}")
            print(f"           info : H(M0)={rec['H_M'][0]:.3f} H(M1)={rec['H_M'][1]:.3f} bits;"
                  f"  I(o0;x0)={rec['ER'][0]:.3f} I(o1;x1)={rec['ER'][1]:.3f} bits;"
                  f"  train R (eps on, last 500) = {rec['train_reward_last500']:.3f}")
            print(f"           traj : {trajectory_string(rec['snapshots'])}", flush=True)

        print(f"\n{'=' * 78}\nREPRODUCIBILITY CHECKS (seeds {check_seeds[0]}..{check_seeds[-1]} of every variant)\n{'=' * 78}")
        reruns = list(pool.map(run_one, rerun_jobs))
        plains = list(pool.map(run_one, plain_jobs))

    by_key = {(r["variant"], r["seed"]): r for v in VARIANTS for r in records[v]}
    n_same_traj = n_same_hash = n_plain = 0
    for rr, pl in zip(reruns, plains):
        orig = by_key[(rr["variant"], rr["seed"])]
        same_traj = rr["snapshots"] == orig["snapshots"]
        same_hash = rr["hash"] == orig["hash"] and rr["final_hash"] == orig["final_hash"]
        plain_ok = pl["hash"] == orig["hash"] and pl["conv"] == orig["conv"] and pl["reward"] == orig["reward"]
        n_same_traj += same_traj
        n_same_hash += same_hash
        n_plain += plain_ok
        print(f"  {rr['variant']:11s} seed {rr['seed']:2d}: re-run trajectory identical={same_traj}, "
              f"weights identical={same_hash} ({orig['hash']}); plain 4000-ep run (no snapshots) "
              f"identical={plain_ok} ({pl['hash']})")
    total = len(reruns)
    print(f"\n  DETERMINISM: {n_same_traj}/{total} re-runs reproduced the full snapshot trajectory and"
          f" {n_same_hash}/{total} the exact network weights.")
    print(f"  NON-INVASIVE MEASUREMENT: {n_plain}/{total} plain (un-instrumented) {NUM_EPISODES}-episode runs gave"
          f" bit-identical weights, convention and reward at the primary endpoint.")
    summary("all", "determinism_reruns_identical", n_same_hash / total, note=f"{n_same_hash}/{total}")
    summary("all", "instrumentation_noninvasive", n_plain / total, note=f"{n_plain}/{total}")

    # per-variant analysis
    results = {v: analyse(v, records[v], rng) for v in VARIANTS}

    # cross-variant comparison
    print(f"\n{'#' * 78}\n# CROSS-VARIANT COMPARISON (is the answer the same for every parameter-sharing scheme?)\n{'#' * 78}")
    all_recs = [r for v in VARIANTS for r in records[v]]
    chi, p_rate, _ = permutation_homogeneity([r["variant"] for r in all_recs], [r["perfect"] for r in all_recs], rng)
    print(f"  H0_C2: equal convergence rate across variants: permutation chi2 = {chi:.3f}, p = {fmt_p(p_rate)}"
          f"  -> {verdict(p_rate)}")
    summary("all", "variant_rate_homogeneity", chi, p=p_rate)
    pair_p = {}
    for i, a in enumerate(VARIANTS):
        for b in VARIANTS[i + 1:]:
            ka, na, kb, nb = results[a]["k"], results[a]["n"], results[b]["k"], results[b]["n"]
            pair_p[f"{a} vs {b}"] = fisher_exact(ka, na - ka, kb, nb - kb)
    pair_adj = holm(pair_p)
    for key, p in pair_p.items():
        a, b = key.split(" vs ")
        print(f"      {key:26s}: {100 * results[a]['rate']:.1f}% vs {100 * results[b]['rate']:.1f}%   Fisher p = {fmt_p(p)},"
              f" Holm p = {fmt_p(pair_adj[key])}  -> {verdict(pair_adj[key])}")
        summary("all", f"fisher_rate:{key}", None, p=p, note=f"holm p = {fmt_p(pair_adj[key])}")

    perfect_all = [r for r in all_recs if r["perfect"]]
    chi, p_kind, tab = permutation_homogeneity([r["variant"] for r in perfect_all],
                                               [r["catalogue"][1] for r in perfect_all], rng)
    print(f"  H0_A11: same tracking/XOR split across variants: chi2 = {chi:.3f}, p = {fmt_p(p_kind)}  -> {verdict(p_kind)}")
    chi, p_conv, _ = permutation_homogeneity([r["variant"] for r in perfect_all],
                                             [r["conv"] for r in perfect_all], rng)
    print(f"  H0_A12: same 24-way convention distribution across variants: chi2 = {chi:.3f}, p = {fmt_p(p_conv)}"
          f"  -> {verdict(p_conv)}")
    summary("all", "variant_kind_homogeneity", None, p=p_kind)
    summary("all", "variant_convention_homogeneity", None, p=p_conv)

    results_pooled = analyse("POOLED (all variants)", all_recs, rng)
    print("  (pooled analysis is only interpretable as 'IQL in general' if H0_A12 above is not rejected)")

    # final answers
    print(f"\n{'=' * 78}\nANSWERS TO THE THREE QUESTIONS (alpha = {ALPHA}; 95% CIs)\n{'=' * 78}")
    for v in VARIANTS + [None]:
        r = results[v] if v else results_pooled
        name = v or "POOLED"
        if r["k"] == 0:
            print(f"\n  {name}: no perfect runs.")
            continue
        dom_conv, dom_c = r["dom"]
        nt, kk, tlo, thi = r["track"]
        print(f"\n  {name}:")
        print(f"   Q-A same convention every time?  NO -- {r['distinct']} distinct perfect conventions in {r['k']} perfect"
              f" runs; the most common (#{_PERFECT_CATALOGUE[dom_conv][0]} {dom_conv}) occurs in"
              f" {100 * dom_c / r['k']:.1f}% {fmt_ci(*wilson(dom_c, r['k']))}." if r["distinct"] > 1 else
              f"   Q-A same convention every time?  YES -- all {r['k']} perfect runs share {dom_conv}.")
        print(f"       arbitrary (uniform over 24)? chi2 p_holm = {fmt_p(r['adj'].get('A2 uniform over 24 (chi2)'))};"
              f" tracking {100 * nt / kk:.1f}% {fmt_ci(tlo, thi)} vs 33.3%, p_holm = "
              f"{fmt_p(r['adj'].get('A3 P(tracking) = 1/3'))}; H = {r['H']:.3f} bits"
              f" [{r['H_ci'][0]:.3f}, {r['H_ci'][1]:.3f}] of {math.log2(24):.3f} max.")
        print(f"   Q-B how many?                    {r['distinct']} observed of 24; Chao1 {min(r['chao1'], 24):.1f};"
              f" coverage {100 * r['coverage']:.1f}%.")
        print(f"   Q-C same success rate?           {100 * r['rate']:.1f}% {fmt_ci(*r['rate_ci'])};"
              + (f" block homogeneity p = {fmt_p(r['block_p'])} ({verdict(r['block_p'])});" if v else "")
              + f" bands: " + ", ".join(f"{b.split(' ')[0]} {c}" for b, c in sorted(r["bands"].items())))
    print(f"\n  Reproducibility of the measurement itself: {n_same_hash}/{total} same-seed re-runs bit-identical;"
          f" {n_plain}/{total} un-instrumented runs identical to instrumented ones.")

    write_csvs(all_recs)
    try:
        make_figures(results)
        print(f"  figures          -> {os.path.relpath(FIG_DIR)}/")
    except Exception:  # never lose a finished run's log/CSVs to a plotting problem
        import traceback
        traceback.print_exc()
        print("  FIGURES FAILED (log and CSVs above are complete)")
    print(f"\nq1_hypothesis_tests_k22.py complete in {(time.time() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
