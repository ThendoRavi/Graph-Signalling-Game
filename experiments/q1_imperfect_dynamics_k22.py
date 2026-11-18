"""What do imperfect K_{2,2} runs look like, and do they plateau or oscillate?

Post-processes the log written by q1_hypothesis_tests_k22.py (q1_hypothesis.sh),
which records, for every run, the greedy signaller convention and exact team
reward at every 250-episode snapshot where the convention changed ("traj :"
lines), up to episode 6,000. No training is repeated.

Reports:
  * the composition of imperfect conventions at episode 4,000, by reward band:
    which kinds of signaller rule (tracking, one-entry hybrid, parity, constant)
    are paired together;
  * how often the greedy convention changes in each 250-episode window, for
    perfect runs, imperfect runs that later reach R = 1.0, and imperfect runs
    that never do;
  * a classification of the imperfect runs over episodes 4,000 -> 6,000:
    escaped to perfect / frozen (no change) / oscillating (returns to a
    convention it held before) / drifting (changes without returning);
  * how large each convention change is (signaller table entries flipped) and
    whether the team reward moves when the convention does.

Run:  python experiments/q1_imperfect_dynamics_k22.py <path/to/q1_hypothesis log>
Output: results/question1/figures/hypothesis/imperfect_dynamics.png
"""

from __future__ import annotations

import collections
import os
import re
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "question1", "figures", "hypothesis")

WINDOW = 250
END_EP = 6_000
EPS_FLOOR_EP = 2_500
PRIMARY_EP = 4_000

TRACKING = {3, 5, 10, 12}
PARITY = {6, 9}
CONSTANT = {0, 15}

INK = "#333333"
COLORS = {"perfect": "#2a6fd6", "escaped": "#19a974", "stuck": "#e8673a"}


def rule_kind(rid: int) -> str:
    if rid in TRACKING:
        return "tracking"
    if rid in PARITY:
        return "parity"
    if rid in CONSTANT:
        return "constant"
    return "hybrid"  # one table entry away from two tracking rules (AND/OR-type)


def parse_log(path):
    """Return one dict per run: variant, seed, R (ep 4000), R6000, conv, traj."""
    runs, variant, cur = [], None, None
    for line in open(path, encoding="utf-8", errors="replace"):
        m = re.match(r"PER-RUN RECORDS: (\w+)", line.strip())
        if m:
            variant = m.group(1)
            continue
        if variant is None:
            continue
        m = re.match(r"\s+seed\s+(\d+) blk\d\s+R=([\d.]+)", line)
        if m:
            conv = re.search(r"conv=\(\s*(\d+),\s*(\d+)\)", line)
            r6 = re.search(r"ep6000:R=([\d.]+)", line)
            cur = {"variant": variant, "seed": int(m.group(1)), "R": float(m.group(2)),
                   "conv": (int(conv.group(1)), int(conv.group(2))), "R6000": float(r6.group(1))}
            runs.append(cur)
            continue
        m = re.match(r"\s+traj : (.*)", line)
        if m and cur is not None and "traj" not in cur:
            cur["traj"] = [(int(e), (int(a), int(b)), float(r))
                           for e, a, b, r in re.findall(r"ep(\d+) \((\d+),\s*(\d+)\) ([\d.]+)", m.group(1))]
    return [r for r in runs if "traj" in r]


def state_at(traj, ep):
    s = None
    for e, c, r in traj:
        if e <= ep:
            s = (c, r)
    return s


def hamming(a, b):
    return bin(a ^ b).count("1")


def classify(run):
    """Episode 4000 -> 6000 behaviour of an imperfect run."""
    if run["R6000"] >= 0.999:
        return "escaped"
    start = state_at(run["traj"], PRIMARY_EP)
    seq = [start[0]] + [c for e, c, _ in run["traj"] if e > PRIMARY_EP]
    if len(seq) == 1:
        return "frozen"
    if any(c in seq[:k] for k, c in enumerate(seq)):
        return "oscillating"
    return "drifting"


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    runs = parse_log(sys.argv[1])
    print(f"parsed {len(runs)} runs: {dict(collections.Counter(r['variant'] for r in runs))}")
    perfect = [r for r in runs if r["R"] >= 0.999]
    imperfect = [r for r in runs if r["R"] < 0.999]
    escaped = [r for r in imperfect if r["R6000"] >= 0.999]
    stuck = [r for r in imperfect if r["R6000"] < 0.999]

    # --- 1. what imperfect conventions look like at episode 4000 ---
    print(f"\n1. IMPERFECT CONVENTIONS AT EPISODE {PRIMARY_EP} ({len(imperfect)} runs)")
    for band in (0.875, 0.75):
        sub = [r for r in imperfect if abs(r["R"] - band) < 1e-6]
        kinds = collections.Counter(tuple(sorted(rule_kind(c) for c in r["conv"])) for r in sub)
        print(f"   R = {band}: {len(sub)} runs")
        for k, v in kinds.most_common():
            print(f"      {v:3d} ({100 * v / len(sub):4.1f}%)  {' + '.join(k)}")
        dup = sum(1 for r in sub if all(c in TRACKING for c in r["conv"])
                  and len({c in (3, 12) for c in r["conv"]}) == 1)
        print(f"      of which both signallers track the SAME item: {dup}")

    # --- 2. convention changes per window ---
    edges = np.arange(WINDOW, END_EP + 1, WINDOW)

    def change_rate(group):
        counts = np.zeros(len(edges))
        for r in group:
            for e, _c, _r in r["traj"][1:]:
                counts[np.searchsorted(edges, e)] += 1
        return counts / max(len(group), 1)

    rates = {"perfect": change_rate(perfect), "escaped": change_rate(escaped), "stuck": change_rate(stuck)}
    print(f"\n2. CONVENTION CHANGES PER RUN PER {WINDOW}-EPISODE WINDOW (after epsilon floor, ep > {EPS_FLOOR_EP})")
    post = edges > EPS_FLOOR_EP
    for k, v in rates.items():
        print(f"   {k:8s}: mean {v[post].mean():.3f} changes/window "
              f"(ep 2500-4000: {v[(edges > EPS_FLOOR_EP) & (edges <= PRIMARY_EP)].mean():.3f}, "
              f"ep 4000-6000: {v[edges > PRIMARY_EP].mean():.3f})")

    # --- 3. episode 4000 -> 6000 classification ---
    cls = collections.Counter(classify(r) for r in imperfect)
    print(f"\n3. IMPERFECT RUNS, EPISODE {PRIMARY_EP} -> {END_EP} ({len(imperfect)} runs)")
    for k in ("escaped", "frozen", "oscillating", "drifting"):
        sub = [r for r in imperfect if classify(r) == k]
        bands = collections.Counter(r["R"] for r in sub)
        print(f"   {k:11s}: {cls[k]:3d} ({100 * cls[k] / len(imperfect):4.1f}%)  "
              f"by ep-4000 band: " + ", ".join(f"{b}: {n}" for b, n in sorted(bands.items(), reverse=True)))
    osc = [r for r in imperfect if classify(r) == "oscillating"]
    two_state = sum(1 for r in osc if len({state_at(r["traj"], PRIMARY_EP)[0]}
                                         | {c for e, c, _ in r["traj"] if e > PRIMARY_EP}) == 2)
    print(f"   oscillating runs that alternate between exactly two conventions: {two_state}/{len(osc)}")

    # --- 4. size of each change and reward movement (stuck runs, after epsilon floor) ---
    steps, same_reward = [], 0
    for r in stuck:
        prev = state_at(r["traj"], EPS_FLOOR_EP)
        for e, c, rw in r["traj"]:
            if e <= EPS_FLOOR_EP:
                continue
            steps.append(hamming(prev[0][0], c[0]) + hamming(prev[0][1], c[1]))
            same_reward += abs(rw - prev[1]) < 1e-9
            prev = (c, rw)
    steps = np.array(steps)
    print(f"\n4. CONVENTION CHANGES IN NEVER-PERFECT RUNS AFTER EPISODE {EPS_FLOOR_EP} ({len(steps)} changes, {len(stuck)} runs)")
    print(f"   one signaller-table entry flipped: {np.sum(steps == 1)} ({100 * np.mean(steps == 1):.0f}%), "
          f"two: {np.sum(steps == 2)}, three or more: {np.sum(steps >= 3)}")
    print(f"   team reward unchanged by the change: {same_reward} ({100 * same_reward / len(steps):.0f}%)")
    const_reward = sum(1 for r in stuck
                       if len({rw for e, _c, rw in r["traj"] if e > EPS_FLOOR_EP}
                              | {state_at(r["traj"], EPS_FLOOR_EP)[1]}) == 1)
    print(f"   never-perfect runs whose reward never changes after episode {EPS_FLOOR_EP}: "
          f"{const_reward}/{len(stuck)}")

    # --- figure ---
    os.makedirs(FIG_DIR, exist_ok=True)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.3), gridspec_kw={"width_ratios": [1.6, 1]})
    centres = edges - WINDOW / 2
    labels = {"perfect": f"perfect by ep 4000 (n={len(perfect)})",
              "escaped": f"perfect only by ep 6000 (n={len(escaped)})",
              "stuck": f"never perfect (n={len(stuck)})"}
    # the first snapshot (episode 250) is the starting state, not a change, so
    # the first window is omitted
    for k in ("stuck", "escaped", "perfect"):
        a1.plot(centres[1:], rates[k][1:], drawstyle="steps-mid", lw=2, color=COLORS[k], label=labels[k])
    a1.set_ylim(0, 1.08)
    for x, t in ((EPS_FLOOR_EP, "epsilon reaches 0.05"), (PRIMARY_EP, "primary endpoint")):
        a1.axvline(x, ls="--", lw=1, color=INK)
        a1.text(x - 50, 1.05, t, fontsize=8, color=INK, rotation=90, ha="right", va="top")
    a1.set_xlabel("training episode")
    a1.set_ylabel(f"convention changes per run\nper {WINDOW}-episode window")
    a1.set_title("Do runs keep changing convention after exploration ends?", fontsize=10, loc="left")
    a1.legend(fontsize=8, frameon=False, loc="upper right")
    a1.set_xlim(WINDOW, END_EP)

    order = ["escaped", "frozen", "oscillating", "drifting"]
    cmap = {"escaped": "#19a974", "frozen": "#8c8c8c", "oscillating": "#e8673a", "drifting": "#f2b134"}
    left = 0
    for k in order:
        a2.barh([0], [cls[k]], left=left, color=cmap[k], edgecolor="white")
        if cls[k] >= 15:
            a2.text(left + cls[k] / 2, 0, f"{k}\n{cls[k]}", ha="center", va="center", fontsize=8,
                    color="white")
        elif cls[k]:  # too narrow to hold its label: put it above the bar
            a2.text(left + cls[k] / 2, 0.47, f"{k} {cls[k]}", ha="center", va="bottom", fontsize=8,
                    color=INK)
        left += cls[k]
    a2.set_xlim(0, len(imperfect))
    a2.set_ylim(-0.6, 0.75)
    a2.set_yticks([])
    a2.set_xlabel(f"imperfect runs at episode {PRIMARY_EP} (n={len(imperfect)})")
    a2.set_title(f"What imperfect runs do, episode {PRIMARY_EP} to {END_EP}", fontsize=10, loc="left")
    for ax in (a1, a2):
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    p = os.path.join(FIG_DIR, "imperfect_dynamics.png")
    fig.savefig(p, dpi=150)
    print(f"\nsaved -> {os.path.normpath(p)}")


if __name__ == "__main__":
    main()
